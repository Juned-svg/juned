# sai_portal.py
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import smtplib
from email.message import EmailMessage
import os
from typing import List, Tuple, Optional

# ---------- Mediapipe setup ----------
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# ---------- Utility functions ----------
def calculate_angle(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
    """
    Calculate angle ABC (in degrees) using vector dot product.
    a, b, c are (x, y) points.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    # Protect against division by zero
    if np.linalg.norm(ba) == 0 or np.linalg.norm(bc) == 0:
        return 0.0

    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    # numerical errors might push cos_angle slightly out of [-1,1]
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_angle))
    return angle

def detect_cheating(landmarks, test_type: str) -> bool:
    """
    Basic cheat detection: if required joints have low visibility it's flagged.
    """
    required = {
        "pushup": [mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.RIGHT_ELBOW],
        "jump": [mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_ANKLE],
        "sprint": [mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.RIGHT_KNEE],
        "situp": [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP],
        "curl": [mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.RIGHT_ELBOW]
    }
    missing = 0
    for joint in required.get(test_type, []):
        lm = landmarks[joint.value]
        if getattr(lm, "visibility", 0) < 0.5:
            missing += 1
    return missing > 0

def estimate_jump_height(landmarks, frame_height: int) -> float:
    """
    Estimate jump height based on left ankle y position.
    Returns height in pixels measured from bottom (higher = bigger).
    """
    try:
        ankle_y = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * frame_height
        return round(frame_height - ankle_y, 2)
    except Exception:
        return 0.0

def estimate_sprint_frequency(ankle_history: List[float], fps: float) -> float:
    """
    Rough cycles per second estimate using ankle y-position oscillations.
    """
    if fps <= 0 or len(ankle_history) < 2:
        return 0.0
    diffs = [abs(ankle_history[i] - ankle_history[i-1]) for i in range(1, len(ankle_history))]
    avg_diff = sum(diffs) / len(diffs)
    # Normalize by fps to get rough cycles/sec-like number
    return round(avg_diff * fps, 2)

def send_email_to_sai(
    test_type: str,
    score,
    confidence: float,
    cheat_flag: bool,
    feedback: List[str],
    video_path: str,
    sender_email: str,
    sender_password: str,
    recipient: str = "sai_official@example.com",
):
    """
    Sends a simple email with basic fields and attaches the video file.
    NOTE: For Gmail use an App Password and allow SMTP (depends on account).
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError("Video file not found for attachment.")

    msg = EmailMessage()
    msg["Subject"] = f"SAI Fitness Report - {test_type}"
    msg["From"] = sender_email
    msg["To"] = recipient

    body = (
        f"Test Type: {test_type}\n"
        f"Score: {score}\n"
        f"Confidence: {confidence}\n"
        f"Cheat Detected: {cheat_flag}\n"
        f"Feedback: {'; '.join(feedback)}\n"
    )
    msg.set_content(body)

    with open(video_path, "rb") as f:
        file_data = f.read()
        # Set subtype mp4; if different extension, adjust
        msg.add_attachment(file_data, maintype="video", subtype="mp4", filename=os.path.basename(video_path))

    # Send via gmail secure SMTP
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(sender_email, sender_password)
        smtp.send_message(msg)

# ---------- Streamlit UI ----------
st.set_page_config(page_title="SAI Athlete Performance Portal", layout="wide")
st.title("🏋️‍♂️ SAI Athlete Performance Portal")
st.write("Upload a recorded attempt and get reps / metrics + optional report sending.")

col1, col2 = st.columns([1, 1])

with col1:
    test_type = st.selectbox("Choose Test Type", ["curl", "pushup", "situp", "jump", "sprint"])
    age = st.slider("Athlete Age", 10, 60, 25)
    gender = st.selectbox("Gender", ["male", "female", "other"])
    video_file = st.file_uploader("Upload Video (mp4 / mov)", type=["mp4", "mov"])

with col2:
    show_draw = st.checkbox("Show pose overlay on video (preview only)", value=True)
    min_detection_confidence = st.slider("MP Detection Confidence", 0.2, 0.95, 0.5)
    min_tracking_confidence = st.slider("MP Tracking Confidence", 0.2, 0.95, 0.5)
    sender_email = st.text_input("Your Email (for sending report)", "")
    sender_password = st.text_input("App Password (for sending report)", type="password")

analyze_button = st.button("Analyze")

# ---------- Analysis flow ----------
if analyze_button:
    if not video_file:
        st.error("Upload a video first dumbass (jk). But seriously, upload a file.")
    else:
        # Save uploaded video to temp file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        try:
            tfile.write(video_file.read())
            tfile.flush()
            video_path = tfile.name

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                st.error("Couldn't open uploaded video. Try a different file.")
            else:
                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

                # Counters and histories
                counter = 0
                stage = None
                jump_heights = []
                ankle_history = []
                cheat_flag = False

                # For stabilized rep counting we only count transition once.
                last_counted_frame = -1
                frame_idx = 0

                # We'll capture a short preview video with overlays if requested.
                # For simplicity we won't re-encode the whole video (heavy). We show the original video and display metrics.

                with mp_pose.Pose(min_detection_confidence=min_detection_confidence,
                                  min_tracking_confidence=min_tracking_confidence) as pose:
                    progress_text = st.empty()
                    pb = st.progress(0)

                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break

                        frame_idx += 1
                        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = pose.process(image_rgb)

                        landmarks = None
                        if results.pose_landmarks:
                            landmarks = results.pose_landmarks.landmark
                            # Check cheat quickly
                            cheat_flag |= detect_cheating(landmarks, test_type)

                        # Wrap per-test logic in try/except to avoid crash on single-frame bad data
                        try:
                            if landmarks:
                                if test_type == "curl":
                                    shoulder = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y)
                                    elbow = (landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                             landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y)
                                    wrist = (landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                             landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y)
                                    angle = calculate_angle(shoulder, elbow, wrist)
                                    # rep logic: down -> up transition counts 1
                                    if angle > 160:
                                        stage = "down"
                                    if angle < 40 and stage == "down":
                                        counter += 1
                                        stage = "up"

                                elif test_type == "pushup":
                                    shoulder = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y)
                                    elbow = (landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                             landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y)
                                    wrist = (landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                             landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y)
                                    angle = calculate_angle(shoulder, elbow, wrist)
                                    if angle > 160:
                                        stage = "up"
                                    if angle < 60 and stage == "up":
                                        counter += 1
                                        stage = "down"

                                elif test_type == "situp":
                                    hip = (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                                           landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y)
                                    shoulder = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y)
                                    knee = (landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                                            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y)
                                    angle = calculate_angle(shoulder, hip, knee)
                                    if angle > 140:
                                        stage = "down"
                                    if angle < 90 and stage == "down":
                                        counter += 1
                                        stage = "up"

                                elif test_type == "jump":
                                    h = estimate_jump_height(landmarks, frame_height)
                                    jump_heights.append(h)

                                elif test_type == "sprint":
                                    ankle_y = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y
                                    ankle_history.append(ankle_y)
                                    # keep history size limited (1-3 seconds)
                                    max_hist = int(fps * 3)
                                    if len(ankle_history) > max_hist:
                                        ankle_history.pop(0)

                        except Exception:
                            # ignore single-frame issues
                            pass

                        # update progress UI every few frames
                        if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0:
                            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            pb.progress(min(frame_idx / total, 1.0))
                            progress_text.text(f"Processing frame {frame_idx}/{total}")
                        else:
                            # unknown length — just update spinner
                            pb.progress(min(frame_idx % 100 / 100.0, 1.0))

                    # finished processing
                    cap.release()
                    pb.empty()
                    progress_text.empty()

                # Prepare results
                st.success("✅ Analysis Complete")
                st.video(video_path)

                # Metrics & feedback logic
                feedback = []
                score = None

                if test_type in ["curl", "pushup", "situp"]:
                    st.metric("Reps Counted", counter)
                    score = counter
                    feedback.append("Good form!" if counter >= 10 else "Try to improve rep count and consistency.")
                elif test_type == "jump":
                    max_height = max(jump_heights) if jump_heights else 0.0
                    st.metric("Max Jump Height (px)", max_height)
                    score = max_height
                    feedback.append("Explosive jump!" if max_height > (frame_height * 0.6) else "Work on jump height and leg drive.")
                elif test_type == "sprint":
                    freq = estimate_sprint_frequency(ankle_history, fps)
                    st.metric("Sprint Frequency (approx)", f"{freq} cycles/sec")
                    score = freq
                    feedback.append("Great pace!" if freq > 2.5 else "Increase stride frequency and cadence.")

                if cheat_flag:
                    st.warning("⚠️ Cheat detected: low visibility for required joints or suspicious form detected.")

                st.write("Feedback:")
                for tip in feedback:
                    st.markdown(f"- {tip}")

                # Report sending UI
                with st.expander("📤 Send Report to SAI"):
                    st.write("Enter your email and app password to send report (video attached). Use App Password if using Gmail.")
                    send_btn = st.button("Send Report")
                    if send_btn:
                        if not sender_email or not sender_password:
                            st.error("Provide your email and app password to send the report.")
                        else:
                            try:
                                with st.spinner("Sending report..."):
                                    send_email_to_sai(
                                        test_type=test_type,
                                        score=score,
                                        confidence=1.0,
                                        cheat_flag=cheat_flag,
                                        feedback=feedback,
                                        video_path=video_path,
                                        sender_email=sender_email,
                                        sender_password=sender_password,
                                    )
                                st.success("📤 Report sent to SAI successfully")
                            except Exception as e:
                                st.error(f"Failed to send report: {e}")

        finally:
            try:
                tfile.close()
            except Exception:
                pass

# ---------- Footer/help ----------
st.markdown("---")
st.write("Tips: Record with the athlete visible, keep the camera steady, and ensure joints are not occluded. Use an email app password for Gmail SMTP.")
