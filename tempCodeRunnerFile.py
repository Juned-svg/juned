import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import smtplib
from email.message import EmailMessage

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Angle calculation
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

# Cheat detection
def detect_cheating(landmarks, test_type):
    required = {
        "pushup": [mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.RIGHT_ELBOW],
        "jump": [mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_ANKLE],
        "sprint": [mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.RIGHT_KNEE],
        "situp": [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP],
        "curl": [mp_pose.PoseLandmark.LEFT_ELBOW]
    }
    missing = 0
    for joint in required.get(test_type, []):
        if landmarks[joint.value].visibility < 0.5:
            missing += 1
    return missing > 0

# Jump height estimation
def estimate_jump_height(landmarks, frame_height):
    ankle_y = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * frame_height
    return round(frame_height - ankle_y, 2)

# Sprint frequency estimation
def estimate_sprint_frequency(ankle_history, fps):
    if len(ankle_history) < fps:
        return 0
    diffs = [abs(ankle_history[i] - ankle_history[i-1]) for i in range(1, len(ankle_history))]
    return round(sum(diffs) / len(diffs) * fps, 2)

# Email sender
def send_email_to_sai(test_type, score, confidence, cheat_flag, feedback, video_path, sender_email, sender_password):
    msg = EmailMessage()
    msg['Subject'] = f"SAI Fitness Report - {test_type}"
    msg['From'] = sender_email
    msg['To'] = "sai_official@example.com"
    msg.set_content(f"""
Test Type: {test_type}
Score: {score}
Confidence: {confidence}
Cheat Detected: {cheat_flag}
Feedback: {'; '.join(feedback)}
""")
    with open(video_path, 'rb') as f:
        msg.add_attachment(f.read(), maintype='video', subtype='mp4', filename='submission.mp4')
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(sender_email, sender_password)
        smtp.send_message(msg)

# Streamlit UI
st.set_page_config(page_title="SAI Athlete Performance Portal", layout="wide")
st.title("🏋️‍♂️ SAI Athlete Performance Portal")

test_type = st.selectbox("Choose Test Type", ["curl", "pushup", "situp", "jump", "sprint"])
video_file = st.file_uploader("Upload Video", type=["mp4", "mov"])
age = st.slider("Athlete Age", 10, 60, 25)
gender = st.selectbox("Gender", ["male", "female", "other"])

if st.button("Analyze") and video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    video_path = tfile.name
    cap = cv2.VideoCapture(video_path)

    counter, stage = 0, None
    jump_heights = []
    ankle_history = []
    cheat_flag = False
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark
                cheat_flag |= detect_cheating(landmarks, test_type)

                if test_type == "curl":
                    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    angle = calculate_angle(shoulder, elbow, wrist)
                    if angle > 160:
                        stage = "down"
                    elif angle < 40 and stage == "down":
                        stage = "up"
                        counter += 1

                elif test_type == "pushup":
                    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    angle = calculate_angle(shoulder, elbow, wrist)
                    if angle > 160:
                        stage = "up"
                    elif angle < 60 and stage == "up":
                        stage = "down"
                        counter += 1

                elif test_type == "situp":
                    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    angle = calculate_angle(shoulder, hip, knee)
                    if angle > 140:
                        stage = "down"
                    elif angle < 90 and stage == "down":
                        stage = "up"
                        counter += 1

                elif test_type == "jump":
                    height = estimate_jump_height(landmarks, frame_height)
                    jump_heights.append(height)

                elif test_type == "sprint":
                    ankle_y = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y
                    ankle_history.append(ankle_y)
                    if len(ankle_history) > fps:
                        ankle_history.pop(0)

            except:
                pass

        cap.release()

    st.success("✅ Analysis Complete")
    st.video(video_path)

    if cheat_flag:
        st.warning("⚠️ Cheat detected: Incomplete form or skipped reps")

    if test_type in ["curl", "pushup", "situp"]:
        st.metric("Reps Counted", counter)
        score = counter
        feedback = ["Good form!" if counter >= 10 else "Try to improve rep count."]
    elif test_type == "jump":
        max_height = max(jump_heights) if jump_heights else 0
        st.metric("Max Jump Height (px)", max_height)
        score = max_height
        feedback = ["Explosive jump!" if max_height > 300 else "Work on jump height."]
    elif test_type == "sprint":
        freq = estimate_sprint_frequency(ankle_history, fps)
        st.metric("Sprint Frequency", f"{freq} cycles/sec")
        score = freq
        feedback = ["Great pace!" if freq > 2.5 else "Increase stride frequency."]

    st.write("Feedback:")
    for tip in feedback:
        st.markdown(f"- {tip}")

    with st.expander("📤 Send Report to SAI"):
    sender_email = st.text_input("Your Email")
    sender_password = st.text_input("App Password", type="password")
    if st.button("Send"):
        send_email_to_sai(
            test_type=test_type,
            score=score,
            confidence=1.0,
            cheat_flag=cheat_flag,
            feedback=feedback,
            video_path=video_path,
            sender_email=sender_email,
            sender_password=sender_password
        )
        st.success("📤 Report sent to SAI successfully")
