import smtplib

email = input("SENDER EMAIL: ")
receiver_email = input("RECEIVER EMAIL: ")
server_starttls = input()
subject = input("SUBJECT: ")
message = input("MESSAGE: ")

text = f"Subject: {subject}\n\n{message}"

server = smtplib.SMTP("smtp.gmail.com", 587)
server.starttls()

server.login(email, "fzfdsixlpxlabjbfli")

server.sendmail(email, receiver_email, text)

print("Email has been sent to " + receiver_email)
