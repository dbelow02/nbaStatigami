#Goal is to publish to twitter / discord

#Unfortunately, twitter recently changed their api to be used based, meaning, it isn't free use.
#Could potentially do browser simulation via selenium/ playwright?

#Or, just email results to myself and manually post after copying text

import smtplib
from email.mime.text import MIMEText
import datetime
from dotenv import load_dotenv
import os

load_dotenv()
GMAIL_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")
GMAIL = os.getenv("GMAIL_EMAIL")

def send_headline_email(body, app_password = GMAIL_PASSWORD, email = GMAIL):
    msg = MIMEText(body)
    date_string = datetime.datetime.now().strftime("%Y-%m-%d")
    msg["Subject"] = f"NBA Statigami Posts for {date_string}"
    msg["From"] = email
    msg["To"] = email

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(email, app_password)
        server.sendmail(email, email, msg.as_string())



if __name__ == "__main__":
    print("This should not be called directly")


#How to use it
#Go to the settings for your Google Account in the application or device you are trying to set up. 
#Replace your password with the 16-character password shown above.
#Just like your normal password, this app password grants complete access to your Google Account. 
#You won't need to remember it, so don't write it down or share it with anyone.