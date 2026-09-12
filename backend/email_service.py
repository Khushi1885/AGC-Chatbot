"""
email_service.py
Sends emails via Resend (https://resend.com) - an HTTP API, not SMTP, so it
avoids antivirus/firewall software that blocks or breaks SMTP connections.

Setup required:
1. Create a free account at https://resend.com
2. Get your API key from the Resend dashboard (API Keys section)
3. Add to your .env file (project root):

   RESEND_API_KEY=re_xxxxxxxxxxxxxxxxxxxxxx
   RESEND_FROM_EMAIL=AGC Admissions <onboarding@resend.dev>

   Note: "onboarding@resend.dev" works out of the box for testing without any
   domain setup, but Resend's test mode only delivers to the email address you
   signed up with. To email real students, verify your own domain in the
   Resend dashboard and use an address on that domain instead
   (e.g. "AGC Admissions <admissions@youragcdomain.com>").
"""

import os
import requests
from dotenv import load_dotenv

_env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path=_env_path)

RESEND_API_KEY = os.environ.get("RESEND_API_KEY")
RESEND_FROM_EMAIL = os.environ.get("RESEND_FROM_EMAIL", "AGC Admissions <onboarding@resend.dev>")
RESEND_URL = "https://api.resend.com/emails"


def _send_email(to_email: str, subject: str, html_body: str) -> bool:
    if not to_email:
        print("[EMAIL] No email address provided - skipping")
        return False

    if not RESEND_API_KEY:
        print("[EMAIL] RESEND_API_KEY not configured in .env - skipping email send")
        return False

    try:
        response = requests.post(
            RESEND_URL,
            headers={
                "Authorization": f"Bearer {RESEND_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "from": RESEND_FROM_EMAIL,
                "to": [to_email],
                "subject": subject,
                "html": html_body,
            },
            timeout=15,
        )
        if response.status_code >= 400:
            print(f"[EMAIL ERROR] Resend API error {response.status_code}: {response.text}")
            return False
        print(f"[EMAIL] Sent to {to_email}")
        return True
    except Exception as e:
        print(f"[EMAIL ERROR] Failed to send email: {e}")
        return False


def send_application_confirmation(to_email: str, name: str, program: str, meeting_time: str = None) -> bool:
    """Sends a confirmation email after a student submits the admission enquiry form."""
    meeting_html = ""
    if meeting_time:
        meeting_html = f"""
      <p style="background:#f4f3ec; padding:12px; border-radius:6px;">
        📅 <b>Your counseling call is scheduled for:</b><br/>{meeting_time}
      </p>
        """

    html_body = f"""
    <div style="font-family: Arial, sans-serif; max-width: 480px; margin: 0 auto;">
      <h2 style="color: #7A1B35;">Application Received</h2>
      <p>Dear {name},</p>
      <p>Thank you for submitting your admission enquiry for <b>{program}</b> at
      Amritsar Group of Colleges (AGC).</p>
      {meeting_html}
      <p>Our admission counselors will contact you at this time on the
      mobile number you provided.</p>
      <p>If you have any questions in the meantime, feel free to chat with
      AdmitAgent on our website or call us at <b>+91 8872009950</b>.</p>
      <p style="margin-top: 24px;">Best regards,<br/>AGC Admission Cell</p>
    </div>
    """
    return _send_email(to_email, f"AGC Admission - We've received your enquiry for {program}", html_body)


def send_nest_credentials(to_email: str, name: str, username: str, password: str, test_url: str = None) -> bool:
    """Sends AGC NEST scholarship test login credentials after chatbot registration."""
    link_html = ""
    if test_url:
        link_html = f"""
      <p style="text-align:center; margin: 24px 0;">
        <a href="{test_url}" style="background:#7A1B35; color:white; padding:12px 28px;
           border-radius:6px; text-decoration:none; font-weight:bold; display:inline-block;">
          Take Test Now
        </a>
      </p>
      <p style="font-size:12px; color:#888;">Or copy this link: {test_url}</p>
        """

    html_body = f"""
    <div style="font-family: Arial, sans-serif; max-width: 480px; margin: 0 auto;">
      <h2 style="color: #7A1B35;">AGC NEST Registration Successful</h2>
      <p>Dear {name},</p>
      <p>You have been successfully registered for <b>AGC NEST</b>
      (National Entrance Scholarship Test).</p>
      <p>Your login details for the test portal:</p>
      <p style="background:#f4f3ec; padding:12px; border-radius:6px;">
        <b>Username:</b> {username}<br/>
        <b>Password:</b> {password}
      </p>
      {link_html}
      <p>Please keep these details safe. Good luck!</p>
      <p style="margin-top: 24px;">Best regards,<br/>AGC Admission Cell</p>
    </div>
    """
    return _send_email(to_email, "AGC NEST - Your Scholarship Test Login Details", html_body)