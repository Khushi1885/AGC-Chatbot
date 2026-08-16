"""
email_service.py
Sends confirmation emails via Resend (https://resend.com).

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


def send_application_confirmation(to_email: str, name: str, program: str) -> bool:
    """Sends a confirmation email after a student submits the admission enquiry form."""
    if not to_email:
        print("[EMAIL] No email address provided - skipping")
        return False

    if not RESEND_API_KEY:
        print("[EMAIL] RESEND_API_KEY not configured in .env - skipping email send")
        return False

    html_body = f"""
    <div style="font-family: Arial, sans-serif; max-width: 480px; margin: 0 auto;">
      <h2 style="color: #7A1B35;">Application Received</h2>
      <p>Dear {name},</p>
      <p>Thank you for submitting your admission enquiry for
      <b>{program}</b> at Amritsar Group of Colleges (AGC).</p>
      <p>Our admission counselors will contact you within 24 hours on the
      mobile number you provided.</p>
      <p>If you have any questions in the meantime, feel free to chat with
      AdmitAgent on our website or call us at <b>+91 8872009950</b>.</p>
      <p style="margin-top: 24px;">Best regards,<br/>AGC Admission Cell</p>
    </div>
    """

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
                "subject": f"AGC Admission - We've received your enquiry for {program}",
                "html": html_body,
            },
            timeout=15,
        )
        if response.status_code >= 400:
            print(f"[EMAIL ERROR] Resend API error {response.status_code}: {response.text}")
            return False
        print(f"[EMAIL] Confirmation sent to {to_email}")
        return True
    except Exception as e:
        print(f"[EMAIL ERROR] Failed to send confirmation email: {e}")
        return False