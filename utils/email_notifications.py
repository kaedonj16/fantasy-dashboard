"""
Email notification utilities for cron job failures and important events.
"""

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import traceback
import json


def get_email_config():
    """Get email configuration from environment variables."""
    return {
        'smtp_server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
        'smtp_port': int(os.getenv('SMTP_PORT', '587')),
        'email_user': os.getenv('EMAIL_USER'),
        'email_password': os.getenv('EMAIL_PASSWORD'),
        'recipient_email': os.getenv('RECIPIENT_EMAIL'),
    }


def is_email_configured():
    """Check if email configuration is properly set up."""
    config = get_email_config()
    return all([config['email_user'], config['email_password'], config['recipient_email']])


def is_sender_configured():
    """True when outbound-mail *sender* creds are set (recipient not required).

    ``is_email_configured`` also requires RECIPIENT_EMAIL, which is only for the
    admin error digest. User-facing mail (weekly digest) picks its own
    recipients, so it needs the SMTP login but not that admin address."""
    config = get_email_config()
    return bool(config['email_user'] and config['email_password'])


def send_html_email(to_email: str, subject: str, html_body: str,
                    text_body: str = None, unsubscribe_url: str = None) -> bool:
    """Send one HTML email (with a plain-text fallback part) to a single address.

    Returns True on send, False if mail isn't configured or the send failed.
    Kept dependency-free (stdlib smtplib) to match send_error_email; callers
    are responsible for not sending to opted-out users.

    When ``unsubscribe_url`` is given, adds the List-Unsubscribe headers
    (RFC 2369 + RFC 8058 one-click) so Gmail/Apple Mail surface a native
    unsubscribe control — which both improves the recipient experience and
    materially lowers the odds the message is flagged as spam."""
    if not is_sender_configured():
        print("[email] sender not configured, skipping send")
        return False
    to_email = (to_email or "").strip()
    if not to_email or "@" not in to_email:
        return False

    config = get_email_config()
    try:
        msg = MIMEMultipart("alternative")
        msg["From"] = config["email_user"]
        msg["To"] = to_email
        msg["Subject"] = subject
        if unsubscribe_url:
            msg["List-Unsubscribe"] = f"<{unsubscribe_url}>"
            msg["List-Unsubscribe-Post"] = "List-Unsubscribe=One-Click"
        # A plain-text part first, then HTML: clients render the last part they
        # can display, so HTML wins where supported and text is the fallback.
        msg.attach(MIMEText(text_body or _html_to_text(html_body), "plain"))
        msg.attach(MIMEText(html_body, "html"))
        with smtplib.SMTP(config["smtp_server"], config["smtp_port"]) as server:
            server.starttls()
            server.login(config["email_user"], config["email_password"])
            server.send_message(msg)
        return True
    except Exception as e:
        print(f"[email] send_html_email to {to_email[:40]} failed: {e}")
        return False


def _html_to_text(html: str) -> str:
    """Very small HTML→text fallback (no external deps): drop tags, keep text."""
    import re
    text = re.sub(r"<\s*br\s*/?>", "\n", html or "", flags=re.I)
    text = re.sub(r"</\s*(p|div|tr|h[1-6]|li)\s*>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def send_error_email(subject: str, error_message: str, context: dict = None):
    """
    Send an error notification email.
    
    Args:
        subject: Email subject line
        error_message: Main error message
        context: Additional context information (optional)
    """
    if not is_email_configured():
        print("[email] Email not configured, skipping notification")
        return False
    
    config = get_email_config()
    
    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = config['email_user']
        msg['To'] = config['recipient_email']
        msg['Subject'] = f"[Fantasy Dashboard Alert] {subject}"
        
        # Build email body
        body_parts = [
            f"Error occurred at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Environment: {os.getenv('PYTHON_ENV', 'development')}",
            "",
            f"Error: {error_message}",
            ""
        ]
        
        if context:
            body_parts.append("Context:")
            body_parts.append("-" * 20)
            for key, value in context.items():
                if isinstance(value, dict):
                    body_parts.append(f"{key}:")
                    for k, v in value.items():
                        body_parts.append(f"  {k}: {v}")
                else:
                    body_parts.append(f"{key}: {value}")
            body_parts.append("")
        
        # Add traceback if available
        if traceback.format_exc() != 'NoneType: None\n':
            body_parts.append("Traceback:")
            body_parts.append("-" * 20)
            body_parts.append(traceback.format_exc())
        
        body = "\n".join(body_parts)
        msg.attach(MIMEText(body, 'plain'))
        
        # Send email
        with smtplib.SMTP(config['smtp_server'], config['smtp_port']) as server:
            server.starttls()
            server.login(config['email_user'], config['email_password'])
            server.send_message(msg)
        
        print(f"[email] Error notification sent to {config['recipient_email']}")
        return True
        
    except Exception as e:
        print(f"[email] Failed to send error notification: {e}")
        return False


def send_cron_failure_notification(error: Exception, context: dict = None):
    """
    Send a notification when cron_daily fails.
    
    Args:
        error: The exception that occurred
        context: Additional context information
    """
    context = context or {}
    context.update({
        'script': 'cron_daily.py',
        'error_type': type(error).__name__,
    })
    
    return send_error_email(
        subject="Cron Daily Job Failed",
        error_message=str(error),
        context=context
    )


