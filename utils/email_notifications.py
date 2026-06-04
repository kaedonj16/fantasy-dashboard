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


