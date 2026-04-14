#!/usr/bin/env python3
"""
Check current environment variables
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

email_vars = [
    'EMAIL_USER',
    'EMAIL_PASSWORD', 
    'RECIPIENT_EMAIL',
    'SMTP_SERVER',
    'SMTP_PORT',
    'PYTHON_ENV'
]

print("Current email environment variables:")
for var in email_vars:
    value = os.getenv(var)
    if value:
        if 'PASSWORD' in var:
            print(f"{var}: {'*' * len(value)}")
        else:
            print(f"{var}: {value}")
    else:
        print(f"{var}: (not set)")
