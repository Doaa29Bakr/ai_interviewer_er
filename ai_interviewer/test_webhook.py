"""
Webhook Connection Test
=======================
Sends a fake interview evaluation payload to WEBHOOK_URL
and prints the HTTP response to verify connectivity.

Run with:
    python test_webhook.py
"""

import json
import sys
import httpx

# ── Load the same config your app uses ──────────────────────────────────────
sys.path.insert(0, ".")
from config import get_key

WEBHOOK_URL = get_key("WEBHOOK_URL")

if not WEBHOOK_URL:
    print("❌  WEBHOOK_URL is not set in api_keys.json")
    sys.exit(1)

# ── Fake payload (matches InterviewResultRequestDto exactly) ─────────────────
fake_payload = {
    "session_id": "00000000-0000-0000-0000-000000000001",
    "candidate_name": "Test Candidate",
    "job_role": "Software Engineer",
    "level": "Mid",
    "average_score": 7.5,
    "overall_summary": "This is a webhook connectivity test. Not a real interview result.",
    "evaluations": [
        {
            "question": "What is a REST API?",
            "score": 8,
            "covered_requirements": ["Defines REST", "Mentions HTTP methods"],
            "missing_requirements": ["HATEOAS constraint"]
        },
        {
            "question": "Explain the difference between SQL and NoSQL.",
            "score": 7,
            "covered_requirements": ["Schema differences", "Use cases"],
            "missing_requirements": ["CAP theorem"]
        }
    ]
}

# ── Send the request ─────────────────────────────────────────────────────────
print(f"\n Sending test payload to:\n    {WEBHOOK_URL}\n")

try:
    with httpx.Client(timeout=15.0) as client:
        response = client.post(WEBHOOK_URL, json=fake_payload)

    print(f"HTTP Status : {response.status_code}")
    print(f"Response    : {response.text[:500]}")

    if response.status_code == 200:
        print("\nWebhook is connected and accepting payloads!")
    else:
        print(f"\nWebhook responded with a non-200 status.")
        print("    Check if the .NET backend expects any auth headers.")

except httpx.ConnectError:
    print("Connection refused -- server may be down or the URL is wrong.")
except httpx.TimeoutException:
    print("Request timed out -- server did not respond within 15 seconds.")
except Exception as exc:
    print(f"Unexpected error: {exc}")
