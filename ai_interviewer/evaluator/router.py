"""
Evaluator Router
================

FastAPI router — previously exposed POST /evaluator/evaluate/{session_id}
for the .NET backend to manually trigger evaluation after an interview.

** DEPRECATED — endpoint removed **

Evaluation is now fully automatic and push-based:
1. The interview WebSocket closes.
2. `_save_and_evaluate()` in api.py runs the evaluation pipeline in the background.
3. The final result is POSTed directly to the .NET webhook URL configured via
   WEBHOOK_URL in api_keys.json (or as an environment variable).

The .NET backend no longer needs to call this service to retrieve evaluation
results — they will be delivered automatically via webhook.

If you need to re-expose the endpoint for debugging, restore it from git history.
"""

import logging
import sys
import os

# Allow importing project-level modules (config, etc.) from the parent dir
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import APIRouter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/evaluator", tags=["Evaluator"])

# No endpoints — evaluation is now push-based via webhook.
# See _save_and_evaluate() in api.py for the implementation.
