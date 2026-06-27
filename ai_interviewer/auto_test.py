import asyncio
import json
import requests
import websockets
import time

MOCK_PLAN = {
  "session_id": "test-timeout-1234",
  "candidate_name": "Test Candidate",
  "job_role": "Python Developer",
  "level": "Intern",
  "duration_limit": 40,
  "questions": [
    {
      "question": "What is a decorator in Python?",
      "golden_answer": "A decorator is a function that takes another function and extends its behavior without explicitly modifying it.",
      "skill": "python"
    }
  ]
}

async def run_test():
    print("1. Preparing session via REST API...")
    res = requests.post("http://localhost:8000/api/interview/prepare", json=MOCK_PLAN)
    if res.status_code != 200:
        print(f"Failed to prepare: {res.text}")
        # Try to delete and retry if it exists
        requests.delete(f"http://localhost:8000/api/interview/{MOCK_PLAN['session_id']}")
        res = requests.post("http://localhost:8000/api/interview/prepare", json=MOCK_PLAN)
        
    session_id = MOCK_PLAN["session_id"]
    ws_url = f"ws://localhost:8000/ws/interview/{session_id}"
    
    print(f"2. Connecting to WebSocket {ws_url}...")
    start_time = time.time()
    
    async with websockets.connect(ws_url) as ws:
        async def listen():
            async for msg in ws:
                if isinstance(msg, str):
                    data = json.loads(msg)
                    msg_type = data.get('type')
                    state = data.get('state', 'N/A')
                    text = data.get('text', '')
                    
                    if msg_type in ["intro", "question", "followup", "close"]:
                        elapsed = time.time() - start_time
                        print(f"\n[{elapsed:.1f}s] AI ({state}): {text}")
                        
                        if state == "INTRO":
                            # Send an automated reply to move to WARMUP
                            print(f"\n[{elapsed:.1f}s] Sending reply to INTRO...")
                            await ws.send(json.dumps({"type": "answer", "text": "Yes, I am ready to begin."}))
                        elif state == "WARMUP":
                            # Don't send an answer here; just wait and let it timeout!
                            print(f"\n[{elapsed:.1f}s] Received WARMUP question. Now simulating candidate silence to trigger the 20-second timeout...")
                    elif msg_type == 'complete':
                        elapsed = time.time() - start_time
                        print(f"\n[{elapsed:.1f}s] Interview marked as complete by server.")
                        return
        
        await listen()

if __name__ == "__main__":
    asyncio.run(run_test())
