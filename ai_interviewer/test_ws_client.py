"""
WebSocket STT & TTS Tester Client
==================================

Prerequisites:
    pip install websockets requests pyaudio

Usage:
    1. Make sure `python api.py` is running in another terminal.
    2. Run this script: `python test_ws_client.py`
    
This script mimics your website:
- Sends a mock plan (like the .NET Planner) to the REST API.
- Connects to the WebSocket.
- Lets you press ENTER to start/stop recording from your microphone (STT).
- Plays the binary audio received from the server (TTS) through your speakers.
"""

import asyncio
import json
import uuid
import requests
import websockets
import threading

try:
    import pyaudio
except ImportError:
    print("\n[!] Please install required packages: pip install pyaudio websockets requests\n")
    exit(1)

# =========================================================================
# 1. THE MOCK PLANNER OUTPUT (Edit this to test different plans)
# =========================================================================

MOCK_PLAN = MOCK_PLAN = {
  "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",

  "candidate_name": "Ahmed Hassan",

  "job_role": "Machine Learning Intern",

  "level": "Intern",

  "duration_limit": 3000,

  "questions": [

    {
      "question": "What is the difference between a list and a tuple in Python?",

      "golden_answer": "A list is mutable, meaning you can add, remove, or modify its elements after creation. A tuple is immutable, meaning its elements cannot be changed after creation. Lists use square brackets [] while tuples use parentheses ().",

      "skill": "python"
    },
    {
      "question": "What is the difference between supervised and unsupervised learning?",

      "golden_answer": "Supervised learning uses labeled data where the correct output is known. Unsupervised learning uses unlabeled data to discover hidden patterns or structures. Classification and regression are supervised tasks, while clustering is an example of unsupervised learning.",

      "skill": "machine_learning"
    },

    {
      "question": "What is train test split and why do we use it?",

      "golden_answer": "Train test split divides the dataset into a training set and a testing set. The training set is used to train the model and the testing set is used to evaluate its performance on unseen data. This helps detect overfitting and measure generalization.",

      "skill": "model_evaluation"
    },

    {
      "question": "What is cross validation?",

      "golden_answer": "Cross validation is a model evaluation technique in which the dataset is divided into several folds. The model is trained and tested multiple times using different folds, and the average performance is used as the final score.",

      "skill": "model_evaluation"
    },

  ]
}

# =========================================================================
# AUDIO SETTINGS
# =========================================================================
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000     # 16kHz for STT (Whisper)
CHUNK = 1024

audio = pyaudio.PyAudio()

# =========================================================================
# GLOBALS & HELPERS
# =========================================================================

is_recording = False
audio_buffer = []

def record_audio_thread():
    """Continuously reads from the microphone while is_recording is True."""
    global is_recording, audio_buffer
    
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print("\n[🎤 Recording Started... Speak now! Press ENTER to stop & send]")
    
    while is_recording:
        data = stream.read(CHUNK, exception_on_overflow=False)
        audio_buffer.append(data)
        
    stream.stop_stream()
    stream.close()
    print("[🛑 Recording Stopped]")

# =========================================================================
# WEBSOCKET CLIENT
# =========================================================================

async def test_websocket_flow():
    # 1. Register the plan via REST
    print("--- 1. Registering Plan via REST ---")
    res = requests.post("http://localhost:8000/api/interview/prepare", json=MOCK_PLAN)
    if res.status_code != 200:
        print(f"Failed to prepare session: {res.text}")
        return
    session_id = MOCK_PLAN["session_id"]
    print(f"Plan registered! Session ID: {session_id}")
    
    # 2. Connect to WebSocket
    ws_url = f"ws://localhost:8000/ws/interview/{session_id}"
    print(f"\n--- 2. Connecting to WebSocket {ws_url} ---")
    
    async with websockets.connect(ws_url) as websocket:
        print("Connected! Waiting for AI's introduction...\n")
        
        # Audio output stream for playing the AI's TTS response
        play_stream = audio.open(format=FORMAT, channels=CHANNELS, rate=24000, output=True) # ElevenLabs/Deepgram usually 24kHz or 44kHz. Adjust if needed.
        
        async def listen_to_server():
            """Background task to listen for text/audio frames from the server."""
            try:
                async for message in websocket:
                    if isinstance(message, str):
                        # Text frame (JSON)
                        data = json.loads(message)
                        msg_type = data.get("type")
                        
                        if msg_type in ["intro", "question", "followup", "close"]:
                            print(f"\n[AI ({data.get('state')}): {data.get('text')}]")
                            print("\n>>> PRESS ENTER TO START RECORDING YOUR ANSWER <<<")
                            
                        elif msg_type == "clarification":
                            print(f"\n[🔄 CLARIFICATION ({data.get('state')}): {data.get('text')}]")
                            print("[⏱️  +45 seconds bonus time added!]")
                            print("\n>>> PRESS ENTER TO START RECORDING YOUR ANSWER <<<")

                        elif msg_type == "question_timeout":
                            q_idx = data.get("question_index", "?")
                            print(f"\n[⏰ TIME'S UP for question #{q_idx}! Moving on...]")

                        elif msg_type == "complete":
                            print("\n[✅ INTERVIEW COMPLETE]")
                            print("You can now fetch the final report via REST.")
                            break
                            
                        elif msg_type == "tts_start":
                            pass # Incoming audio frames will follow
                        elif msg_type == "tts_end":
                            pass # Audio streaming finished
                        else:
                            print(f"[Server JSON]: {data}")
                            
                    else:
                        # Binary frame (TTS audio chunk from server)
                        play_stream.write(message)
                        
            except websockets.exceptions.ConnectionClosed:
                print("\n[WebSocket Connection Closed by Server]")

        # Start the listener task in the background
        listener_task = asyncio.create_task(listen_to_server())
        
        # 3. Interactive Loop
        global is_recording, audio_buffer
        
        # We use asyncio.get_event_loop().run_in_executor to not block the async loop while waiting for Enter
        loop = asyncio.get_running_loop()
        
        while not listener_task.done():
            # Wait for user to press Enter to start recording
            await loop.run_in_executor(None, input)
            if listener_task.done(): break
            
            # Start recording
            is_recording = True
            audio_buffer = []
            rec_thread = threading.Thread(target=record_audio_thread)
            rec_thread.start()
            
            # Wait for user to press Enter again to stop recording
            await loop.run_in_executor(None, input)
            is_recording = False
            rec_thread.join()
            
            # Combine binary audio
            raw_audio = b"".join(audio_buffer)
            
            # Wrap the raw PCM audio in a valid WAV container
            import io
            import wave
            wav_io = io.BytesIO()
            with wave.open(wav_io, 'wb') as wav_file:
                wav_file.setnchannels(CHANNELS)
                wav_file.setsampwidth(audio.get_sample_size(FORMAT))
                wav_file.setframerate(RATE)
                wav_file.writeframes(raw_audio)
            
            full_audio = wav_io.getvalue()
            
            # Send signal that audio is coming
            await websocket.send(json.dumps({"type": "answer_audio"}))
            # Send the valid WAV binary data
            await websocket.send(full_audio)
            
            print("[Sent Audio to Server. Waiting for evaluation and response...]")

        play_stream.stop_stream()
        play_stream.close()

if __name__ == "__main__":
    try:
        asyncio.run(test_websocket_flow())
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        audio.terminate()
