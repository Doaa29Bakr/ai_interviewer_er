# AI-driven Interview System

This project is a backend application for an **AI-driven Interview System**. It provides a fully automated interviewer agent that conducts live, voice-based technical interviews with candidates over WebSockets.

The system acts as a middleware component:
- An external service (like a .NET backend) prepares the interview session.
- A frontend (e.g., React) connects to conduct the actual conversation in real-time.

## Features

- **Live Voice Interactions**: Real-time bidirectional communication over WebSockets using Speech-To-Text (STT) and Text-To-Speech (TTS).
- **AI Interviewer Agent**: Powered by Groq LLM API and dynamic prompts to simulate a senior technical interviewer persona ("Alex").
- **State Machine Architecture**: Strictly controls the interview flow through sequential phases: `INTRO` -> `WARMUP` -> `ASK` -> `FOLLOWUP` -> `CLOSE`.
- **Impartial Evaluation**: An independent evaluator agent grades candidate answers against "golden answers" across multiple dimensions (correctness, completeness, clarity).
- **Interactive CLI Testing**: Includes a standalone script (`interview_me.py`) for sandbox testing directly in the terminal.

## Architecture

The system is built primarily with Python and FastAPI. Here is a high-level overview of the core components:

- **`api.py`**: FastAPI entry point handling REST API endpoints for session management and WebSocket endpoint for the real-time interview.
- **`orchestrator.py`**: The Interview Engine that manages the session lifecycle, LLM interactions, and final report generation.
- **`state_machine.py`**: Defines the `InterviewStateMachine` controlling the interview phases.
- **`conversation.py`**: Manages conversational history and limits context window for the LLM.
- **`prompts.py`**: Stores the System and User prompts for the Interviewer and Evaluator agents.
- **`session.py` & `storage.py`**: Manage in-memory state of active sessions and handle persistent storage of final reports (via Redis).
- **`stt.py` & `tts.py`**: Wrappers for Speech-To-Text (Groq Whisper Large V3) and Text-To-Speech (Deepgram Aura) APIs.

For more details on the file breakdown and sequence diagrams, refer to [Project Overview](project_overview.md).

## Getting Started

*(Ensure you have your virtual environment set up and activated.)*

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Environment Variables:**
   Ensure you have configured the necessary API keys and environment settings (e.g., in `.env` or `api_keys.json`).
3. **Run the API Server:**
   ```bash
   uvicorn api:app --reload
   ```
4. **Try the CLI Sandbox:**
   ```bash
   python interview_me.py
   ```
