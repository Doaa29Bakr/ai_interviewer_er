FROM python:3.11-slim

WORKDIR /app

# Install system dependencies if any are needed for audio/networking
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Hugging Face Spaces typically uses port 7860
EXPOSE 7860

# Command to run the application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
