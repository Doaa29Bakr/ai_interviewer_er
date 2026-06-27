from huggingface_hub import HfApi
import os

api = HfApi()

print("Uploading to Hugging Face Spaces...")
api.upload_folder(
    folder_path=".",
    repo_id="Doaa-Helmy/interviewer2",
    repo_type="space",
    token="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",
    ignore_patterns=[
        ".git*",
        "__pycache__/*",
        "deploy.py",
        "conversations/*",
        ".env",
        "api_keys.json",
        "*.pyc"
    ]
)
print("Deployment to Hugging Face Spaces successful!")
