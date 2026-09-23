"""Hugging Face Space Automatic Deployment Script.

Automates creating a Hugging Face Space and uploading the pdf-knowledge-assistant application.

Usage:
    python scripts/deploy_hf.py --token <HF_WRITE_TOKEN> --space-name pdf-knowledge-assistant
    OR
    export HF_TOKEN="hf_..."
    python scripts/deploy_hf.py
"""

import argparse
import os
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from huggingface_hub import HfApi, create_repo, upload_folder


def deploy(token: str, space_name: str, private: bool = False):
    api = HfApi(token=token)
    user_info = api.whoami()
    username = user_info["name"]
    repo_id = f"{username}/{space_name}"

    print(f"Authenticated as Hugging Face user: {username}")
    print(f"Target Space repository: https://huggingface.co/spaces/{repo_id}")

    # 1. Create Space repo if not exists
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="gradio",
            private=private,
            token=token,
            exist_ok=True,
        )
        print("Space repository initialized on Hugging Face.")
    except Exception as e:
        if "402" in str(e):
            print("\n⚠️ Note: Automated Space creation via API returned 402.")
            print("Please create the Space once manually on the web: https://huggingface.co/new-space")
            print("  - Space name: " + space_name)
            print("  - Space SDK: Gradio")
            print("Once created, this script will upload all code files directly.\n")
        else:
            print(f"Note on repo creation: {e}")

    # 2. Upload application files (ignoring local data and environments)
    print("Uploading application files to Hugging Face Space...")
    ignore_patterns = [
        ".env",
        ".env.*",
        "data/uploads/**",
        "data/processed/**",
        "data/vectorstore/*.faiss",
        "data/vectorstore/*.json",
        ".venv/**",
        "venv/**",
        "__pycache__/**",
        "*.pyc",
        ".pytest_cache/**",
        ".git/**",
        "tests/**",
    ]

    upload_folder(
        folder_path=str(PROJECT_DIR),
        repo_id=repo_id,
        repo_type="space",
        ignore_patterns=ignore_patterns,
        token=token,
        commit_message="Deploy PDF Knowledge Assistant to Hugging Face Spaces",
    )

    print("\n" + "=" * 65)
    print(f"🎉 Deployment uploaded successfully!")
    print(f"👉 Live Space URL: https://huggingface.co/spaces/{repo_id}")
    print("=" * 65)
    print("\nIMPORTANT NEXT STEP:")
    print("1. Go to your Space on Hugging Face: https://huggingface.co/spaces/" + repo_id)
    print("2. Click on 'Settings' -> 'Variables and secrets'")
    print("3. Under 'Secrets', click 'New secret'")
    print("   - Name: GEMINI_API_KEY")
    print("   - Value: <your google gemini api key>")
    print("4. Your Space will build and launch automatically!")


def main():
    parser = argparse.ArgumentParser(description="Deploy PDF Knowledge Assistant to Hugging Face Spaces.")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face User Access Token (Write permissions)")
    parser.add_argument("--space-name", type=str, default="pdf-knowledge-assistant", help="Target Hugging Face Space name")
    parser.add_argument("--private", action="store_true", help="Set the space as private")
    args = parser.parse_args()

    token = args.token or os.getenv("HF_TOKEN")
    if not token:
        print("\n❌ Error: Hugging Face token not found.")
        print("Please obtain a WRITE token from: https://huggingface.co/settings/tokens")
        print("Then run:")
        print("    python scripts/deploy_hf.py --token <YOUR_HF_TOKEN>\n")
        sys.exit(1)

    deploy(token=token, space_name=args.space_name, private=args.private)


if __name__ == "__main__":
    main()

