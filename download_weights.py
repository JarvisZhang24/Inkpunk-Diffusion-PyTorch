#!/usr/bin/env python3
"""
Script to download Stable Diffusion model weights from Hugging Face.
Uses token from .env file and downloads to ./assets directory.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, login

# Load environment variables from .env file
load_dotenv()

# Configuration
ASSETS_DIR = Path("./assets")
TOKENIZER_DIR = ASSETS_DIR / "tokenizer"

# Model repository and files to download
REPO_ID_STABLE_DIFFUSION = "runwayml/stable-diffusion-v1-5"
REPO_ID_INKPUNK = "Envvi/Inkpunk-Diffusion"

# Files to download
STABLE_DIFFUSION_FILES = [
    "v1-5-pruned-emaonly.ckpt"
]

TOKENIZER_FILES = [
    "tokenizer/merges.txt",
    "tokenizer/vocab.json"
]

INKPUNK_FILES = [
    "Inkpunk-Diffusion-v2.ckpt"
]


def setup_directories():
    """Create necessary directories."""
    ASSETS_DIR.mkdir(exist_ok=True)
    TOKENIZER_DIR.mkdir(exist_ok=True)
    print(f"✓ Created directories: {ASSETS_DIR}")


def authenticate():
    """Authenticate with Hugging Face using token from .env file."""
    token = os.getenv("HUGGINGFACE_TOKEN")
    
    if not token:
        print("❌ HUGGINGFACE_TOKEN not found in .env file")
        print("Please create a .env file with your Hugging Face token:")
        print("HUGGINGFACE_TOKEN=your_token_here")
        return False
    
    try:
        login(token=token)
        print("✓ Successfully authenticated with Hugging Face")
        return True
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        return False


def download_files(repo_id: str, files: list, description: str):
    """Download files from a Hugging Face repository."""
    print(f"\n📥 Downloading {description} from {repo_id}...")
    
    for file in files:
        try:
            print(f"  Downloading {file}...")
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=file,
                local_dir=str(ASSETS_DIR),
                local_dir_use_symlinks=False  # Create actual files instead of symlinks
            )
            print(f"  ✓ {file} -> {downloaded_path}")
        except Exception as e:
            print(f"  ❌ Failed to download {file}: {e}")


def main():
    """Main download function."""
    print("🚀 Starting Stable Diffusion weights download...")
    
    # Setup directories
    setup_directories()
    
    # Authenticate with Hugging Face
    if not authenticate():
        return 1
    
    try:
        # Download Inkpunk Diffusion weights
        download_files(
            repo_id=REPO_ID_INKPUNK,
            files=INKPUNK_FILES,
            description="Inkpunk Diffusion weights"
        )
        
        # Download tokenizer files
        download_files(
            repo_id=REPO_ID_INKPUNK,
            files=TOKENIZER_FILES,
            description="CLIP tokenizer"
        )
        
        # Download Inkpunk Diffusion (optional - commented out by default)
        # Uncomment the following lines if you want to download Inkpunk Diffusion
        # download_files(
        #     repo_id=REPO_ID_INKPUNK,
        #     files=INKPUNK_FILES,
        #     description="Inkpunk Diffusion"
        # )
        
        print("\n✅ All downloads completed successfully!")
        print(f"📁 Files saved to: {ASSETS_DIR.absolute()}")
        
        # List downloaded files
        print("\n📋 Downloaded files:")
        for file in ASSETS_DIR.rglob("*"):
            if file.is_file():
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"  {file.relative_to(ASSETS_DIR)} ({size_mb:.1f} MB)")
                
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
