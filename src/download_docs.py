"""
Document Downloader for NIST AI RMF Documents.
Downloads:
1. NIST AI Risk Management Framework (AI RMF 1.0)
2. NIST AI RMF: Generative Artificial Intelligence Profile (NIST AI 600-1)
"""

import os
import sys
import urllib.request
import urllib.error
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

DOCUMENTS = [
    {
        "name": "NIST AI RMF 1.0",
        "filename": "nist_ai_rmf_1.0.pdf",
        "url": "https://nvlpubs.nist.gov/nistpubs/ai/nist.ai.100-1.pdf",
        "description": "NIST AI Risk Management Framework (AI RMF 1.0)"
    },
    {
        "name": "NIST AI RMF GenAI Profile",
        "filename": "nist_ai_rmf_genai.pdf",
        "url": "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf",
        "description": "NIST AI RMF Generative Artificial Intelligence Profile (NIST AI 600-1)"
    }
]


def download_file(url: str, dest_path: Path, doc_name: str) -> bool:
    """Download a file with user-agent headers and progress logging."""
    if dest_path.exists() and dest_path.stat().st_size > 100000:
        print(f"[OK] {doc_name} already exists at {dest_path} ({dest_path.stat().st_size:,} bytes). Skipping download.")
        return True

    print(f"[-] Downloading {doc_name} from {url}...")
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=60) as response, open(dest_path, "wb") as out_file:
            total_size = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            block_size = 64 * 1024  # 64 KB

            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                downloaded += len(buffer)
                out_file.write(buffer)
                if total_size > 0:
                    percent = downloaded / total_size * 100
                    sys.stdout.write(f"\r    Progress: {downloaded:,} / {total_size:,} bytes ({percent:.1f}%)")
                    sys.stdout.flush()

        print(f"\n[SUCCESS] Saved {doc_name} to {dest_path} ({dest_path.stat().st_size:,} bytes).")
        return True
    except urllib.error.URLError as e:
        print(f"\n[ERROR] Failed to download {doc_name} from {url}: {e}")
        return False


def download_all_documents() -> bool:
    """Download all required NIST documents if not already downloaded."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    all_success = True
    for doc in DOCUMENTS:
        dest = DATA_DIR / doc["filename"]
        success = download_file(doc["url"], dest, doc["name"])
        if not success:
            all_success = False
    return all_success


if __name__ == "__main__":
    success = download_all_documents()
    if not success:
        sys.exit(1)
