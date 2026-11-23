"""
Root Ingestion Entrypoint.
Delegates directly to src.ingest.ingest_documents.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingest import ingest_documents

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingest NIST AI RMF documents into ChromaDB.")
    parser.add_argument("--force", action="store_true", help="Force re-indexing even if database exists")
    args = parser.parse_args()

    total = ingest_documents(force_reindex=args.force)
    print(f"[SUCCESS] Ingestion completed. Total chunks in ChromaDB: {total}")
