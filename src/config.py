"""
Configuration and Environment Settings for the NIST AI RMF Agentic RAG System.
Primary Model: Google Gemini (e.g. gemini-2.0-flash-lite / gemini-1.5-flash)
Fallback Model: Groq (e.g. openai/gpt-oss-120b / llama-3.3-70b-versatile)
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

# Paths
DATA_DIR = ROOT_DIR / "data"
PARSED_DATA_DIR = DATA_DIR / "parsed"
CHROMA_DIR = ROOT_DIR / "chroma_db"
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "nist_ai_rmf_collection")

# Primary LLM: Google Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite")

# Fallback LLM: Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# Optional Alternative: OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Provider selection
PRIMARY_PROVIDER = os.getenv("PRIMARY_PROVIDER", "gemini").lower()
FALLBACK_PROVIDER = os.getenv("FALLBACK_PROVIDER", "groq").lower()

# Embedding Configuration
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "huggingface")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# Chunking Strategy
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# Retrieval Settings
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "6"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "1"))
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "6"))
