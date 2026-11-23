"""
Hybrid Vector Store and Retrieval Management for NIST AI RMF RAG System.

Features:
- Local HuggingFace embeddings ('sentence-transformers/all-MiniLM-L6-v2'; optional upgrade: 'BAAI/bge-small-en-v1.5').
- Persistent ChromaDB vector store using langchain_chroma.
- Hybrid Retrieval: Chroma dense similarity search + BM25 keyword search via EnsembleRetriever.
- Exact-ID Shortcut: Instant metadata lookup for Action IDs (e.g. GV-3.2-001), Subcategories (e.g. MEASURE 2.11),
  and Figures (e.g. Figure 4), placing exact-match nodes at the top of results.
- Fully decoupled from test databases to ensure `./chroma_db` is never polluted.
"""

import os
import sys
import re
import pickle
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Avoid unauthenticated online checks if model is already downloaded locally
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from pydantic import Field

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

try:
    from langchain.retrievers import EnsembleRetriever
except ImportError:
    try:
        from langchain_classic.retrievers import EnsembleRetriever
    except ImportError:
        from langchain_community.retrievers import EnsembleRetriever

from langchain_community.retrievers import BM25Retriever

from src.config import (
    CHROMA_DIR,
    PARSED_DATA_DIR,
    EMBEDDING_PROVIDER,
    EMBEDDING_MODEL_NAME,
    OPENAI_API_KEY,
    TOP_K_RETRIEVAL,
)

logger = logging.getLogger("rag.vector_store")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[-] [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Global caches
_embeddings_instance = None
_corpus_cache: Optional[List[Document]] = None
_bm25_instance: Optional[BM25Retriever] = None


def get_embeddings():
    """
    Initialize and return the embedding model.
    Defaults to local HuggingFace embeddings ('sentence-transformers/all-MiniLM-L6-v2').
    (Note: 'BAAI/bge-small-en-v1.5' can be configured via EMBEDDING_MODEL_NAME as an optional upgrade).
    """
    global _embeddings_instance
    if _embeddings_instance is not None:
        return _embeddings_instance

    if EMBEDDING_PROVIDER.lower() == "openai" and OPENAI_API_KEY:
        try:
            from langchain_openai import OpenAIEmbeddings
            _embeddings_instance = OpenAIEmbeddings(
                model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
                api_key=OPENAI_API_KEY,
            )
            return _embeddings_instance
        except Exception as e:
            logger.warning(f"Failed to load OpenAIEmbeddings ({e}), falling back to HuggingFace.")

    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings

    _embeddings_instance = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return _embeddings_instance


def get_vector_store(persist_dir: Optional[Union[str, Path]] = None, collection_name: str = "nist_ai_rmf_collection") -> Chroma:
    """
    Get or initialize persistent Chroma vector store.
    """
    embeddings = get_embeddings()
    target_dir = Path(persist_dir) if persist_dir else CHROMA_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(target_dir),
    )
    return vector_store


def load_corpus(corpus_path: Optional[Path] = None, vector_store: Optional[Chroma] = None) -> List[Document]:
    """
    Load document corpus from disk cache or ChromaDB.
    """
    global _corpus_cache
    if _corpus_cache is not None and not corpus_path:
        return _corpus_cache

    path = corpus_path or (PARSED_DATA_DIR / "corpus.pkl")
    if path.exists():
        try:
            with open(path, "rb") as f:
                docs = pickle.load(f)
                if not corpus_path:
                    _corpus_cache = docs
                return docs
        except Exception as e:
            logger.warning(f"Could not load corpus from pickle ({e}), falling back to Chroma.")

    vs = vector_store or get_vector_store()
    try:
        raw = vs.get()
        docs = []
        for text, meta in zip(raw.get("documents", []), raw.get("metadatas", [])):
            docs.append(Document(page_content=text, metadata=meta or {}))
        if not corpus_path:
            _corpus_cache = docs
        return docs
    except Exception as e:
        logger.warning(f"Failed to load corpus from ChromaDB: {e}")
        return []


def exact_id_lookup(query: str, corpus: List[Document]) -> List[Document]:
    """
    Shortcut lookup for exact IDs in query:
    - Action IDs: e.g. GV-3.2-001, MS-1.1-002
    - Subcategories: e.g. GOVERN 4.2, MEASURE 2.11
    - Figures: e.g. Figure 4, Fig. 5
    """
    matches: List[Document] = []
    seen_ids = set()

    # 1. Action ID Match: e.g. GV-3.2-001
    action_matches = re.findall(r"\b([A-Z]{2}-\d+\.\d+-\d{3})\b", query, re.IGNORECASE)
    for aid in action_matches:
        target = aid.upper()
        for doc in corpus:
            doc_aid = doc.metadata.get("action_id", "").upper()
            if doc_aid == target and target not in seen_ids:
                matches.append(doc)
                seen_ids.add(target)

    # 2. Subcategory Match: e.g. MEASURE 2.11, GOVERN 4.2
    subcat_matches = re.findall(r"\b((?:GOVERN|MAP|MEASURE|MANAGE)\s+\d+\.\d+)\b", query, re.IGNORECASE)
    for sc in subcat_matches:
        target = " ".join(sc.upper().split())
        for doc in corpus:
            doc_sc = doc.metadata.get("subcategory_id", "").upper()
            if doc_sc == target and target not in seen_ids:
                matches.append(doc)
                seen_ids.add(target)

    # 2b. Category Match: e.g. GOVERN 4, MEASURE 2
    cat_matches = re.findall(r"\b((?:GOVERN|MAP|MEASURE|MANAGE)\s+\d+)\b", query, re.IGNORECASE)
    for cat in cat_matches:
        target = " ".join(cat.upper().split())
        for doc in corpus:
            doc_cat = doc.metadata.get("category_id", "").upper()
            cid = doc.metadata.get("chunk_id", "")
            if doc_cat == target and cid not in seen_ids:
                matches.append(doc)
                seen_ids.add(cid)

    # 2c. Function Subcategories Query: e.g. "subcategories of GOVERN", "GOVERN subcategories"
    fn_subcat_match = re.search(
        r"\b(?:subcategories|categories)\s+of\s+(?:the\s+)?(GOVERN|MAP|MEASURE|MANAGE)\b|\b(GOVERN|MAP|MEASURE|MANAGE)\s+(?:function\s+)?(?:categories|subcategories)\b",
        query,
        re.IGNORECASE,
    )
    if fn_subcat_match:
        fn = (fn_subcat_match.group(1) or fn_subcat_match.group(2)).upper()
        for doc in corpus:
            if doc.metadata.get("node_type") == "table_row" and doc.metadata.get("function") == fn:
                cid = doc.metadata.get("chunk_id", "")
                if cid not in seen_ids:
                    matches.append(doc)
                    seen_ids.add(cid)

    # 3. Figure Match: e.g. Figure 4, Fig. 5
    fig_matches = re.findall(r"\b(?:Fig(?:ure|\.)?)\s*(\d+)\b", query, re.IGNORECASE)
    for fnum in fig_matches:
        target = f"fig_{fnum}"
        for doc in corpus:
            if doc.metadata.get("node_type") == "figure":
                fid = doc.metadata.get("figure_id", "").lower()
                if fid == target and target not in seen_ids:
                    matches.append(doc)
                    seen_ids.add(target)

    # 3b. Trustworthiness Characteristics match -> Figure 4
    if re.search(r"\b(?:trustworth(?:y|iness)\s+characteristics?|characteristics?\s+of\s+trustworthy\b)", query, re.IGNORECASE):
        for doc in corpus:
            if doc.metadata.get("node_type") == "figure" and doc.metadata.get("figure_id") == "fig_4":
                cid = doc.metadata.get("chunk_id", "")
                if cid not in seen_ids:
                    matches.append(doc)
                    seen_ids.add(cid)

    return matches


class HybridRetriever(BaseRetriever):
    """
    Retriever that combines Chroma similarity search and BM25 with
    Exact-ID metadata priority shortcut.
    """
    ensemble: Any = Field(default=None)
    corpus: List[Document] = Field(default_factory=list)
    k: int = Field(default=TOP_K_RETRIEVAL)

    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        # 1. Check exact-ID shortcut first
        shortcut_docs = exact_id_lookup(query, self.corpus)

        # 2. Query ensemble (Chroma + BM25)
        ensemble_docs = []
        if self.ensemble is not None:
            try:
                ensemble_docs = self.ensemble.invoke(query)
            except Exception as e:
                logger.warning(f"Ensemble retriever failed: {e}")

        # 3. Merge: place shortcut docs at the very top, followed by ensemble docs (deduplicated)
        final_results: List[Document] = []
        seen_chunks = set()

        for doc in shortcut_docs:
            cid = doc.metadata.get("chunk_id", doc.page_content[:50])
            if cid not in seen_chunks:
                final_results.append(doc)
                seen_chunks.add(cid)

        for doc in ensemble_docs:
            cid = doc.metadata.get("chunk_id", doc.page_content[:50])
            if cid not in seen_chunks:
                final_results.append(doc)
                seen_chunks.add(cid)
            if len(final_results) >= self.k:
                break

        return final_results[: self.k]


def get_retriever(
    k: int = TOP_K_RETRIEVAL,
    persist_dir: Optional[Union[str, Path]] = None,
    collection_name: str = "nist_ai_rmf_collection",
    corpus_docs: Optional[List[Document]] = None,
) -> BaseRetriever:
    """
    Get configured HybridRetriever combining Chroma similarity, BM25, and Exact-ID shortcut.
    """
    global _bm25_instance
    vs = get_vector_store(persist_dir=persist_dir, collection_name=collection_name)
    chroma_retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": k})

    corpus = corpus_docs if corpus_docs is not None else load_corpus(vector_store=vs)
    if not corpus:
        return chroma_retriever

    if _bm25_instance is None or corpus_docs is not None:
        try:
            bm25 = BM25Retriever.from_documents(corpus)
            bm25.k = k
            if corpus_docs is None:
                _bm25_instance = bm25
        except Exception as e:
            logger.warning(f"BM25 initialization failed ({e}). Using Chroma alone.")
            return chroma_retriever
    else:
        bm25 = _bm25_instance

    try:
        ensemble = EnsembleRetriever(
            retrievers=[chroma_retriever, bm25],
            weights=[0.5, 0.5],
        )
    except Exception as e:
        logger.warning(f"EnsembleRetriever creation failed ({e}). Using Chroma alone.")
        ensemble = chroma_retriever

    return HybridRetriever(ensemble=ensemble, corpus=corpus, k=k)


def store_documents(
    documents: List[Document],
    batch_size: int = 100,
    persist_dir: Optional[Union[str, Path]] = None,
    collection_name: str = "nist_ai_rmf_collection",
) -> int:
    """
    Add document chunks to Chroma vector store in batches and persist.
    """
    vs = get_vector_store(persist_dir=persist_dir, collection_name=collection_name)
    total_docs = len(documents)

    target_dir = persist_dir or CHROMA_DIR
    logger.info(f"Storing {total_docs} document chunks in ChromaDB at {target_dir}...")
    for i in range(0, total_docs, batch_size):
        batch = documents[i : i + batch_size]
        vs.add_documents(batch)
        logger.info(f"Indexed chunks {i + 1} to {min(i + batch_size, total_docs)} / {total_docs}")

    logger.info("All chunks indexed successfully into ChromaDB.")
    return total_docs
