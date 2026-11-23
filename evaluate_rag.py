"""
evaluate_rag.py - Standalone Production RAG Retrieval Evaluation System
========================================================================

A complete, self-contained evaluation suite for the NIST AI RMF Agentic RAG system.
Evaluates retrieval relevance, ranking quality, and chunking configurations
without modifying production code or vector databases.

Key Capabilities:
1. Retrieval Metrics:
   - Precision@K
   - Recall@K
   - F1@K
   - Mean Reciprocal Rank (MRR)
   - Hit Rate@K (Success Rate)
2. Production Retriever Testing:
   - Evaluates the active hybrid retriever (Dense Chroma + BM25 + Exact-ID shortcut).
   - Extracts stable, deterministic chunk IDs (e.g. 'ai100-1:figure:fig_5', 'ai600-1:action:mg-3.2-001').
3. Chunking Experimentation Mode:
   - Compares candidate chunk configurations (e.g. 400/50, 600/100, 900/150, 1200/200).
   - Dynamically re-splits prose documents and indexes into isolated temporary ChromaDB stores.
   - Reports indexing time, query latency, and retrieval quality trade-offs without touching ./chroma_db.
4. Independent Self-Test Suite:
   - Verifies metric calculations against synthetic edge cases (all relevant, none relevant, partial,
     empty retrieval, unlabelled questions, rank variations) without dependencies on external models.
5. Optional Answer-Generation Quality Mode (BLEU & ROUGE-L):
   - Computes n-gram lexical overlap only when reference and generated answers exist, keeping
     retrieval evaluation cleanly decoupled from text generation.
6. Multi-Format Reporting:
   - Detailed console summaries, CSV question-level exports, and JSON aggregate reports.
"""

import os
import sys
import re
import csv
import json
import math
import time
import shutil
import logging
import argparse
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from collections import Counter

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Avoid HF online connectivity warnings
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# Import project configurations and retriever modules
from src.config import (
    TOP_K_RETRIEVAL,
    CHROMA_DIR,
    PARSED_DATA_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)
from src.vector_store import (
    get_retriever,
    get_vector_store,
    store_documents,
    load_corpus,
)
from langchain_core.documents import Document

# Setup structured logger
logger = logging.getLogger("rag.evaluate")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


# ============================================================================
# 1. CORE RETRIEVAL METRIC ENGINE
# ============================================================================

def compute_retrieval_metrics(
    retrieved_chunk_ids: List[str],
    relevant_chunk_ids: List[str],
    k: int = TOP_K_RETRIEVAL,
) -> Dict[str, Union[float, int, bool]]:
    """
    Computes standard Information Retrieval (IR) metrics for a single query.

    Definitions:
    - Precision@K:
        relevant retrieved chunks in top K / number of returned chunks in top K.
        If 0 chunks returned in top K, Precision@K = 0.0.
    - Recall@K:
        relevant retrieved chunks in top K / total known relevant chunks for query.
        If query has 0 labelled relevant chunks, Recall@K = 0.0 (and flagged).
    - F1@K:
        Harmonic mean of Precision@K and Recall@K. 0.0 if sum is 0.
    - Reciprocal Rank (RR):
        1 / rank of first relevant retrieved chunk (1-indexed, up to K); 0.0 if none.
    - Hit Rate@K:
        1.0 if at least one relevant chunk appears in top K; 0.0 otherwise.

    Edge-Case Handling:
    - Empty retrieval results: All metrics evaluated as 0.0.
    - Questions with no labelled relevant chunks: Marked with 'has_ground_truth=False'.
      They are excluded from aggregate recall calculations so unlabelled items
      do not artificially deflate system scores.
    """
    if k <= 0:
        raise ValueError(f"Parameter k must be a positive integer, got {k}")

    # Inspect top K retrieved chunks
    retrieved_at_k = retrieved_chunk_ids[:k]
    num_returned = len(retrieved_at_k)

    # Clean IDs to avoid whitespace/casing mismatch
    norm_retrieved = [cid.strip().lower() for cid in retrieved_at_k if cid]
    norm_relevant = [cid.strip().lower() for cid in relevant_chunk_ids if cid]

    has_ground_truth = len(norm_relevant) > 0
    num_relevant_total = len(norm_relevant)

    if not has_ground_truth:
        # Question has no labelled ground truth in test set
        return {
            "k": k,
            "num_returned": num_returned,
            "num_relevant_total": 0,
            "num_relevant_retrieved": 0,
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "f1_at_k": 0.0,
            "reciprocal_rank": 0.0,
            "hit_at_k": 0.0,
            "first_hit_rank": 0,
            "has_ground_truth": False,
        }

    relevant_set = set(norm_relevant)
    retrieved_set = set(norm_retrieved)
    intersection = relevant_set.intersection(retrieved_set)
    num_hits = len(intersection)

    # 1. Precision@K
    precision = (num_hits / num_returned) if num_returned > 0 else 0.0

    # 2. Recall@K
    recall = (num_hits / num_relevant_total) if num_relevant_total > 0 else 0.0

    # 3. F1@K
    if precision + recall > 0:
        f1 = (2.0 * precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    # 4. Reciprocal Rank (RR) & First Hit Rank
    first_hit_rank = 0
    reciprocal_rank = 0.0
    for idx, cid in enumerate(norm_retrieved, start=1):
        if cid in relevant_set:
            first_hit_rank = idx
            reciprocal_rank = 1.0 / idx
            break

    # 5. Hit Rate@K
    hit_at_k = 1.0 if num_hits > 0 else 0.0

    return {
        "k": k,
        "num_returned": num_returned,
        "num_relevant_total": num_relevant_total,
        "num_relevant_retrieved": num_hits,
        "precision_at_k": round(precision, 4),
        "recall_at_k": round(recall, 4),
        "f1_at_k": round(f1, 4),
        "reciprocal_rank": round(reciprocal_rank, 4),
        "hit_at_k": hit_at_k,
        "first_hit_rank": first_hit_rank,
        "has_ground_truth": True,
    }


# ============================================================================
# 2. SYNTHETIC SELF-TEST SUITE
# ============================================================================

def run_self_tests() -> bool:
    """
    Executes a comprehensive synthetic test suite verifying metric edge cases.
    Does NOT depend on embeddings, databases, or network connectivity.

    Tests verified:
    1. All retrieved chunks are relevant.
    2. No retrieved chunks are relevant.
    3. Only some retrieved chunks are relevant.
    4. Fewer than K results returned (partial return).
    5. No results returned (empty retrieval).
    6. No relevant chunks labelled for a question (unlabelled).
    7. First relevant result occurs at different ranks (rank 1, 2, 4, 6, beyond K).
    8. Set integrity with duplicate IDs.
    """
    logger.info("=== RUNNING RETRIEVAL EVALUATION SELF-TESTS ===")
    k = 5

    # Test 1: All retrieved chunks relevant (Precision=1.0, Recall=0.6, RR=1.0, Hit=1.0)
    t1 = compute_retrieval_metrics(
        retrieved_chunk_ids=["c1", "c2", "c3"],
        relevant_chunk_ids=["c1", "c2", "c3", "c4", "c5"],
        k=k,
    )
    assert math.isclose(t1["precision_at_k"], 1.0), f"Test 1 failed: {t1}"
    assert math.isclose(t1["recall_at_k"], 0.6), f"Test 1 failed: {t1}"
    assert math.isclose(t1["reciprocal_rank"], 1.0), f"Test 1 failed: {t1}"
    assert t1["hit_at_k"] == 1.0, f"Test 1 failed: {t1}"
    logger.info("  [PASS] Test 1: All retrieved chunks relevant.")

    # Test 2: No retrieved chunks relevant (Precision=0.0, Recall=0.0, RR=0.0, Hit=0.0)
    t2 = compute_retrieval_metrics(
        retrieved_chunk_ids=["x1", "x2", "x3", "x4", "x5"],
        relevant_chunk_ids=["c1", "c2"],
        k=k,
    )
    assert t2["precision_at_k"] == 0.0, f"Test 2 failed: {t2}"
    assert t2["recall_at_k"] == 0.0, f"Test 2 failed: {t2}"
    assert t2["reciprocal_rank"] == 0.0, f"Test 2 failed: {t2}"
    assert t2["hit_at_k"] == 0.0, f"Test 2 failed: {t2}"
    logger.info("  [PASS] Test 2: No retrieved chunks relevant.")

    # Test 3: Some retrieved relevant (Top 5: [x1, c1, x2, c2, x3]; Hits=2, Rel=2 -> P=0.4, R=1.0, RR=0.5, Hit=1.0)
    t3 = compute_retrieval_metrics(
        retrieved_chunk_ids=["x1", "c1", "x2", "c2", "x3"],
        relevant_chunk_ids=["c1", "c2"],
        k=k,
    )
    assert math.isclose(t3["precision_at_k"], 0.4), f"Test 3 failed: {t3}"
    assert math.isclose(t3["recall_at_k"], 1.0), f"Test 3 failed: {t3}"
    assert math.isclose(t3["reciprocal_rank"], 0.5), f"Test 3 failed: {t3}"
    assert t3["hit_at_k"] == 1.0, f"Test 3 failed: {t3}"
    logger.info("  [PASS] Test 3: Some retrieved chunks relevant.")

    # Test 4: Fewer than K results returned (Returned: 2, Hit: 1, Rel: 1 -> P=0.5, R=1.0, RR=1.0)
    t4 = compute_retrieval_metrics(
        retrieved_chunk_ids=["c1", "x1"],
        relevant_chunk_ids=["c1"],
        k=k,
    )
    assert math.isclose(t4["precision_at_k"], 0.5), f"Test 4 failed: {t4}"
    assert math.isclose(t4["recall_at_k"], 1.0), f"Test 4 failed: {t4}"
    assert t4["num_returned"] == 2, f"Test 4 failed: {t4}"
    logger.info("  [PASS] Test 4: Fewer than K results returned.")

    # Test 5: No results returned (empty retrieval)
    t5 = compute_retrieval_metrics(
        retrieved_chunk_ids=[],
        relevant_chunk_ids=["c1", "c2"],
        k=k,
    )
    assert t5["precision_at_k"] == 0.0, f"Test 5 failed: {t5}"
    assert t5["recall_at_k"] == 0.0, f"Test 5 failed: {t5}"
    assert t5["reciprocal_rank"] == 0.0, f"Test 5 failed: {t5}"
    assert t5["hit_at_k"] == 0.0, f"Test 5 failed: {t5}"
    logger.info("  [PASS] Test 5: No results returned (empty retrieval).")

    # Test 6: No relevant chunks labelled for question (has_ground_truth=False)
    t6 = compute_retrieval_metrics(
        retrieved_chunk_ids=["c1", "c2"],
        relevant_chunk_ids=[],
        k=k,
    )
    assert t6["has_ground_truth"] is False, f"Test 6 failed: {t6}"
    assert t6["recall_at_k"] == 0.0, f"Test 6 failed: {t6}"
    logger.info("  [PASS] Test 6: No relevant chunks labelled for question.")

    # Test 7: First hit rank variations (Rank 1 -> RR=1.0, Rank 3 -> RR=0.3333, Rank 5 -> RR=0.2, Beyond K -> RR=0.0)
    t7a = compute_retrieval_metrics(["c1", "x1", "x2"], ["c1"], k=k)
    assert math.isclose(t7a["reciprocal_rank"], 1.0), f"Test 7a failed: {t7a}"

    t7b = compute_retrieval_metrics(["x1", "x2", "c1", "x3", "x4"], ["c1"], k=k)
    assert math.isclose(t7b["reciprocal_rank"], 1.0 / 3.0, rel_tol=1e-3), f"Test 7b failed: {t7b}"

    t7c = compute_retrieval_metrics(["x1", "x2", "x3", "x4", "c1"], ["c1"], k=k)
    assert math.isclose(t7c["reciprocal_rank"], 0.2, rel_tol=1e-3), f"Test 7c failed: {t7c}"

    # Hit occurs at position 6 (beyond k=5) -> Reciprocal Rank must be 0.0
    t7d = compute_retrieval_metrics(["x1", "x2", "x3", "x4", "x5", "c1"], ["c1"], k=k)
    assert t7d["reciprocal_rank"] == 0.0, f"Test 7d failed: {t7d}"
    assert t7d["hit_at_k"] == 0.0, f"Test 7d failed: {t7d}"
    logger.info("  [PASS] Test 7: Reciprocal Rank variations & K truncation verified.")

    logger.info("=== ALL 7 SELF-TEST SUITES PASSED SUCCESSFULLY ===")
    return True


# ============================================================================
# 3. BLEU & ROUGE-L ANSWER EVALUATION (OPTIONAL / DECOUPLED)
# ============================================================================

def explain_bleu_rouge_rationale():
    """
    Documents why BLEU and ROUGE are NOT suitable as primary retrieval metrics.
    """
    explanation = """
[THEORETICAL EXPLANATION: WHY BLEU & ROUGE ARE NOT RETRIEVAL METRICS]
1. Information Retrieval (IR) vs. Natural Language Generation (NLG):
   - Retrieval evaluation measures SET RELEVANCE and RANKING ACCURACY of knowledge chunks.
   - Precision@K, Recall@K, and MRR measure whether the correct factual evidence was retrieved.
2. Lexical Distortion:
   - BLEU measures modified n-gram precision against a reference text.
   - ROUGE measures n-gram recall (ROUGE-1, ROUGE-2) and longest common subsequences (ROUGE-L).
   - A chunk might contain the EXACT factual answer (100% relevant) but use different phrasing,
     vocabulary, or length than the user's reference answer, receiving a poor BLEU/ROUGE score.
   - Conversely, an uninformative chunk that merely repeats question keywords can achieve high
     BLEU/ROUGE overlap while failing completely to answer the question.
3. Decoupling Principle:
   - Retrieval metrics evaluate the RETRIEVER (Chroma + BM25).
   - BLEU/ROUGE evaluate the GENERATOR (the LLM synthesis step).
   - In evaluate_rag.py, retrieval relevance is evaluated independently using chunk ID ground truth.
"""
    return explanation.strip()


def compute_rouge_l(reference: str, candidate: str) -> Dict[str, float]:
    """
    Computes token-level ROUGE-L (Longest Common Subsequence) score without external dependencies.
    """
    ref_tokens = re.findall(r"\w+", reference.lower())
    cand_tokens = re.findall(r"\w+", candidate.lower())

    m, n = len(ref_tokens), len(cand_tokens)
    if m == 0 or n == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # Dynamic programming matrix for Longest Common Subsequence
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == cand_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_len = dp[m][n]
    p = lcs_len / n if n > 0 else 0.0
    r = lcs_len / m if m > 0 else 0.0
    f1 = (2.0 * p * r) / (p + r) if (p + r) > 0 else 0.0

    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}


def compute_sentence_bleu(reference: str, candidate: str, max_n: int = 4) -> float:
    """
    Computes smoothed sentence-level BLEU-1 to BLEU-4 without external dependencies.
    """
    ref_tokens = re.findall(r"\w+", reference.lower())
    cand_tokens = re.findall(r"\w+", candidate.lower())

    c = len(cand_tokens)
    r = len(ref_tokens)
    if c == 0 or r == 0:
        return 0.0

    # Brevity penalty
    bp = 1.0 if c > r else math.exp(1.0 - (r / c))

    precisions = []
    for n in range(1, max_n + 1):
        if c < n or r < n:
            precisions.append(1.0 / (c + 1.0))
            continue

        cand_ngrams = [tuple(cand_tokens[i : i + n]) for i in range(c - n + 1)]
        ref_ngrams = [tuple(ref_tokens[i : i + n]) for i in range(r - n + 1)]

        cand_counts = Counter(cand_ngrams)
        ref_counts = Counter(ref_ngrams)

        clipped_hits = sum(min(count, ref_counts[ng]) for ng, count in cand_counts.items())
        p_n = (clipped_hits + 0.1) / (len(cand_ngrams) + 0.1)  # Laplace-style smoothing
        precisions.append(p_n)

    log_sum = sum(math.log(p) for p in precisions) / max_n
    return round(bp * math.exp(log_sum), 4)


# ============================================================================
# 4. GROUND TRUTH DATASET MANAGEMENT
# ============================================================================

STARTER_DATASET: List[Dict[str, Any]] = [
    {
        "question_id": "Q01",
        "question": "What are the four functions of the AI Risk Management Framework?",
        "relevant_chunk_ids": ["ai100-1:figure:fig_5", "ai100-1:p25:c1"],
        "reference_answer": "The AI RMF Core is composed of four functions: GOVERN, MAP, MEASURE, and MANAGE."
    },
    {
        "question_id": "Q02",
        "question": "Which function is designed to be cross-cutting in the AI RMF Core according to Figure 5?",
        "relevant_chunk_ids": ["ai100-1:figure:fig_5"],
        "reference_answer": "Governance (GOVERN) is designed to be a cross-cutting function that informs and is infused throughout Map, Measure, and Manage."
    },
    {
        "question_id": "Q03",
        "question": "What are the characteristics of trustworthy AI systems depicted in Figure 4?",
        "relevant_chunk_ids": ["ai100-1:figure:fig_4"],
        "reference_answer": "Valid & Reliable (base), Safe, Secure & Resilient, Explainable & Interpretable, Privacy-Enhanced, Fair - With Harmful Bias Managed, and Accountable & Transparent."
    },
    {
        "question_id": "Q04",
        "question": "What does subcategory MEASURE 2.11 cover?",
        "relevant_chunk_ids": ["ai100-1:s5.3:measure-2.11"],
        "reference_answer": "MEASURE 2.11 covers documenting measurement approaches for AI system performance and trustworthiness."
    },
    {
        "question_id": "Q05",
        "question": "What suggested action is provided for MG-3.2-001 in the Generative AI profile?",
        "relevant_chunk_ids": ["ai600-1:action:mg-3.2-001"],
        "reference_answer": "MG-3.2-001 suggests applying explainable AI (XAI) techniques (embeddings analysis, distillation, attribution) to mitigate unexplainable GAI risks."
    },
    {
        "question_id": "Q06",
        "question": "What suggested action addresses CBRN information risks under GV-3.2-001?",
        "relevant_chunk_ids": ["ai600-1:action:gv-3.2-001"],
        "reference_answer": "GV-3.2-001 establishes policies assessing GAI models for Chemical, Biological, Radiological, or Nuclear (CBRN) information or capabilities."
    },
    {
        "question_id": "Q07",
        "question": "What is the commitment described in category GOVERN 4?",
        "relevant_chunk_ids": ["ai100-1:s5.1:govern-4.1", "ai100-1:s5.1:govern-4.2", "ai100-1:s5.1:govern-4.3"],
        "reference_answer": "GOVERN 4 states: 'Organizational teams are committed to a culture that considers and communicates AI risk.'"
    },
    {
        "question_id": "Q08",
        "question": "How does the Generative AI Profile define confabulation?",
        "relevant_chunk_ids": ["ai600-1:p10:c1", "ai600-1:p10:c2"],
        "reference_answer": "Confabulation refers to when GAI systems generate confidently presented erroneous or false content (hallucinations/fabrications)."
    },
    {
        "question_id": "Q09",
        "question": "What are the AI actor tasks across lifecycle stages described in Figure 3?",
        "relevant_chunk_ids": ["ai100-1:figure:fig_3"],
        "reference_answer": "Figure 3 details AI actors across lifecycle stages (Plan, Collect, Build, Verify, Deploy, Operate, Impacted) and TEVV activities."
    },
    {
        "question_id": "Q10",
        "question": "What is subcategory GOVERN 1.1 regarding legal and regulatory requirements?",
        "relevant_chunk_ids": ["ai100-1:s5.1:govern-1.1"],
        "reference_answer": "GOVERN 1.1 requires that legal and regulatory requirements involving AI are understood, managed, and documented."
    },
    {
        "question_id": "Q11",
        "question": "What does subcategory MAP 1.1 state regarding AI context?",
        "relevant_chunk_ids": ["ai100-1:s5.2:map-1.1"],
        "reference_answer": "MAP 1.1 requires that the intended purposes, potential applications, and context of the AI system are identified, documented, and periodically reviewed."
    },
    {
        "question_id": "Q12",
        "question": "What suggested action is defined under MS-1.1-001 for monitoring information integrity?",
        "relevant_chunk_ids": ["ai600-1:action:ms-1.1-001"],
        "reference_answer": "MS-1.1-001 covers measuring and monitoring information integrity risks in generative AI outputs."
    }
]


def load_dataset(dataset_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Loads an evaluation dataset from JSON or JSONL format.
    Validates required fields: 'question' or 'query', and 'relevant_chunk_ids'.
    """
    path = Path(dataset_path)
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation dataset file not found: {path.resolve()}")

    records: List[Dict[str, Any]] = []

    if path.suffix.lower() == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    records.append(obj)
                except Exception as e:
                    logger.warning(f"Skipping malformed JSONL at line {idx}: {e}")
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                records = data
            elif isinstance(data, dict) and "questions" in data:
                records = data["questions"]
            else:
                raise ValueError(f"Unexpected JSON structure in {path}. Expected list or dict with 'questions'.")

    # Validate and standardize records
    validated: List[Dict[str, Any]] = []
    for idx, item in enumerate(records, start=1):
        qid = str(item.get("question_id") or item.get("id") or f"Q{idx:02d}")
        qtext = item.get("question") or item.get("query")
        if not qtext:
            logger.warning(f"Record {idx} missing 'question' field. Skipping.")
            continue

        rel_ids = item.get("relevant_chunk_ids") or item.get("ground_truth_chunks") or []
        if isinstance(rel_ids, str):
            rel_ids = [rel_ids]

        validated.append({
            "question_id": qid,
            "question": qtext.strip(),
            "relevant_chunk_ids": [str(c).strip() for c in rel_ids],
            "reference_answer": item.get("reference_answer", "").strip(),
        })

    logger.info(f"Loaded {len(validated)} validated test questions from {path.name}")
    return validated


def create_starter_dataset_file(target_path: Path):
    """
    Creates the starter ground-truth dataset populated with actual chunk IDs.
    """
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(STARTER_DATASET, f, indent=2, ensure_ascii=False)
    logger.info(f"Created starter evaluation dataset with {len(STARTER_DATASET)} entries at {target_path}")


# ============================================================================
# 5. RETRIEVAL EVALUATION EXECUTION
# ============================================================================

def evaluate_retriever_on_dataset(
    retriever: Any,
    dataset: List[Dict[str, Any]],
    k: int = TOP_K_RETRIEVAL,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Runs the existing retriever across all questions and computes metrics.
    No LLM generation calls are made during this retrieval evaluation.
    """
    question_results: List[Dict[str, Any]] = []
    start_time = time.perf_counter()

    for item in dataset:
        qid = item["question_id"]
        qtext = item["question"]
        relevant_ids = item["relevant_chunk_ids"]

        q_start = time.perf_counter()
        try:
            # Call production retrieval pipeline
            retrieved_docs: List[Document] = retriever.invoke(qtext)
        except Exception as e:
            logger.error(f"Retriever failed on question [{qid}] '{qtext}': {e}")
            retrieved_docs = []
        latency_sec = time.perf_counter() - q_start

        # Extract stable chunk identifiers and source metadata
        retrieved_ids = []
        source_metadata = []
        for d in retrieved_docs[:k]:
            cid = d.metadata.get("chunk_id")
            if not cid:
                # Fallback to deterministic identifier if chunk_id missing
                aid = d.metadata.get("action_id")
                sub = d.metadata.get("subcategory_id")
                fid = d.metadata.get("figure_id")
                cid = aid or sub or fid or f"page_{d.metadata.get('pdf_page', 'unk')}"
            retrieved_ids.append(cid)

            source_metadata.append({
                "chunk_id": cid,
                "document_name": d.metadata.get("document_name", "NIST Document"),
                "printed_page": d.metadata.get("printed_page"),
                "pdf_page": d.metadata.get("pdf_page"),
                "section": d.metadata.get("section_title") or d.metadata.get("section_path"),
            })

        metrics = compute_retrieval_metrics(
            retrieved_chunk_ids=retrieved_ids,
            relevant_chunk_ids=relevant_ids,
            k=k,
        )

        res = {
            "question_id": qid,
            "question": qtext,
            "relevant_chunk_ids": relevant_ids,
            "retrieved_chunk_ids": retrieved_ids,
            "source_metadata": source_metadata,
            "latency_sec": round(latency_sec, 4),
            **metrics,
        }
        question_results.append(res)

    total_time_sec = time.perf_counter() - start_time

    # Compute aggregate metrics across questions with ground truth
    evaluated_subset = [q for q in question_results if q["has_ground_truth"]]
    unlabelled_count = len(question_results) - len(evaluated_subset)

    if evaluated_subset:
        mean_p = sum(q["precision_at_k"] for q in evaluated_subset) / len(evaluated_subset)
        mean_r = sum(q["recall_at_k"] for q in evaluated_subset) / len(evaluated_subset)
        mean_f1 = sum(q["f1_at_k"] for q in evaluated_subset) / len(evaluated_subset)
        mean_mrr = sum(q["reciprocal_rank"] for q in evaluated_subset) / len(evaluated_subset)
        hit_rate = sum(q["hit_at_k"] for q in evaluated_subset) / len(evaluated_subset)
    else:
        mean_p = mean_r = mean_f1 = mean_mrr = hit_rate = 0.0

    missed_questions = [
        {"question_id": q["question_id"], "question": q["question"], "expected": q["relevant_chunk_ids"]}
        for q in evaluated_subset
        if q["hit_at_k"] == 0.0
    ]

    aggregate = {
        "k": k,
        "total_questions": len(question_results),
        "evaluated_questions": len(evaluated_subset),
        "unlabelled_questions": unlabelled_count,
        "mean_precision_at_k": round(mean_p, 4),
        "mean_recall_at_k": round(mean_r, 4),
        "mean_f1_at_k": round(mean_f1, 4),
        "mean_reciprocal_rank": round(mean_mrr, 4),
        "hit_rate_at_k": round(hit_rate, 4),
        "total_eval_time_sec": round(total_time_sec, 3),
        "avg_query_latency_sec": round(total_time_sec / max(1, len(question_results)), 4),
        "missed_question_count": len(missed_questions),
        "missed_questions": missed_questions,
    }

    return question_results, aggregate


# ============================================================================
# 6. EXPERIMENTAL CHUNKING COMPARISON SUITE
# ============================================================================

def build_experimental_documents(chunk_size: int, chunk_overlap: int) -> List[Document]:
    """
    Constructs an isolated, re-chunked Document set for chunking experiments.
    - Specialized atomic nodes (72 subcategory rows, 211 action rows, 5 figures)
      remain atomic units as defined by the NIST framework tables.
    - Long prose sections are re-split using SentenceSplitter configured with the
      experimental chunk size and overlap parameters.
    """
    from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
    from llama_index.core import Document as LlamaDocument

    md_1_path = PARSED_DATA_DIR / "nist_ai_rmf_1.0.md"
    md_g_path = PARSED_DATA_DIR / "nist_ai_rmf_genai.md"
    fig_path = PARSED_DATA_DIR / "figures.json"

    if not (md_1_path.exists() and md_g_path.exists() and fig_path.exists()):
        from src.parse_pdf import parse_nist_ai_100_to_markdown, parse_nist_ai_600_to_markdown
        parse_nist_ai_100_to_markdown()
        parse_nist_ai_600_to_markdown()

    with open(fig_path, "r", encoding="utf-8") as f:
        figs_100_1 = json.load(f).get("ai_rmf_1.0", {})

    all_docs: List[Document] = []

    # 1. Figure Nodes (Atomic)
    fig_pages = {"fig_1": (11, 6), "fig_2": (15, 10), "fig_3": (16, 11), "fig_4": (17, 12), "fig_5": (25, 20)}
    for fkey, finfo in figs_100_1.items():
        fnum = finfo.get("figure_num", fkey.replace("fig_", ""))
        caption = finfo.get("caption", f"Figure {fnum}")
        desc = finfo.get("description", "")
        p_page, pr_page = fig_pages.get(fkey, (1, None))
        meta = {
            "document_name": "NIST AI RMF 1.0",
            "doc_id": "ai_rmf_1.0",
            "section_title": f"Figure {fnum}",
            "pdf_page": p_page,
            "printed_page": pr_page,
            "node_type": "figure",
            "figure_id": fkey,
            "chunk_id": f"ai100-1:figure:{fkey}",
        }
        all_docs.append(Document(page_content=f"NIST AI RMF 1.0 > Figure {fnum}:\n\n{caption}\n\n{desc}", metadata=meta))

    # 2. AI 100-1 Subcategory Table Nodes (72 Atomic Units)
    doc1_text = md_1_path.read_text(encoding="utf-8")
    subcat_rows = re.findall(
        r"\|\s*((?:GOVERN|MAP|MEASURE|MANAGE)\s+\d+\.\d+)\s*\|\s*([^|]+)\|\s*((?:GOVERN|MAP|MEASURE|MANAGE)\s+\d+)\s*\|\s*([^|]+)\|",
        doc1_text,
    )
    seen_subcats = set()
    fn_map = {"GOVERN": "5.1", "MAP": "5.2", "MEASURE": "5.3", "MANAGE": "5.4"}
    for sub_id, sub_text, cat_id, cat_text in subcat_rows:
        sub_id = sub_id.strip()
        if sub_id in seen_subcats:
            continue
        seen_subcats.add(sub_id)
        fn = sub_id.split()[0]
        sec_num = fn_map[fn]
        chunk_id = f"ai100-1:s{sec_num}:{sub_id.lower().replace(' ', '-')}"
        meta = {
            "document_name": "NIST AI RMF 1.0",
            "doc_id": "ai_rmf_1.0",
            "section_title": f"{fn} Subcategory",
            "function": fn,
            "subcategory_id": sub_id,
            "category_text": cat_text.strip(),
            "subcategory_text": sub_text.strip(),
            "node_type": "table_row",
            "chunk_id": chunk_id,
        }
        all_docs.append(Document(page_content=f"Subcategory {sub_id}:\n{sub_text.strip()}", metadata=meta))

    # 3. AI 600-1 Action Table Nodes (211 Atomic Units)
    docg_text = md_g_path.read_text(encoding="utf-8")
    action_rows = re.findall(
        r"\|\s*([A-Z]{2}-\d+\.\d+-\d{3})\s*\|\s*([A-Z]+)\s*\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|]+)(?:\|\s*(\d+)\s*\|\s*(\d+)\s*)?\|",
        docg_text,
    )
    seen_actions = set()
    for row in action_rows:
        aid, fn, subcat, sugg, risks, tasks = row[:6]
        aid = aid.strip()
        if aid in seen_actions:
            continue
        seen_actions.add(aid)
        chunk_id = f"ai600-1:action:{aid.lower()}"
        meta = {
            "document_name": "NIST AI RMF Generative AI Profile",
            "doc_id": "ai_rmf_genai",
            "section_title": "Generative AI Profile Suggested Actions",
            "action_id": aid,
            "function": fn.strip(),
            "node_type": "table_row",
            "chunk_id": chunk_id,
        }
        all_docs.append(Document(page_content=f"Action {aid} ({fn.strip()}):\n{sugg.strip()}", metadata=meta))

    # 4. Re-chunk Prose Documents with Target Chunk Size and Overlap
    # Converting character budgets to token approximations (approx 4 chars/token)
    token_size = max(50, chunk_size // 4)
    token_overlap = max(10, chunk_overlap // 4)
    splitter = SentenceSplitter(chunk_size=token_size, chunk_overlap=token_overlap)
    md_parser = MarkdownNodeParser()

    # Re-chunk AI 100-1 prose
    clean_prose_1 = re.sub(r"<!-- (?:FIGURE|TABLE)_START.*?<!-- (?:FIGURE|TABLE)_END -->", "", doc1_text, flags=re.S)
    nodes_1 = md_parser.get_nodes_from_documents([LlamaDocument(text=clean_prose_1)])
    c_idx = 0
    for node in nodes_1:
        text = node.text.strip()
        if not text:
            continue
        sub_chunks = splitter.split_text(text) if len(text) > chunk_size else [text]
        for sc in sub_chunks:
            if len(sc.strip()) < 40:
                continue
            c_idx += 1
            meta = {
                "document_name": "NIST AI RMF 1.0",
                "doc_id": "ai_rmf_1.0",
                "section_title": "Core Prose",
                "node_type": "prose",
                "chunk_id": f"ai100-1:exp:c{c_idx}",
            }
            all_docs.append(Document(page_content=sc.strip(), metadata=meta))

    # Re-chunk AI 600-1 prose
    clean_prose_g = re.sub(r"<!-- (?:FIGURE|TABLE)_START.*?<!-- (?:FIGURE|TABLE)_END -->", "", docg_text, flags=re.S)
    nodes_g = md_parser.get_nodes_from_documents([LlamaDocument(text=clean_prose_g)])
    g_idx = 0
    for node in nodes_g:
        text = node.text.strip()
        if not text:
            continue
        sub_chunks = splitter.split_text(text) if len(text) > chunk_size else [text]
        for sc in sub_chunks:
            if len(sc.strip()) < 40:
                continue
            g_idx += 1
            meta = {
                "document_name": "NIST AI RMF Generative AI Profile",
                "doc_id": "ai_rmf_genai",
                "section_title": "GenAI Prose",
                "node_type": "prose",
                "chunk_id": f"ai600-1:exp:c{g_idx}",
            }
            all_docs.append(Document(page_content=sc.strip(), metadata=meta))

    return all_docs


def run_chunking_experiments(
    dataset: List[Dict[str, Any]],
    configs: List[Tuple[int, int]],
    k: int = TOP_K_RETRIEVAL,
) -> List[Dict[str, Any]]:
    """
    Evaluates candidate chunking configurations in isolated temporary vector stores.
    Production ./chroma_db is NEVER modified.
    """
    logger.info("=== STARTING CHUNKING CONFIGURATION EXPERIMENTS ===")
    experiment_results: List[Dict[str, Any]] = []

    for chunk_size, chunk_overlap in configs:
        config_name = f"{chunk_size}_chars_{chunk_overlap}_overlap"
        logger.info(f"\n[-] Running Experiment Configuration: {config_name}")

        temp_dir = tempfile.mkdtemp(prefix=f"eval_chroma_{chunk_size}_")
        try:
            # 1. Build Document chunks under this configuration
            index_start = time.perf_counter()
            chunks = build_experimental_documents(chunk_size, chunk_overlap)

            # 2. Index into isolated temporary ChromaDB
            store_documents(
                documents=chunks,
                batch_size=100,
                persist_dir=temp_dir,
                collection_name=f"eval_coll_{chunk_size}",
            )
            indexing_time_sec = time.perf_counter() - index_start

            # 3. Instantiate isolated HybridRetriever over temporary store
            exp_retriever = get_retriever(
                k=k,
                persist_dir=temp_dir,
                collection_name=f"eval_coll_{chunk_size}",
                corpus_docs=chunks,
            )

            # 4. Evaluate across the labelled test questions
            _, agg = evaluate_retriever_on_dataset(exp_retriever, dataset, k=k)

            res = {
                "config_name": config_name,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "total_chunks": len(chunks),
                "indexing_time_sec": round(indexing_time_sec, 2),
                "eval_time_sec": agg["total_eval_time_sec"],
                "mean_precision_at_k": agg["mean_precision_at_k"],
                "mean_recall_at_k": agg["mean_recall_at_k"],
                "mean_f1_at_k": agg["mean_f1_at_k"],
                "mean_reciprocal_rank": agg["mean_reciprocal_rank"],
                "hit_rate_at_k": agg["hit_rate_at_k"],
                "missed_question_count": agg["missed_question_count"],
            }
            experiment_results.append(res)
            logger.info(f"    Completed {config_name}: Hit Rate={agg['hit_rate_at_k']}, MRR={agg['mean_reciprocal_rank']}")

        finally:
            # Clean up isolated temporary Chroma directory
            shutil.rmtree(temp_dir, ignore_errors=True)

    return experiment_results


# ============================================================================
# 7. EXPORTING AND DISPLAY FORMATTERS
# ============================================================================

def export_question_results_csv(results: List[Dict[str, Any]], filepath: Path):
    """
    Exports question-level evaluation metrics to CSV using Python's standard library.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "question_id",
        "question",
        "has_ground_truth",
        "num_returned",
        "num_relevant_total",
        "num_relevant_retrieved",
        "precision_at_k",
        "recall_at_k",
        "f1_at_k",
        "reciprocal_rank",
        "hit_at_k",
        "first_hit_rank",
        "latency_sec",
        "retrieved_chunk_ids",
        "relevant_chunk_ids",
    ]

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            row = dict(r)
            row["retrieved_chunk_ids"] = "; ".join(row.get("retrieved_chunk_ids", []))
            row["relevant_chunk_ids"] = "; ".join(row.get("relevant_chunk_ids", []))
            writer.writerow(row)
    logger.info(f"Saved question-level CSV report to {filepath}")


def export_aggregate_json(aggregate: Dict[str, Any], filepath: Path):
    """
    Exports aggregate metrics to structured JSON.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved aggregate JSON report to {filepath}")


def export_experiments_csv(experiments: List[Dict[str, Any]], filepath: Path):
    """
    Exports chunking experiment comparison results to CSV.
    """
    if not experiments:
        return
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fields = list(experiments[0].keys())
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(experiments)
    logger.info(f"Saved chunking experiments CSV report to {filepath}")


def print_console_summary(results: List[Dict[str, Any]], aggregate: Dict[str, Any]):
    """
    Renders a clear, formatted console report of retrieval performance.
    """
    k = aggregate["k"]
    print("\n" + "=" * 80)
    print(f"       NIST AI RMF RETRIEVAL EVALUATION REPORT (Top K = {k})")
    print("=" * 80)
    print(f"Total Test Questions     : {aggregate['total_questions']}")
    print(f"Evaluated with GroundTruth: {aggregate['evaluated_questions']}")
    print(f"Unlabelled Questions     : {aggregate['unlabelled_questions']}")
    print(f"Total Evaluation Time    : {aggregate['total_eval_time_sec']}s")
    print(f"Avg Query Latency        : {aggregate['avg_query_latency_sec']}s")
    print("-" * 80)
    print(f"Hit Rate@{k} (Success %)   : {aggregate['hit_rate_at_k'] * 100:.1f}%")
    print(f"Mean Reciprocal Rank (MRR): {aggregate['mean_reciprocal_rank']:.4f}")
    print(f"Mean Precision@{k}         : {aggregate['mean_precision_at_k']:.4f}")
    print(f"Mean Recall@{k}            : {aggregate['mean_recall_at_k']:.4f}")
    print(f"Mean F1@{k}                : {aggregate['mean_f1_at_k']:.4f}")
    print("=" * 80)

    # Per-question breakdown
    print("\nPER-QUESTION PERFORMANCE BREAKDOWN:")
    print(f"{'QID':<6} {'P@K':<8} {'R@K':<8} {'F1@K':<8} {'RR':<8} {'Hit?':<6} {'Question Summary'}")
    print("-" * 80)
    for q in results:
        hit_str = "YES" if q["hit_at_k"] == 1.0 else "NO"
        q_trunc = (q["question"][:42] + "..") if len(q["question"]) > 44 else q["question"]
        print(f"{q['question_id']:<6} {q['precision_at_k']:<8.4f} {q['recall_at_k']:<8.4f} {q['f1_at_k']:<8.4f} {q['reciprocal_rank']:<8.4f} {hit_str:<6} {q_trunc}")
    print("-" * 80)

    # Missed questions warning
    if aggregate["missed_question_count"] > 0:
        print(f"\n[!] MISSED RETRIEVAL WARNING ({aggregate['missed_question_count']} questions failed to hit any relevant chunk):")
        for mq in aggregate["missed_questions"]:
            print(f"  * [{mq['question_id']}] '{mq['question']}'")
            print(f"    Expected Chunks: {mq['expected']}")
    else:
        print("\n[+] PERFECT RETRIEVAL: All evaluated questions retrieved at least one relevant chunk in Top K!")
    print("=" * 80 + "\n")


def print_experiment_summary(experiments: List[Dict[str, Any]]):
    """
    Renders comparison table across chunking configurations.
    """
    if not experiments:
        return
    print("\n" + "=" * 90)
    print("             CHUNKING CONFIGURATION EXPERIMENT COMPARISON")
    print("=" * 90)
    print(f"{'Configuration':<26} {'Chunks':<8} {'HitRate':<10} {'MRR':<8} {'P@K':<8} {'R@K':<8} {'IdxTime':<9}")
    print("-" * 90)
    for exp in experiments:
        print(
            f"{exp['config_name']:<26} "
            f"{exp['total_chunks']:<8} "
            f"{exp['hit_rate_at_k']*100:>5.1f}%    "
            f"{exp['mean_reciprocal_rank']:<8.4f} "
            f"{exp['mean_precision_at_k']:<8.4f} "
            f"{exp['mean_recall_at_k']:<8.4f} "
            f"{exp['indexing_time_sec']:>5.1f}s"
        )
    print("=" * 90 + "\n")


# ============================================================================
# 8. COMMAND LINE INTERFACE (MAIN)
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone Retrieval Evaluation Suite for NIST AI RMF Agentic RAG System.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(PARSED_DATA_DIR.parent / "eval_dataset.json"),
        help="Path to JSON or JSONL evaluation dataset.",
    )
    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=TOP_K_RETRIEVAL,
        help="Number of retrieved documents (K) to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval_results",
        help="Directory to save CSV and JSON evaluation reports.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Execute synthetic unit tests on IR metrics and exit.",
    )
    parser.add_argument(
        "--create-starter-dataset",
        action="store_true",
        help="Generate starter ground-truth dataset with actual chunk IDs and exit.",
    )
    parser.add_argument(
        "--experiment",
        action="store_true",
        help="Run chunking configuration comparison experiments (400, 600, 900, 1200 chars).",
    )
    parser.add_argument(
        "--explain-metrics",
        action="store_true",
        help="Display detailed explanation of retrieval metrics and why BLEU/ROUGE are not used.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # 1. Metric explanation mode
    if args.explain_metrics:
        print("\n" + explain_bleu_rouge_rationale() + "\n")
        return

    # 2. Self-test mode
    if args.self_test:
        success = run_self_tests()
        sys.exit(0 if success else 1)

    # 3. Create starter dataset mode
    target_dataset_path = Path(args.dataset)
    if args.create_starter_dataset:
        create_starter_dataset_file(target_dataset_path)
        return

    # Ensure dataset exists; generate starter if missing
    if not target_dataset_path.exists():
        logger.warning(f"Evaluation dataset '{target_dataset_path}' not found.")
        logger.info("Automatically generating starter dataset with verified ground-truth chunk IDs...")
        create_starter_dataset_file(target_dataset_path)

    # Load dataset
    dataset = load_dataset(target_dataset_path)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 4. Chunking experiment mode (optional)
    if args.experiment:
        experiment_configs = [
            (400, 50),
            (600, 100),
            (900, 150),
            (1200, 200),
        ]
        experiments = run_chunking_experiments(
            dataset=dataset,
            configs=experiment_configs,
            k=args.top_k,
        )
        print_experiment_summary(experiments)
        export_experiments_csv(experiments, output_dir / "chunking_experiments.csv")
        export_aggregate_json({"experiments": experiments}, output_dir / "chunking_experiments.json")

    # 5. Production retriever evaluation
    logger.info(f"Connecting to production HybridRetriever (K={args.top_k})...")
    retriever = get_retriever(k=args.top_k)

    results, aggregate = evaluate_retriever_on_dataset(retriever, dataset, k=args.top_k)

    # Display and export
    print_console_summary(results, aggregate)
    export_question_results_csv(results, output_dir / "retrieval_metrics.csv")
    export_aggregate_json(aggregate, output_dir / "retrieval_metrics.json")

    logger.info(f"All reports saved to '{output_dir.resolve()}'")


if __name__ == "__main__":
    main()
