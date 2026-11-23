"""
LangGraph Nodes for NIST AI RMF Agentic RAG System.

Nodes:
1. contextualize_query: Reformulates follow-ups using chat history (capped to last N turns, sources stripped)
   and applies input guard for short/trivial non-follow-up queries.
2. retrieve: Retrieves candidate chunks using hybrid retrieval (Chroma + BM25 + Exact-ID shortcut).
3. grade_documents: Evaluates relevance in ONE batched LLM call with exponential backoff on 429/5xx.
4. rewrite_query: Reformulates query on retrieval failure without injecting framework terms into out-of-scope queries.
5. generate_answer: Grounded generation; detects NO_INFO: refusals and cites ONLY chunks actually used.
6. handle_insufficient_context: Refusal node with Sources: None.
"""

import re
import json
import logging
from typing import Dict, Any, List, Optional

from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.documents import Document

from src.state import GraphState
from src.vector_store import get_retriever
from src.llm_factory import get_llm, extract_text_content
from src.config import TOP_K_RETRIEVAL, MAX_HISTORY_TURNS

logger = logging.getLogger("rag.nodes")


def format_docs_for_prompt(docs: List[Document]) -> str:
    """Format document chunks with page and section headers for LLM prompt."""
    formatted_chunks = []
    for idx, doc in enumerate(docs):
        doc_name = doc.metadata.get("document_name", "NIST Document")
        section = doc.metadata.get("section_path") or doc.metadata.get("section", "General")
        printed_p = doc.metadata.get("printed_page")
        pdf_p = doc.metadata.get("pdf_page", doc.metadata.get("page_number", "?"))
        p_str = f"Page {printed_p} (PDF p.{pdf_p})" if printed_p is not None and str(printed_p) != "None" else f"PDF p.{pdf_p}"
        formatted_chunks.append(
            f"--- [Chunk {idx} | {doc_name} | {section} | {p_str}] ---\n"
            f"{doc.page_content}\n"
        )
    return "\n".join(formatted_chunks)


def extract_sources_from_docs(docs: List[Document]) -> List[str]:
    """
    Format unique source citations strictly from chunk metadata.
    Format: "<document_name> - <section_path | subcategory id | action id | figure id>, Page <printed_page> (PDF p.<pdf_page>)"
    """
    seen = set()
    sources = []
    for doc in docs:
        doc_name = doc.metadata.get("document_name", "NIST AI RMF")
        node_type = doc.metadata.get("node_type", "prose")

        if node_type == "figure" and doc.metadata.get("figure_id"):
            fnum = doc.metadata.get("figure_id", "").replace("fig_", "")
            identifier = f"Figure {fnum}"
        elif doc.metadata.get("action_id"):
            identifier = doc.metadata["action_id"]
        elif doc.metadata.get("subcategory_id"):
            identifier = doc.metadata["subcategory_id"]
        else:
            identifier = doc.metadata.get("section_path") or doc.metadata.get("section", "General")

        printed_p = doc.metadata.get("printed_page")
        pdf_p = doc.metadata.get("pdf_page", doc.metadata.get("page_number", "N/A"))
        pr_str = str(printed_p) if printed_p is not None and str(printed_p) != "None" else "None"

        citation = f"{doc_name} - {identifier}, Page {pr_str} (PDF p.{pdf_p})"
        if citation not in seen:
            seen.add(citation)
            sources.append(citation)
    return sources


# ============================================================================
# NODE 1: Contextualize Query & Input Guard
# ============================================================================

def contextualize_query_node(state: GraphState) -> Dict[str, Any]:
    """
    Formulates a standalone search query if conversational history exists.
    Applies Input Guard for empty or trivial queries (< 3 meaningful words and not a follow-up).
    Limits history to MAX_HISTORY_TURNS and passes only answer text (never Sources blocks).
    """
    messages = state.get("messages", [])
    query = state.get("query", "").strip()
    retry_count = state.get("retry_count", 0)

    # 1. Input Guard: Check if standalone query has fewer than 3 meaningful words and is not a follow-up
    meaningful_words = [w for w in re.findall(r"\w+", query.lower()) if len(w) >= 2]
    is_follow_up = len(messages) > 1

    if not is_follow_up and (len(meaningful_words) < 3 or query in ("?", "hi", "hello", "hey")):
        logger.info(f"Input guard triggered for query: '{query}'")
        return {
            "standalone_query": query,
            "retry_count": retry_count,
            "retrieval_status": "input_guard_prompt",
        }

    # If no prior conversation history, current query is already standalone
    if not is_follow_up:
        return {
            "standalone_query": query,
            "retry_count": retry_count,
            "retrieval_status": "initialized",
        }

    # 2. Conversational Memory: Take only last N turns and strip Sources block from AIMessages
    recent_messages = messages[-(MAX_HISTORY_TURNS * 2) : -1] if len(messages) > 1 else []
    history_lines = []
    for msg in recent_messages:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        raw_text = extract_text_content(msg.content)
        # Strip Sources block from Assistant text
        if "Sources:" in raw_text:
            raw_text = raw_text.split("Sources:")[0]
        raw_text = raw_text.replace("Answer:", "").strip()
        history_lines.append(f"{role}: {raw_text}")

    history_str = "\n".join(history_lines)

    system_prompt = (
        "You are an expert conversational query reformulation assistant for the NIST AI Risk Management Framework (AI RMF 1.0) "
        "and the Generative AI Profile (NIST AI 600-1).\n"
        "Given the chat history and the latest user question, rephrase the question into a clear, self-contained standalone search query.\n\n"
        "Guidelines:\n"
        "- Resolve pronouns ('which one', 'it', 'this function', 'they') using prior assistant responses.\n"
        "- If the latest user question is a complete topic switch (e.g. 'Now, completely different: what is confabulation?'), "
        "do NOT include concepts from earlier conversation turns.\n"
        "- Do NOT answer the question. Only return the reformulated standalone search query.\n"
        "- If the question is already standalone, return it unchanged."
    )

    user_prompt = (
        f"Chat History:\n{history_str}\n\n"
        f"Latest User Question: {query}\n\n"
        f"Standalone Question:"
    )

    llm = get_llm(temperature=0.0)
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        standalone = extract_text_content(response)
        if standalone.startswith('"') and standalone.endswith('"'):
            standalone = standalone[1:-1].strip()
    except Exception as e:
        logger.warning(f"Failed to contextualize query ({e}), using raw query.")
        standalone = query

    logger.info(f"Query resolved to: '{standalone}'")
    return {
        "standalone_query": standalone,
        "retry_count": retry_count,
        "retrieval_status": "contextualized"
    }


# ============================================================================
# NODE 2: Retrieve (Chroma + BM25 Hybrid + Exact-ID Shortcut)
# ============================================================================

def retrieve_node(state: GraphState) -> Dict[str, Any]:
    """
    Retrieves top candidate document chunks using HybridRetriever (Chroma similarity + BM25 + Exact-ID shortcut).
    """
    standalone_query = state.get("standalone_query") or state.get("query", "")
    logger.info(f"Retrieving candidate chunks for: '{standalone_query}'...")

    retriever = get_retriever(k=TOP_K_RETRIEVAL)
    docs = retriever.invoke(standalone_query)

    logger.info(f"Retrieved {len(docs)} candidate chunks.")
    return {
        "documents": docs,
        "retrieval_status": "retrieved"
    }


# ============================================================================
# NODE 3: Grade Documents (ONE Batched LLM Call with Retry)
# ============================================================================

@retry(
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(Exception),
    reraise=False,
)
def _call_batched_grader(llm, messages):
    return llm.invoke(messages)


def grade_documents_node(state: GraphState) -> Dict[str, Any]:
    """
    Evaluates retrieved chunks in ONE batched LLM call returning {"relevant_indices": [...]}.
    Applies exponential backoff on 429/5xx errors on top of Gemini->Groq fallback.
    """
    standalone_query = state.get("standalone_query") or state.get("query", "")
    docs = state.get("documents", [])

    if not docs:
        return {
            "relevant_documents": [],
            "retrieval_status": "graded_insufficient"
        }

    # If top chunk is an exact-ID match, preserve it automatically
    shortcut_indices = set()
    for idx, doc in enumerate(docs):
        if doc.metadata.get("action_id") and doc.metadata["action_id"].upper() in standalone_query.upper():
            shortcut_indices.add(idx)
        if doc.metadata.get("subcategory_id") and doc.metadata["subcategory_id"].upper() in standalone_query.upper():
            shortcut_indices.add(idx)
        if doc.metadata.get("category_id") and doc.metadata["category_id"].upper() in standalone_query.upper():
            shortcut_indices.add(idx)
        if doc.metadata.get("function") and doc.metadata["function"].upper() in standalone_query.upper() and any(w in standalone_query.lower() for w in ["subcategory", "subcategories", "category", "categories"]):
            shortcut_indices.add(idx)
        if doc.metadata.get("figure_id"):
            fnum = doc.metadata["figure_id"].replace("fig_", "")
            if f"figure {fnum}" in standalone_query.lower() or f"fig. {fnum}" in standalone_query.lower():
                shortcut_indices.add(idx)

    chunks_text = []
    for idx, doc in enumerate(docs):
        preview = doc.page_content.replace("\n", " ").strip()
        if len(preview) > 1200:
            preview = preview[:1200] + "..."
        chunks_text.append(f"[Chunk {idx}]: {preview}")

    system_prompt = (
        "You are a strict relevance grader for the NIST AI Risk Management Framework (AI RMF 1.0) "
        "and Generative AI Profile (NIST AI 600-1).\n"
        "Evaluate which chunks are relevant to the user query.\n"
        "Respond ONLY with a JSON object: {\"relevant_indices\": [indices of relevant chunks]}.\n"
        "If NO chunks are relevant or the query is completely unrelated/out-of-scope, return {\"relevant_indices\": []}."
    )

    user_prompt = (
        f"Query: {standalone_query}\n\n"
        f"Retrieved Chunks:\n" + "\n".join(chunks_text) + "\n\n"
        "JSON Response:"
    )

    llm = get_llm(temperature=0.0)
    relevant_indices = list(shortcut_indices)

    try:
        res = _call_batched_grader(llm, [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        content = extract_text_content(res)
        # Parse JSON
        match = re.search(r"\{.*\}", content, re.S)
        if match:
            data = json.loads(match.group(0))
            parsed_indices = data.get("relevant_indices", [])
            for i in parsed_indices:
                if isinstance(i, int) and 0 <= i < len(docs):
                    relevant_indices.append(i)
    except Exception as e:
        logger.warning(f"Batched grader failed ({e}), retaining candidate chunks.")
        relevant_indices = list(range(len(docs)))

    relevant_indices = sorted(list(set(relevant_indices)))
    relevant_docs = [docs[i] for i in relevant_indices]

    logger.info(f"Grading complete: {len(relevant_docs)} / {len(docs)} chunks determined relevant (indices: {relevant_indices}).")
    status = "graded_relevant" if len(relevant_docs) > 0 else "graded_insufficient"
    return {
        "relevant_documents": relevant_docs,
        "retrieval_status": status
    }


# ============================================================================
# NODE 4: Rewrite Query
# ============================================================================

def rewrite_query_node(state: GraphState) -> Dict[str, Any]:
    """
    Reformulates query on retrieval failure.
    CRITICAL: Does NOT inject GOVERN/MAP/MEASURE/MANAGE or other framework terms
    unless the original query already relates to the framework. Out-of-scope queries
    keep their original terms.
    """
    standalone_query = state.get("standalone_query") or state.get("query", "")
    retry_count = state.get("retry_count", 0) + 1

    # Check if query is clearly out of scope
    ai_keywords = [
        "ai", "model", "risk", "govern", "map", "measure", "manage", "framework",
        "nist", "profile", "actor", "bias", "fairness", "safety", "trust", "cbrn",
        "confabulation", "intellectual", "privacy", "security", "data", "figure", "table"
    ]
    query_tokens = [w.lower() for w in re.findall(r"\w+", standalone_query)]
    is_related = any(k in query_tokens for k in ai_keywords)

    if not is_related:
        logger.info(f"Query is out-of-scope; retaining original query without framework injection: '{standalone_query}'")
        return {
            "standalone_query": standalone_query,
            "retry_count": retry_count,
            "retrieval_status": "rewritten"
        }

    system_prompt = (
        "You are an AI search query optimization assistant for the NIST AI Risk Management Framework.\n"
        "Rewrite the query using technical NIST terminology for improved retrieval.\n"
        "Return ONLY the rewritten query."
    )
    user_prompt = f"Original Query: {standalone_query}\nOptimized Query:"

    llm = get_llm(temperature=0.0)
    try:
        res = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        rewritten = extract_text_content(res)
        if rewritten.startswith('"') and rewritten.endswith('"'):
            rewritten = rewritten[1:-1].strip()
    except Exception:
        rewritten = standalone_query

    logger.info(f"Optimized query: '{rewritten}'")
    return {
        "standalone_query": rewritten,
        "retry_count": retry_count,
        "retrieval_status": "rewritten"
    }


# ============================================================================
# NODE 5: Generate Answer (Over-citation Fix & Grounded Citations)
# ============================================================================

def generate_answer_node(state: GraphState) -> Dict[str, Any]:
    """
    Synthesizes answer grounded in retrieved chunks.
    Instructs model to begin with NO_INFO: if context does not answer the question.
    Extracts Used Chunks indices so only chunks actually used in the answer are cited.
    """
    standalone_query = state.get("standalone_query") or state.get("query", "")
    docs = state.get("relevant_documents") or state.get("documents", [])
    messages = list(state.get("messages", []))

    if not docs:
        return handle_insufficient_context_node(state)

    context_str = format_docs_for_prompt(docs)

    system_prompt = (
        "You are a specialized AI assistant answering questions about the NIST AI Risk Management Framework (AI RMF 1.0) "
        "and the Generative Artificial Intelligence Profile (NIST AI 600-1).\n\n"
        "CRITICAL INSTRUCTIONS:\n"
        "1. Answer the question accurately, comprehensively, and factually using ONLY the provided retrieved context.\n"
        "2. If the provided context does NOT contain information to answer the question, or if the question is out of scope, "
        "begin your response with 'NO_INFO:' followed by a polite explanation.\n"
        "3. When answering, cite specific Action IDs (e.g. GV-3.2-001), Subcategories (e.g. MEASURE 2.11), or Figures when mentioned in the context.\n"
        "4. At the very end of your response, indicate which chunks directly supported your answer in the exact format:\n"
        "Used Chunks: [0, 1]\n"
        "Do NOT write your own 'Sources:' section; sources will be generated programmatically."
    )

    user_prompt = (
        f"Retrieved Document Context:\n{context_str}\n\n"
        f"User Question: {standalone_query}\n\n"
        f"Answer:"
    )

    llm = get_llm(temperature=0.0)
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        raw_answer = extract_text_content(response)

        # Check for NO_INFO: refusal
        if raw_answer.strip().startswith("NO_INFO:"):
            logger.info("Model indicated NO_INFO: routing to refusal format.")
            return handle_insufficient_context_node(state)

        # Extract Used Chunks: [...]
        used_match = re.search(r"Used Chunks:\s*\[([\d\s,]+)\]", raw_answer, re.I)
        used_indices = []
        if used_match:
            try:
                used_indices = [int(x.strip()) for x in used_match.group(1).split(",") if x.strip().isdigit()]
            except Exception:
                pass
            raw_answer = re.sub(r"Used Chunks:\s*\[.*?\]", "", raw_answer).strip()

        # Clean any manual Sources written by model
        if "Sources:" in raw_answer:
            raw_answer = raw_answer.split("Sources:")[0].strip()

        used_docs = [docs[i] for i in used_indices if 0 <= i < len(docs)]
        if not used_docs:
            used_docs = docs

        sources_list = extract_sources_from_docs(used_docs)
        sources_formatted = "\n".join([f"- {src}" for src in sources_list]) if sources_list else "- None"

        if not raw_answer.startswith("Answer:"):
            final_output = f"Answer:\n{raw_answer}\n\nSources:\n{sources_formatted}"
        else:
            final_output = f"{raw_answer}\n\nSources:\n{sources_formatted}"

    except Exception as e:
        logger.warning(f"Answer generation error ({e}).")
        sources_list = extract_sources_from_docs(docs)
        sources_formatted = "\n".join([f"- {src}" for src in sources_list]) if sources_list else "- None"
        final_output = (
            f"Answer:\nBased on the retrieved NIST AI RMF documents:\n"
            f"{context_str[:400]}...\n\nSources:\n{sources_formatted}"
        )

    messages.append(AIMessage(content=final_output))
    return {
        "generation": final_output,
        "sources": sources_list,
        "messages": messages,
        "retrieval_status": "answered"
    }


# ============================================================================
# NODE 6: Handle Insufficient Context (Polite Refusal)
# ============================================================================

def handle_insufficient_context_node(state: GraphState) -> Dict[str, Any]:
    """
    Executed when retrieval yields no relevant context or when input guard triggers.
    Returns standard refusal with Sources: None.
    """
    messages = list(state.get("messages", []))
    status = state.get("retrieval_status", "")

    if status == "input_guard_prompt":
        response_text = (
            "Answer:\n"
            "Please ask a question about the NIST AI RMF documents.\n\n"
            "Sources:\n"
            "- None"
        )
    else:
        response_text = (
            "Answer:\n"
            "I could not find sufficient information in the official NIST AI Risk Management Framework (AI RMF 1.0) "
            "or the Generative AI Profile (NIST AI 600-1) to answer your question.\n\n"
            "The knowledge base is restricted to official NIST guidance on AI risk management, including the four core "
            "functions (Govern, Map, Measure, Manage), trustworthiness characteristics, and generative AI risk profiles.\n\n"
            "Sources:\n"
            "- None"
        )

    messages.append(AIMessage(content=response_text))
    return {
        "generation": response_text,
        "sources": [],
        "messages": messages,
        "retrieval_status": "insufficient_context_handled"
    }
