"""
Conditional Edges and Routing Logic for LangGraph RAG Workflow.
"""

from typing import Literal
from src.state import GraphState
from src.config import MAX_RETRIES


def check_input_guard(state: GraphState) -> Literal["retrieve", "handle_insufficient_context"]:
    """
    Check if the input guard was triggered in contextualize_query.
    If triggered, bypass retrieval and route directly to handle_insufficient_context.
    """
    if state.get("retrieval_status") == "input_guard_prompt":
        return "handle_insufficient_context"
    return "retrieve"


def decide_to_generate(state: GraphState) -> Literal["generate_answer", "rewrite_query", "handle_insufficient_context"]:
    """
    Evaluates graded document results and decides the next step in the workflow:
    - If relevant documents exist -> route to 'generate_answer'
    - If no relevant documents exist and retry limit not reached -> route to 'rewrite_query'
    - If retry limit exceeded and still insufficient -> route to 'handle_insufficient_context'
    """
    relevant_docs = state.get("relevant_documents", [])
    retry_count = state.get("retry_count", 0)

    if len(relevant_docs) > 0:
        return "generate_answer"

    if retry_count < MAX_RETRIES:
        return "rewrite_query"

    return "handle_insufficient_context"
