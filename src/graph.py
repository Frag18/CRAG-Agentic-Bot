"""
LangGraph Workflow Assembly for the NIST AI RMF Agentic RAG System.
Connects State, Nodes, and Conditional Edges into a compiled runnable graph.
"""

from typing import Dict, Any, List, Optional
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from src.state import GraphState
from src.nodes import (
    contextualize_query_node,
    retrieve_node,
    grade_documents_node,
    rewrite_query_node,
    generate_answer_node,
    handle_insufficient_context_node,
)
from src.edges import decide_to_generate, check_input_guard


def build_graph() -> StateGraph:
    """
    Construct the Agentic RAG StateGraph.
    """
    workflow = StateGraph(GraphState)

    # 1. Add all functional nodes
    workflow.add_node("contextualize_query", contextualize_query_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("rewrite_query", rewrite_query_node)
    workflow.add_node("generate_answer", generate_answer_node)
    workflow.add_node("handle_insufficient_context", handle_insufficient_context_node)

    # 2. Add structural edges
    # START -> Contextualizer
    workflow.add_edge(START, "contextualize_query")

    # Contextualizer -> Input Guard -> Retriever or Insufficient Context
    workflow.add_conditional_edges(
        "contextualize_query",
        check_input_guard,
        {
            "retrieve": "retrieve",
            "handle_insufficient_context": "handle_insufficient_context",
        },
    )

    # Retriever -> Document Grader
    workflow.add_edge("retrieve", "grade_documents")

    # Document Grader -> Conditional Router
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "generate_answer": "generate_answer",
            "rewrite_query": "rewrite_query",
            "handle_insufficient_context": "handle_insufficient_context",
        },
    )

    # Rewrite Query loops back to Retrieve
    workflow.add_edge("rewrite_query", "retrieve")

    # Terminal edges
    workflow.add_edge("generate_answer", END)
    workflow.add_edge("handle_insufficient_context", END)

    return workflow


# Compiled singleton graph
_compiled_app = None


def get_graph():
    """Returns the compiled LangGraph application."""
    global _compiled_app
    if _compiled_app is None:
        workflow = build_graph()
        _compiled_app = workflow.compile()
    return _compiled_app


def run_rag_pipeline(
    query: str,
    conversation_history: Optional[List[BaseMessage]] = None
) -> Dict[str, Any]:
    """
    High-level entrypoint to execute the LangGraph RAG pipeline for a given query
    with conversation history preservation.

    Parameters:
    - query: User natural language input.
    - conversation_history: List of prior HumanMessage and AIMessage objects.

    Returns:
    - Dictionary containing 'generation', 'sources', 'messages', 'standalone_query', etc.
    """
    app = get_graph()
    messages = list(conversation_history or [])
    messages.append(HumanMessage(content=query))

    initial_state: GraphState = {
        "messages": messages,
        "query": query,
        "standalone_query": query,
        "documents": [],
        "relevant_documents": [],
        "generation": "",
        "sources": [],
        "retry_count": 0,
        "retrieval_status": "started",
    }

    final_state = app.invoke(initial_state)
    return final_state
