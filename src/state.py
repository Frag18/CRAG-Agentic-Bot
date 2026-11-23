"""
State definition for the NIST AI RMF Agentic RAG System using LangGraph.
"""

from typing import List, Dict, Any, Optional, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document


class GraphState(TypedDict):
    """
    State maintained across nodes in the LangGraph RAG workflow.

    Fields:
    - messages: Conversational message history (HumanMessage, AIMessage)
    - query: Raw query submitted by the user in current turn
    - standalone_query: Contextualized standalone query resolved against conversation history
    - documents: Raw candidate chunks retrieved from ChromaDB
    - relevant_documents: Candidate chunks verified and graded as relevant by the grader
    - generation: Final answer synthesized by the generator node
    - sources: Deduplicated list of source citation strings (Doc Name, Section, Page)
    - retry_count: Count of query rewriting & retrieval attempts (prevents infinite loops)
    - retrieval_status: Workflow status: 'success' | 'insufficient' | 'rewritten' | 'fallback'
    """
    messages: List[BaseMessage]
    query: str
    standalone_query: str
    documents: List[Document]
    relevant_documents: List[Document]
    generation: str
    sources: List[str]
    retry_count: int
    retrieval_status: str
