"""
LLM Factory for NIST AI RMF Agentic RAG System.
Configured as:
- Primary: Google Gemini (gemini-2.0-flash-lite / gemini-1.5-flash)
- Fallback: Groq (openai/gpt-oss-120b / llama-3.3-70b-versatile)
- Offline Fallback: Deterministic Mock LLM (if neither key is configured)
"""

import os
import warnings
from typing import Optional, List, Any

# Suppress repetitive LangChain/Gemini SDK fixed sampling warning
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_google_genai")
warnings.filterwarnings("ignore", message=r".*fixed sampling defaults.*")

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration

from src.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GROQ_API_KEY,
    GROQ_MODEL,
    OPENAI_API_KEY,
    OPENAI_MODEL,
)


def extract_text_content(response: Any) -> str:
    """
    Safely extract plain text from an AIMessage, BaseMessage, dict list, or string.
    Handles both standard strings and Gemini-style list of content blocks:
    e.g. [{'type': 'text', 'text': '...'}, ...]
    """
    if hasattr(response, "content"):
        content = response.content
    else:
        content = response

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            elif isinstance(item, str):
                parts.append(item)
            elif hasattr(item, "text"):
                parts.append(str(item.text))
            else:
                parts.append(str(item))
        return "".join(parts).strip()

    return str(content).strip()


class OfflineMockLLM(BaseChatModel):
    """
    Fallback deterministic LLM for environments where no cloud API key has been configured.
    Allows running the full pipeline, vector search, and state graph locally without crashing.
    """
    model_name: str = "mock-offline-llm"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        last_message = extract_text_content(messages[-1]) if messages else ""
        system_content = extract_text_content(messages[0]) if len(messages) > 1 else ""
        full_text = (system_content + "\n" + last_message).lower()

        # 1. Contextualize query
        if "rephrase the question into a clear, self-contained standalone" in full_text:
            if "cross-cutting" in full_text or "which one" in full_text:
                content = "Which of the four NIST AI RMF core functions is cross-cutting?"
            elif "generative ai" in full_text or "actions" in full_text:
                content = "What are the recommended actions for the Govern function in the NIST Generative AI profile?"
            else:
                content = last_message.split("Latest User Question:")[-1].split("Standalone Question:")[0].strip()

        # 2. Document Grading
        elif "strict relevance evaluator" in full_text:
            if "cake" in full_text or "pizza" in full_text or "recipe" in full_text or "cookie" in full_text:
                content = '{"relevant": false}'
            else:
                content = '{"relevant": true}'

        # 3. Query Rewriting
        elif "query optimization" in full_text or "rewrite the query" in full_text:
            if "cake" in full_text or "pizza" in full_text or "recipe" in full_text or "cookie" in full_text:
                content = "chocolate cake baking instructions recipe"
            else:
                content = "NIST AI RMF core functions Govern Map Measure Manage cross-cutting"

        # 4. Grounded Answer Generation
        else:
            if "cross-cutting" in full_text or "which one" in full_text:
                content = (
                    "Answer:\n"
                    "In the NIST AI Risk Management Framework, the GOVERN function is designed to be cross-cutting. "
                    "Governance is infused throughout AI risk management and informs how organizations prioritize "
                    "and carry out the other three core functions: MAP, MEASURE, and MANAGE. It cultivates and enforces "
                    "a culture of risk management within AI-developing and AI-deploying organizations."
                )
            elif "recommended actions" in full_text or "generative ai" in full_text:
                content = (
                    "Answer:\n"
                    "In the NIST AI RMF Generative AI Profile (NIST AI 600-1), the GOVERN function specifies multiple "
                    "tailored action items (such as GV-1.1 through GV-6.2). Key recommended actions include aligning GAI "
                    "development with applicable regulations, establishing transparent acceptable use policies, establishing "
                    "processes to decommission or halt systems posing severe risks (such as CBRN content), and establishing "
                    "roles and incident response teams for generative AI risks."
                )
            elif "four functions" in full_text or "core functions" in full_text:
                content = (
                    "Answer:\n"
                    "The four core functions of the NIST AI Risk Management Framework (AI RMF 1.0) are "
                    "Govern, Map, Measure, and Manage. These functions provide a structured, outcomes-based approach "
                    "to identifying, assessing, and mitigating risks throughout the entire AI lifecycle."
                )
            else:
                content = (
                    "Answer:\n"
                    "The NIST AI Risk Management Framework provides guidance for managing risks in AI systems across "
                    "trustworthiness characteristics including validity, reliability, safety, security, resilience, "
                    "accountability, transparency, explainability, privacy, and fairness."
                )

        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    @property
    def _llm_type(self) -> str:
        return "offline_mock"


def get_gemini_llm(temperature: float = 0.0) -> Optional[BaseChatModel]:
    """Initialize Google Gemini chat model."""
    if not GEMINI_API_KEY or GEMINI_API_KEY.startswith("your_"):
        return None
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GEMINI_API_KEY,
            temperature=temperature,
        )
    except Exception as e:
        print(f"[WARN] Failed to initialize Google Gemini ({e}).")
        return None


def get_groq_llm(temperature: float = 0.0) -> Optional[BaseChatModel]:
    """Initialize Groq chat model."""
    if not GROQ_API_KEY or GROQ_API_KEY.startswith("your_"):
        return None
    try:
        from langchain_groq import ChatGroq
        return ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name=GROQ_MODEL,
            temperature=temperature,
        )
    except Exception as e:
        print(f"[WARN] Failed to initialize ChatGroq ({e}).")
        return None


from src.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GROQ_API_KEY,
    GROQ_MODEL,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    PRIMARY_PROVIDER,
    FALLBACK_PROVIDER,
)

import logging
logger = logging.getLogger("rag.llm_factory")


def get_openai_llm(temperature: float = 0.0) -> Optional[BaseChatModel]:
    """Initialize OpenAI chat model."""
    if not OPENAI_API_KEY or OPENAI_API_KEY.startswith("your_"):
        return None
    try:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=OPENAI_MODEL,
            api_key=OPENAI_API_KEY,
            temperature=temperature,
        )
    except Exception as e:
        logger.warning(f"Failed to initialize ChatOpenAI ({e}).")
        return None


def get_provider_llm(provider_name: str, temperature: float = 0.0) -> Optional[BaseChatModel]:
    """Resolve and instantiate a chat model based on provider name."""
    p = (provider_name or "").lower().strip()
    if p in ("gemini", "google"):
        return get_gemini_llm(temperature)
    elif p == "groq":
        return get_groq_llm(temperature)
    elif p == "openai":
        return get_openai_llm(temperature)
    return None


_CALL_COUNT = 0


class CountedLLM:
    """Wrapper around BaseChatModel / Runnable to track exact LLM invocations."""
    def __init__(self, inner: BaseChatModel):
        self._inner = inner

    def invoke(self, *args, **kwargs):
        global _CALL_COUNT
        _CALL_COUNT += 1
        return self._inner.invoke(*args, **kwargs)

    def with_fallbacks(self, fallbacks, **kwargs):
        return CountedLLM(self._inner.with_fallbacks(fallbacks, **kwargs))

    def __getattr__(self, name):
        return getattr(self._inner, name)


def get_llm_call_count() -> int:
    global _CALL_COUNT
    return _CALL_COUNT


def reset_llm_call_count() -> None:
    global _CALL_COUNT
    _CALL_COUNT = 0


def get_llm(temperature: float = 0.0) -> BaseChatModel:
    """
    Factory function providing:
    1. Primary LLM configured by PRIMARY_PROVIDER (default: Gemini)
    2. Fallback LLM configured by FALLBACK_PROVIDER (default: Groq)
    3. Automatic runtime fallback (Primary -> Fallback -> Mock)
    """
    primary = get_provider_llm(PRIMARY_PROVIDER, temperature)
    fallback = get_provider_llm(FALLBACK_PROVIDER, temperature)

    if primary and fallback and primary != fallback:
        logger.info(f"Active: Primary={PRIMARY_PROVIDER} with Fallback={FALLBACK_PROVIDER}")
        model = primary.with_fallbacks([fallback])
    elif primary:
        logger.info(f"Active: Primary={PRIMARY_PROVIDER}")
        model = primary
    elif fallback:
        logger.info(f"Active: Fallback={FALLBACK_PROVIDER} as Primary")
        model = fallback
    else:
        # Fallback to standard check
        gemini = get_gemini_llm(temperature)
        if gemini:
            model = gemini
        else:
            groq = get_groq_llm(temperature)
            if groq:
                model = groq
            else:
                logger.info("Active: Deterministic Offline Mode")
                model = OfflineMockLLM()

    return CountedLLM(model)
