"""
Interactive Conversational CLI Application for NIST AI RMF Agentic RAG System.
Features:
- Multi-turn conversational question answering
- Context-aware follow-up question resolution
- Source attribution (Document Name, Section, Page Number)
- Interactive commands (/reset, /history, /debug, /help, /exit)
- Beautiful Rich terminal formatting
"""

import sys
import os
import warnings
from typing import List
from pathlib import Path

# Suppress repetitive library warnings for clean CLI output
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_google_genai")
warnings.filterwarnings("ignore", message=r".*fixed sampling defaults.*")
warnings.filterwarnings("ignore", message=r".*unauthenticated requests to the HF Hub.*")

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.prompt import Prompt

from src.config import GEMINI_API_KEY, GEMINI_MODEL, GROQ_API_KEY, GROQ_MODEL
from src.llm_factory import extract_text_content
from src.graph import run_rag_pipeline
from src.ingest import ingest_documents
from src.vector_store import get_vector_store

console = Console()


def display_welcome_banner():
    """Print welcoming information and system configuration."""
    title = "[bold cyan]NIST AI Risk Management Framework (AI RMF)[/bold cyan]"
    subtitle = "[yellow]Agentic RAG System with LangGraph Orchestration[/yellow]"
    
    primary_status = f"[green]Active[/green] ({GEMINI_MODEL})" if (GEMINI_API_KEY and not GEMINI_API_KEY.startswith("your_")) else "[yellow]Key Not Set (add GEMINI_API_KEY to .env)[/yellow]"
    fallback_status = f"[green]Active[/green] ({GROQ_MODEL})" if (GROQ_API_KEY and not GROQ_API_KEY.startswith("your_")) else "[yellow]Not Configured[/yellow]"

    provider_info = f"""[bold]Primary Model:[/bold] Google Gemini [{primary_status}]
[bold]Fallback Model:[/bold] Groq [{fallback_status}]
[bold]Embeddings:[/bold] SentenceTransformers (all-MiniLM-L6-v2) [Local]"""

    content = f"""
{title}
{subtitle}

{provider_info}
[cyan]Knowledge Base:[/cyan]
  1. NIST AI RMF 1.0 (NIST AI 100-1)
  2. NIST AI RMF: Generative AI Profile (NIST AI 600-1)

[magenta]Commands:[/magenta]
  [bold]/reset[/bold]   - Reset conversation history
  [bold]/history[/bold] - View current conversation history
  [bold]/debug[/bold]   - Toggle intermediate LangGraph workflow tracing
  [bold]/help[/bold]    - Show example questions
  [bold]/exit[/bold]    - Exit the application
"""
    console.print(Panel(content.strip(), border_style="cyan"))


def display_help():
    """Display suggested questions demonstrating RAG and multi-turn capabilities."""
    help_text = """
[bold green]Sample Multi-Turn Questions to Try:[/bold green]

[bold]Turn 1 (Core Knowledge):[/bold]
  > "What are the four core functions of the NIST AI RMF?"

[bold]Turn 2 (Conversational Follow-up):[/bold]
  > "Which one is cross-cutting?"

[bold]Turn 3 (Generative AI Profile Expansion):[/bold]
  > "What are some recommended actions for this function in the Generative AI profile?"

[bold]Turn 4 (Specific Risks):[/bold]
  > "What risks are identified for CBRN and confabulation?"

[bold]Turn 5 (Failure Handling / Out-of-Scope Test):[/bold]
  > "How do I bake homemade pizza?" (Should trigger polite refusal without hallucinating)
"""
    console.print(Panel(help_text.strip(), title="Example Queries", border_style="green"))


def main():
    display_welcome_banner()

    # Verify vector store is initialized
    try:
        vs = get_vector_store()
        count = vs._collection.count()
        if count == 0:
            console.print("[yellow]ChromaDB is empty. Running initial document ingestion...[/yellow]")
            ingest_documents(force_reindex=False)
        else:
            console.print(f"[green][OK][/green] Vector database ready with [bold]{count}[/bold] chunks.\n")
    except Exception as e:
        console.print(f"[red][ERROR][/red] Could not initialize ChromaDB: {e}")
        return

    conversation_history: List[BaseMessage] = []
    debug_mode = False

    while True:
        try:
            user_input = Prompt.ask("\n[bold cyan]User[/bold cyan]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Exiting application. Goodbye![/yellow]")
            break

        if not user_input:
            continue

        cmd = user_input.lower()
        if cmd in ["/exit", "exit", "quit", ":q"]:
            console.print("[yellow]Goodbye![/yellow]")
            break
        elif cmd == "/reset":
            conversation_history.clear()
            console.print("[green]Conversation history has been reset.[/green]")
            continue
        elif cmd == "/debug":
            debug_mode = not debug_mode
            status = "[green]ON[/green]" if debug_mode else "[red]OFF[/red]"
            console.print(f"Debug mode is now {status}.")
            continue
        elif cmd == "/history":
            if not conversation_history:
                console.print("[yellow]No conversation history yet.[/yellow]")
            else:
                table = Table(title="Conversation History")
                table.add_column("Turn", style="cyan")
                table.add_column("Speaker", style="magenta")
                table.add_column("Snippet", style="white")
                for i, msg in enumerate(conversation_history):
                    speaker = "User" if isinstance(msg, HumanMessage) else "Assistant"
                    raw_text = extract_text_content(msg.content)
                    snippet = raw_text[:90] + ("..." if len(raw_text) > 90 else "")
                    table.add_row(str((i // 2) + 1), speaker, snippet)
                console.print(table)
            continue
        elif cmd == "/help":
            display_help()
            continue

        # Execute LangGraph Pipeline
        console.print("[dim]Orchestrating LangGraph RAG workflow...[/dim]")
        try:
            result = run_rag_pipeline(user_input, conversation_history)
            
            # Update conversational history with the user query and assistant response
            conversation_history = result.get("messages", [])

            if debug_mode:
                debug_info = f"""
[bold]Standalone Query:[/bold] {result.get('standalone_query')}
[bold]Retries Attempted:[/bold] {result.get('retry_count')}
[bold]Workflow Status:[/bold] {result.get('retrieval_status')}
[bold]Retrieved Chunks:[/bold] {len(result.get('documents', []))}
[bold]Graded Relevant Chunks:[/bold] {len(result.get('relevant_documents', []))}
"""
                console.print(Panel(debug_info.strip(), title="Debug Information", border_style="magenta"))

            # Display formatted assistant answer
            answer_text = result.get("generation", "No answer generated.")
            console.print("\n[bold magenta]Assistant:[/bold magenta]")
            console.print(Markdown(answer_text))

        except Exception as e:
            console.print(f"[bold red]Pipeline Error:[/bold red] {e}")


if __name__ == "__main__":
    main()
