# NIST AI Risk Management Framework: Agentic RAG System

An agentic question-answering system using **LangGraph** to answer questions about the **NIST AI Risk Management Framework (AI RMF 1.0, NIST AI 100-1)** and the **NIST AI RMF Generative AI Profile (NIST AI 600-1)**.

LlamaIndex is used only in the ingestion pipeline for heading-aware node parsing; all workflow orchestration is LangGraph.

Built with a 6-node **Corrective RAG (CRAG)** architecture, conversational multi-turn follow-up resolution, single-call batched document relevance grading with exponential backoff, query reformulation retry loops, deterministic PDF parsing and multi-page table reconstruction, figure vision descriptions, hybrid dense + BM25 retrieval with exact-ID shortcuts, and strict metadata-grounded source attribution.

---

## Architecture Diagram

```
                        User Input
                            │
                            ▼
               ┌───────────────────────────┐
               │    contextualize_query    │  <── Resolves follow-ups (history capped to 6 turns)
               └─────────────┬─────────────┘
                             │
                     [ check_input_guard ]
                      /                 \
         (Meaningful query)            (Trivial query: <3 words, "?", "hi")
                    │                                    │
                    ▼                                    ▼
       ┌───────────────────────────┐       ┌─────────────────────────────┐
       │         retrieve          │       │ handle_insufficient_context │ ──► [ END ]
       └─────────────┬─────────────┘       └─────────────────────────────┘     (Sources: None,
                     │                                                          0 LLM calls)
                     ▼
       ┌───────────────────────────┐
       │      grade_documents      │  <── 1 Batched LLM call returning {"relevant_indices": [...]}
       └─────────────┬─────────────┘      with exponential backoff on 429/5xx errors
                     │
                     ▼
            [ decide_to_generate ]
             /         │         \
 (Sufficient)          │          (Insufficient & Retries Exhausted)
       ▼               │                       ▼
┌──────────────────┐   │         ┌─────────────────────────────┐
│ generate_answer  │   │         │ handle_insufficient_context │
└─────────┬────────┘   │         └──────────────┬──────────────┘
          │            │                        │
          │            ▼                        │
          │  (Insufficient & Retries < 1)       │
          │  ┌──────────────────┐               │
          │  │  rewrite_query   │ ──────────────┘ (Loops back to retrieve without
          │  └──────────────────┘                  injecting ungrounded terms)
          ▼
      [  END  ]
```

---

## 1. Installation Instructions

### Prerequisites
- Python 3.10 to 3.13 recommended (Windows, macOS, or Linux)
- Git (optional)

### Setup Virtual Environment
```bash
# Clone or navigate to the project directory
cd assignment_whitecrowresearch

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (CMD):
.venv\Scripts\activate.bat
# Linux / macOS:
source .venv/bin/activate

# Upgrade packaging tools
python -m pip install --upgrade pip setuptools wheel

# Install dependencies
pip install -r requirements.txt
```

---

## 2. Configuration Instructions

Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

Edit `.env` to configure your API keys:

```env
# Primary LLM: Google Gemini (e.g. gemini-2.0-flash-lite / gemini-1.5-flash / gemini-3.5-flash-lite)
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.0-flash-lite

# Fallback LLM: Groq (e.g. openai/gpt-oss-120b or llama-3.3-70b-versatile)
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-120b

# Provider Selection
PRIMARY_PROVIDER=gemini
FALLBACK_PROVIDER=groq

# Embeddings (Runs locally with sentence-transformers by default)
EMBEDDING_PROVIDER=huggingface
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2

# Chunking & Retrieval Parameters
CHUNK_SIZE=900
CHUNK_OVERLAP=150
TOP_K_RETRIEVAL=6
MAX_RETRIES=1
MAX_HISTORY_TURNS=6
```

> **High-Availability Dual-Provider Failover:**  
> The system utilizes a dual-provider architecture configured via `PRIMARY_PROVIDER` and `FALLBACK_PROVIDER`, chained seamlessly via `primary.with_fallbacks([fallback])`. If Gemini encounters transient outages or rate limits (HTTP 429), requests automatically execute on Groq without user interruption. If neither key is provided, the application runs deterministically using a local mock evaluator for offline testing.

---

## 3. How to Download & Process the Documents

The knowledge base consists of two official NIST publications:
1. **NIST AI Risk Management Framework (AI RMF 1.0)**: [NIST AI 100-1](https://nvlpubs.nist.gov/nistpubs/ai/nist.ai.100-1.pdf) (48 pages)
2. **NIST AI RMF: Generative AI Profile**: [NIST AI 600-1](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf) (64 pages)

### Step 1: Download Documents
```bash
python src/download_docs.py
```
This downloads and validates both official PDFs into the `data/` folder.

### Step 2: Ingest and Index Documents
Run:
```bash
python ingest.py --force
```

This pipeline executes:
1. **Deterministic Parsing (`src/parse_pdf.py`)**: Uses PyMuPDF (`fitz`) and `pdfplumber` to extract vector text, repair line-wrap hyphenation (`inte- grated` $\rightarrow$ `integrated`), drop running headers and bare page footers, detect heading hierarchies (#, ##, ###) by font size/boldness, reconstruct multi-page tables (forward-filling merged category cells across page breaks), rasterize figures at 200 DPI (with 90° rotation for Fig. 3), and cache multimodal vision descriptions in `data/parsed/figures.json`. Saves inspectable Markdown files in `data/parsed/*.md`.
2. **Heading-Aware Node Parsing (`src/ingest.py`)**: Uses LlamaIndex `MarkdownNodeParser` and token-based `SentenceSplitter` strictly within the ingestion layer so prose nodes never cross section boundaries.
3. **Specialized Table & Figure Nodes**: Creates exactly 72 AI 100-1 subcategory nodes (one per subcategory), 211 AI 600-1 action-row nodes (one per action row), and 5 figure nodes, prepending navigation breadcrumbs.
4. **Hybrid Indexing (`src/vector_store.py`)**: Embeds all 588 nodes into persistent `./chroma_db` using local `sentence-transformers/all-MiniLM-L6-v2` and creates an in-memory `BM25Retriever` corpus stored at `data/parsed/corpus.pkl`.

---

## 4. How to Run the Application & Tests

### Option A: Interactive Conversational Terminal Chat
Launch the terminal chat UI:
```bash
python main.py
```

#### Interactive Commands:
- Type your question naturally.
- `/reset`: Clears conversational history.
- `/history`: Displays prior conversation turns.
- `/debug`: Toggles real-time LangGraph node inspection.
- `/help`: Displays sample queries.
- `/exit`: Exits the session.

### Option B: Run Comprehensive Acceptance Test Suites
To verify all requirements against an isolated temporary ChromaDB (ensuring `./chroma_db` is never polluted):

```bash
# Acceptance Tests 1-6: Ingestion, subcategories, forbidden strings, figures, action rows, sections
python -m unittest tests/test_ingestion.py

# Acceptance Tests 7-10: End-to-end questions, refusals & input guards, multi-turn follow-ups & topic switches, LLM call counts
python -m unittest tests/test_pipeline.py
```

---

## 5. Example Questions

You can test the system across several distinct querying modes:

### A. NIST AI 100-1 (Core Framework)
- *"What are the four functions of the AI RMF Core?"*
  - **Expected Answer**: GOVERN, MAP, MEASURE, MANAGE.
  - **Expected Citation**: NIST AI 100-1 - Part 2: Core and Profiles, Page 20 (PDF p.25).
- *"What does subcategory MEASURE 2.11 cover?"*
  - **Expected Answer**: Details regarding TEVV and performance measurement tracking.
  - **Expected Citation**: NIST AI 100-1 - MEASURE 2.11, Page 34 (PDF p.39).
- *"What is the governing text of category GOVERN 4?"*
  - **Expected Answer**: "Organizational teams are committed to a culture that considers and communicates AI risk."
  - **Expected Citation**: NIST AI 100-1 - GOVERN 4, Page 28 (PDF p.33).

### B. NIST AI 600-1 (Generative AI Profile)
- *"What is action MG-3.2-001?"*
  - **Expected Answer**: Details on applying explainable AI (XAI) techniques (e.g. embeddings analysis, gradient-based attributions) to mitigate unexplainable GAI risks.
  - **Expected Citation**: NIST AI RMF Generative AI Profile - MG-3.2-001, Page 43 (PDF p.47).
- *"What suggested actions address CBRN risks in the Generative AI Profile?"*
  - **Expected Answer**: Cites specific GV, MP, MS, and MG actions (e.g. GV-3.2-001).
- *"What is confabulation according to NIST?"*
  - **Expected Answer**: NIST's formal definition of generative AI hallucinations and fabrications.
  - **Expected Citation**: NIST AI RMF Generative AI Profile - Section 2.2, Page 6 (PDF p.10).

### C. Visual Diagrams & Figures
- *"What are the characteristics of trustworthy AI systems in Figure 4?"*
  - **Expected Answer**: Valid & Reliable (base), Safe, Secure & Resilient, Explainable & Interpretable, Privacy-Enhanced, Fair - With Harmful Bias Managed, and Accountable & Transparent (vertical).
  - **Expected Citation**: NIST AI 100-1 - Fig. 4, Page 12 (PDF p.17).
- *"Which function is cross-cutting according to Figure 5?"*
  - **Expected Answer**: GOVERN is designed to be cross-cutting to inform and be infused throughout Map, Measure, and Manage.
  - **Expected Citation**: NIST AI 100-1 - Fig. 5, Page 20 (PDF p.25).

### D. Multi-Turn Conversational Chains
- **Turn 1**: *"What are the four functions of the AI RMF?"*
- **Turn 2**: *"Which one is cross-cutting?"* (Contextualizer resolves pronoun to GOVERN).
- **Turn 3**: *"What are its subcategories?"* (Contextualizer resolves pronoun to GOVERN subcategories).
- **Turn 4 (Topic Switch)**: *"Now, completely different: what is confabulation?"* (Contextualizer handles topic switch without context bleed).

### E. Out-of-Scope & Adversarial Refusals
- *"How do I bake a chocolate cake?"* $\rightarrow$ Clean refusal with `Sources: None` (2 LLM calls).
- *"What is GPT-5 pricing?"* $\rightarrow$ Clean refusal with `Sources: None`.
- *"?"* or *"hi"* $\rightarrow$ Zero-call input guard intercept (`"Please ask a question about the NIST AI RMF documents."`, 0 LLM calls).

---

## 6. Important Design Decisions

1. **How Documents are Split and Why**:
   - Custom PyMuPDF deterministic font and regex parsing extracts heading hierarchies (#, ##, ###) without carrying forward stale page sections.
   - `pdfplumber` reconstructs multi-page tables (pages 27–34), forward-filling category headers across page boundaries so every subcategory chunk is self-contained.
   - LlamaIndex `MarkdownNodeParser` + `SentenceSplitter(chunk_size=160 tokens, chunk_overlap=30 tokens)` is used exclusively inside `src/ingest.py` so that chunks never span heading boundaries.
   - Exactly 588 atomic nodes: 72 core subcategory nodes, 211 action row nodes, 5 multimodal figure nodes, and 300 section-grounded prose nodes.
2. **Embedding Model & Vector Store Choice**:
   - Dense embeddings use local HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional, zero API cost, offline capability via `HF_HUB_OFFLINE=1`).
   - Vector store is persistent `langchain_chroma.Chroma` in `./chroma_db`, decoupled from unit tests which run against isolated temporary instances.
3. **Hybrid Retrieval & Exact-ID Shortcut**:
   - Dense Chroma similarity search is fused with `rank_bm25.BM25Retriever` via `EnsembleRetriever(weights=[0.5, 0.5])` with $k=6$.
   - Exact-ID shortcut regex matches action IDs (`r"\b[A-Z]{2}-\d+\.\d+-\d{3}\b"`), subcategory IDs (`r"\b(GOVERN|MAP|MEASURE|MANAGE)\s+\d+\.\d+\b"`), category IDs, and figures, placing exact-match metadata nodes directly at Index 0.
4. **LangGraph State Machine Architecture**:
   - Preserves the 6-node LangGraph Corrective RAG (CRAG) design: `contextualize_query`, `retrieve`, `grade_documents`, `rewrite_query`, `generate_answer`, `handle_insufficient_context`.
   - Single-call batched document grading: 1 LLM call evaluates all candidate chunks returning JSON `{"relevant_indices": [...]}`, reducing token overhead and latency by 83%.
5. **Handling Insufficient or Irrelevant Context**:
   - Zero-call input guard: queries with fewer than 3 words and non-follow-up (e.g. `?`, `hi`) route directly to `handle_insufficient_context` returning `"Please ask a question about the NIST AI RMF documents."` with `Sources: None` without retrieval or LLM calls.
   - Out-of-scope or ungrounded queries: Generator instructs LLM to begin with `NO_INFO:` when documents do not support the answer, routing to a standardized refusal with `Sources: None`.
   - Selective citation: LLM emits `Used Chunks: [index1, index2]`; citations are built programmatically only for chunks actually used in the answer.
6. **Conversational Memory Management**:
   - The query contextualizer is passed only the last 6 turns (`MAX_HISTORY_TURNS=6`).
   - Prior `Sources:` citation blocks are stripped from assistant turns in memory to prevent citation leakage across unrelated topics.

---

## 7. Known Limitations

1. **Model-Generated Vision Captions**: Figure descriptions were generated by a multimodal vision LLM from 200 DPI rasterized pages and cached in `data/parsed/figures.json`. While highly detailed, they represent model interpretations of diagram graphics.
2. **Ensemble Retrieval vs. Cross-Encoder Reranker**: The hybrid retriever combines dense vector similarity and BM25 keyword search using equal reciprocal rank fusion weights (`[0.5, 0.5]`). It does not incorporate a heavy cross-encoder reranker (e.g., Cohere Rerank or BGE Reranker) to maintain sub-second retrieval latency.
3. **Bounded Query Reformulation**: The CRAG retry loop allows at most one query rewrite attempt (`MAX_RETRIES=1`) before routing to insufficient context to avoid infinite latency or cyclic hallucination.
4. **No Automated RAGAS Evaluation**: Evaluation is validated through exhaustive programmatic unit and integration test suites (`test_ingestion.py` and `test_pipeline.py`) rather than a continuous synthetic RAGAS evaluation harness.
