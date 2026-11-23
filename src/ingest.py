"""
Heading-Aware Document Ingestion Pipeline for NIST AI RMF Documents.

Uses:
1. PyMuPDF (fitz) and pdfplumber for high-fidelity typographic extraction, table parsing,
   rotation handling, and layout preservation (src/parse_pdf.py).
2. LlamaIndex MarkdownNodeParser and SentenceSplitter strictly within this ingestion pipeline
   for heading-aware node parsing.
3. LangChain Documents with complete metadata schema (document_name, doc_id, section_path,
   section_title, section_number, pdf_page, printed_page, page_number, node_type,
   has_table, has_figure, chunk_id, plus type-specific fields).
4. ChromaDB vector indexing and BM25 corpus persistence for hybrid retrieval.
"""

import os
import sys
import re
import json
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# LlamaIndex is imported ONLY here inside the ingestion pipeline for heading-aware node parsing
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.schema import Document as LlamaDocument, TextNode
from langchain_core.documents import Document

from src.config import (
    DATA_DIR,
    PARSED_DATA_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)
from src.download_docs import download_all_documents
from src.parse_pdf import (
    parse_nist_ai_100_to_markdown,
    parse_nist_ai_600_to_markdown,
)
from src.vector_store import store_documents, get_vector_store

logger = logging.getLogger("rag.ingest")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[-] [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def build_all_documents() -> List[Document]:
    """
    Construct all LangChain Document chunks from parsed Markdown documents,
    figures cache, and structured tables.

    Produces:
    - 72 AI 100-1 Core subcategory nodes (one per subcategory)
    - 211 AI 600-1 Action row nodes (one per action row)
    - 5 Figure nodes with detailed vision descriptions
    - Prose nodes partitioned strictly along heading boundaries
    """
    md_1_path = PARSED_DATA_DIR / "nist_ai_rmf_1.0.md"
    md_g_path = PARSED_DATA_DIR / "nist_ai_rmf_genai.md"
    fig_path = PARSED_DATA_DIR / "figures.json"

    if not (md_1_path.exists() and md_g_path.exists() and fig_path.exists()):
        logger.info("Parsed Markdown or figure cache missing. Running parse_pdf...")
        parse_nist_ai_100_to_markdown()
        parse_nist_ai_600_to_markdown()

    with open(fig_path, "r", encoding="utf-8") as f:
        figures_data = json.load(f)
    figs_100_1 = figures_data.get("ai_rmf_1.0", {})

    all_docs: List[Document] = []

    # -------------------------------------------------------------
    # 1. PARSE AI 100-1 (NIST AI RMF 1.0)
    # -------------------------------------------------------------
    doc1_name = "NIST AI RMF 1.0"
    doc1_id = "ai_rmf_1.0"
    doc1_text = md_1_path.read_text(encoding="utf-8")

    # Specialized Rule: Figure Nodes (5 Figures)
    fig_pages = {
        "fig_1": (11, 6),
        "fig_2": (15, 10),
        "fig_3": (16, 11),
        "fig_4": (17, 12),
        "fig_5": (25, 20),
    }
    for fkey, finfo in figs_100_1.items():
        fnum = finfo.get("figure_num", fkey.replace("fig_", ""))
        caption = finfo.get("caption", f"Figure {fnum}")
        description = finfo.get("description", "")
        source = finfo.get("description_source", "vision_llm")
        p_page, pr_page = fig_pages.get(fkey, (1, None))

        breadcrumb = f"{doc1_name} > Figures > Figure {fnum}:"
        page_content = f"{breadcrumb}\n\n{caption}\n\n{description}"
        meta = {
            "document_name": doc1_name,
            "doc_id": doc1_id,
            "section_path": f"{doc1_name} > Figures > Figure {fnum}",
            "section_title": f"Figure {fnum}",
            "section_number": str(fnum),
            "pdf_page": p_page,
            "printed_page": pr_page,
            "page_number": pr_page if pr_page is not None else p_page,
            "node_type": "figure",
            "has_table": False,
            "has_figure": True,
            "figure_id": fkey,
            "caption": caption,
            "description_source": source,
            "chunk_id": f"ai100-1:figure:{fkey}",
        }
        all_docs.append(Document(page_content=page_content, metadata=meta))

    # Specialized Rule: AI 100-1 Core Table Nodes (ONE node per subcategory, exactly 72)
    subcat_rows = re.findall(
        r"\|\s*((?:GOVERN|MAP|MEASURE|MANAGE)\s+\d+\.\d+)\s*\|\s*([^|]+)\|\s*((?:GOVERN|MAP|MEASURE|MANAGE)\s+\d+)\s*\|\s*([^|]+)\|",
        doc1_text,
    )

    def get_subcat_page(sub_id: str) -> Tuple[int, Optional[int]]:
        fn, num = sub_id.split()
        major = int(num.split('.')[0])
        if fn == "GOVERN":
            return (28, 23) if major >= 4 else (27, 22)
        elif fn == "MAP":
            return (32, 27) if major >= 4 else ((31, 26) if major >= 2 else (30, 25))
        elif fn == "MEASURE":
            return (36, 31) if major >= 3 else ((35, 30) if major >= 2 else (34, 29))
        elif fn == "MANAGE":
            return (38, 33) if major >= 3 else (37, 32)
        return (27, 22)

    fn_sec_map = {
        "GOVERN": ("5.1 Govern", "5.1"),
        "MAP": ("5.2 Map", "5.2"),
        "MEASURE": ("5.3 Measure", "5.3"),
        "MANAGE": ("5.4 Manage", "5.4"),
    }

    seen_subcats = set()
    for sub_id, sub_text, cat_id, cat_text in subcat_rows:
        sub_id = sub_id.strip()
        sub_text = sub_text.strip()
        cat_id = cat_id.strip()
        cat_text = cat_text.strip()
        if sub_id in seen_subcats:
            continue
        seen_subcats.add(sub_id)

        fn = sub_id.split()[0]
        sec_title, sec_num = fn_sec_map[fn]
        pdf_p, pr_p = get_subcat_page(sub_id)
        breadcrumb = f"{doc1_name} > Part 2 > {sec_title} > {cat_id} > {sub_id}:"
        text = f"Category: {cat_text}\nSubcategory {sub_id}: {sub_text}"
        page_content = f"{breadcrumb}\n\n{text}"
        chunk_id = f"ai100-1:s{sec_num}:{sub_id.lower().replace(' ', '-')}"
        meta = {
            "document_name": doc1_name,
            "doc_id": doc1_id,
            "section_path": f"Part 2: Core and Profiles > {sec_title} > {cat_id} > {sub_id}",
            "section_title": sec_title,
            "section_number": sec_num,
            "pdf_page": pdf_p,
            "printed_page": pr_p,
            "page_number": pr_p if pr_p is not None else pdf_p,
            "node_type": "table_row",
            "has_table": True,
            "has_figure": False,
            "function": fn,
            "category_id": cat_id,
            "category_text": cat_text,
            "subcategory_id": sub_id,
            "subcategory_text": sub_text,
            "chunk_id": chunk_id,
        }
        all_docs.append(Document(page_content=page_content, metadata=meta))

    # Prose Node Creation for AI 100-1 using MarkdownNodeParser and SentenceSplitter
    # 160 tokens ~ 700-800 characters, ensuring page_content with breadcrumb remains within CHUNK_SIZE limit
    token_chunk_size = 160
    token_chunk_overlap = 30
    sentence_splitter = SentenceSplitter(chunk_size=token_chunk_size, chunk_overlap=token_chunk_overlap)

    page_splits = re.split(r"<!-- PAGE pdf_page=(\d+) printed_page=(\S+) -->", doc1_text)
    current_sec_path = "Front matter"
    current_sec_title = "Front matter"
    current_sec_num = ""

    for i in range(1, len(page_splits), 3):
        pdf_p = int(page_splits[i])
        pr_str = page_splits[i+1]
        pr_p = None if pr_str == "None" else (int(pr_str) if pr_str.isdigit() else pr_str)
        p_text = page_splits[i+2]

        if pdf_p <= 5:
            current_sec_path = "Front matter"
            current_sec_title = "Front matter"
            current_sec_num = ""
        elif pdf_p == 48:
            current_sec_path = "Back matter"
            current_sec_title = "Back matter"
            current_sec_num = ""

        # Remove table and figure blocks from page prose using explicit boundary comments
        clean_p_text = re.sub(r"<!-- (?:FIGURE|TABLE)_START.*?<!-- (?:FIGURE|TABLE)_END -->", "", p_text, flags=re.S).strip()
        if not clean_p_text:
            continue

        # Use MarkdownNodeParser to partition page into heading-aware nodes
        md_parser = MarkdownNodeParser()
        llama_doc = LlamaDocument(text=clean_p_text)
        page_nodes = md_parser.get_nodes_from_documents([llama_doc])

        chunk_counter = 0
        for p_node in page_nodes:
            text = p_node.text.strip()
            if not text:
                continue

            # Update section tracking if heading is present
            h_match = re.search(r"^(#+)\s+(.+)$", text, re.M)
            if h_match and pdf_p > 5 and pdf_p < 48:
                level = len(h_match.group(1))
                h_text = h_match.group(2).strip()
                if level == 1:
                    current_sec_path = h_text
                    current_sec_title = h_text
                    current_sec_num = ""
                elif level >= 2:
                    current_sec_path = f"{current_sec_path.split(' > ')[0]} > {h_text}"
                    current_sec_title = h_text
                    num_match = re.match(r"^([1-6](?:\.[0-9]+)*)", h_text)
                    if num_match:
                        current_sec_num = num_match.group(1)

            # Split only inside this section if text is long
            sub_chunks = sentence_splitter.split_text(text) if len(text) > CHUNK_SIZE else [text]

            for sc in sub_chunks:
                sc = sc.strip()
                if len(sc) < 40:
                    continue
                chunk_counter += 1
                breadcrumb = f"{doc1_name} > {current_sec_path}:"
                page_content = f"{breadcrumb}\n\n{sc}"
                chunk_id = f"ai100-1:p{pdf_p}:c{chunk_counter}"
                meta = {
                    "document_name": doc1_name,
                    "doc_id": doc1_id,
                    "section_path": current_sec_path,
                    "section_title": current_sec_title,
                    "section_number": current_sec_num,
                    "pdf_page": pdf_p,
                    "printed_page": pr_p,
                    "page_number": pr_p if pr_p is not None else pdf_p,
                    "node_type": "prose",
                    "has_table": False,
                    "has_figure": False,
                    "chunk_id": chunk_id,
                }
                all_docs.append(Document(page_content=page_content, metadata=meta))

    # -------------------------------------------------------------
    # 2. PARSE AI 600-1 (Generative AI Profile)
    # -------------------------------------------------------------
    docg_name = "NIST AI RMF Generative AI Profile"
    docg_id = "ai_rmf_genai"
    docg_text = md_g_path.read_text(encoding="utf-8")

    # Load action page mappings if cached
    actions_json_path = PARSED_DATA_DIR / "genai_actions.json"
    actions_page_map = {}
    if actions_json_path.exists():
        try:
            with open(actions_json_path, "r", encoding="utf-8") as f:
                for act in json.load(f):
                    actions_page_map[act["action_id"]] = (act.get("pdf_page", 17), act.get("printed_page", 13))
        except Exception as e:
            logger.warning(f"Could not load genai_actions.json: {e}")

    # Specialized Rule: AI 600-1 Action Table Nodes (ONE node per action row, exactly 211)
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

        fn = fn.strip()
        subcat = subcat.strip()
        sugg = sugg.strip()
        risks = risks.strip()
        tasks = tasks.strip()

        if len(row) >= 8 and row[6] and row[7]:
            p_page = int(row[6].strip())
            pr_page = int(row[7].strip())
        elif aid in actions_page_map:
            p_page, pr_page = actions_page_map[aid]
        else:
            p_page, pr_page = 17, 13

        breadcrumb = f"{docg_name} > 3. Suggested Actions > {fn} > {aid}:"
        text = f"Action ID: {aid}\nFunction: {fn}\nSubcategory: {subcat}\nSuggested Action: {sugg}\nGAI Risks: {risks}\nAI Actor Tasks: {tasks}"
        page_content = f"{breadcrumb}\n\n{text}"
        chunk_id = f"ai600-1:action:{aid.lower()}"
        meta = {
            "document_name": docg_name,
            "doc_id": docg_id,
            "section_path": f"3. Generative AI Profile Suggested Actions > {fn} > {aid}",
            "section_title": "Generative AI Profile Suggested Actions",
            "section_number": "3",
            "pdf_page": p_page,
            "printed_page": pr_page,
            "page_number": pr_page,
            "node_type": "table_row",
            "has_table": True,
            "has_figure": False,
            "action_id": aid,
            "function": fn,
            "subcategory_id": subcat.split(":")[0].strip() if ":" in subcat else subcat,
            "suggested_action": sugg,
            "gai_risks": risks,
            "ai_actor_tasks": tasks,
            "chunk_id": chunk_id,
        }
        all_docs.append(Document(page_content=page_content, metadata=meta))

    # Prose Node Creation for AI 600-1
    g_page_splits = re.split(r"<!-- PAGE pdf_page=(\d+) printed_page=(\S+) -->", docg_text)
    current_sec_path = "Front matter"
    current_sec_title = "Front matter"
    current_sec_num = ""

    for i in range(1, len(g_page_splits), 3):
        pdf_p = int(g_page_splits[i])
        pr_str = g_page_splits[i+1]
        pr_p = None if pr_str == "None" else (int(pr_str) if pr_str.isdigit() else pr_str)
        p_text = g_page_splits[i+2]

        if pdf_p <= 4:
            current_sec_path = "Front matter"
            current_sec_title = "Front matter"
            current_sec_num = ""

        # Remove action table markdown from page prose
        clean_p_text = re.sub(r"<!-- (?:FIGURE|TABLE)_START.*?<!-- (?:FIGURE|TABLE)_END -->", "", p_text, flags=re.S).strip()
        if not clean_p_text:
            continue

        md_parser = MarkdownNodeParser()
        llama_doc = LlamaDocument(text=clean_p_text)
        page_nodes = md_parser.get_nodes_from_documents([llama_doc])

        chunk_counter = 0
        for p_node in page_nodes:
            text = p_node.text.strip()
            if not text:
                continue

            h_match = re.search(r"^(#+)\s+(.+)$", text, re.M)
            if h_match and pdf_p > 4:
                level = len(h_match.group(1))
                h_text = h_match.group(2).strip()
                if level == 1:
                    current_sec_path = h_text
                    current_sec_title = h_text
                    current_sec_num = h_text.split('.')[0] if '.' in h_text else ""
                elif level >= 2:
                    current_sec_path = f"{current_sec_path.split(' > ')[0]} > {h_text}"
                    current_sec_title = h_text
                    num_match = re.match(r"^(\d+(?:\.[0-9]+)*)", h_text)
                    if num_match:
                        current_sec_num = num_match.group(1)

            sub_chunks = sentence_splitter.split_text(text) if len(text) > CHUNK_SIZE else [text]
            for sc in sub_chunks:
                sc = sc.strip()
                if len(sc) < 40:
                    continue
                chunk_counter += 1
                breadcrumb = f"{docg_name} > {current_sec_path}:"
                page_content = f"{breadcrumb}\n\n{sc}"
                chunk_id = f"ai600-1:p{pdf_p}:c{chunk_counter}"
                meta = {
                    "document_name": docg_name,
                    "doc_id": docg_id,
                    "section_path": current_sec_path,
                    "section_title": current_sec_title,
                    "section_number": current_sec_num,
                    "pdf_page": pdf_p,
                    "printed_page": pr_p,
                    "page_number": pr_p if pr_p is not None else pdf_p,
                    "node_type": "prose",
                    "has_table": False,
                    "has_figure": False,
                    "chunk_id": chunk_id,
                }
                all_docs.append(Document(page_content=page_content, metadata=meta))

    return all_docs


def ingest_documents(force_reindex: bool = False) -> int:
    """
    Main ingestion execution:
    1. Downloads NIST PDFs if missing.
    2. Parses PDFs with PyMuPDF/pdfplumber to clean inspectable Markdown and caches figures.
    3. Builds specialized heading-aware nodes with LlamaIndex MarkdownNodeParser.
    4. Indexes into ChromaDB vector store and persists BM25 corpus.
    """
    vector_store = get_vector_store()
    existing_count = vector_store._collection.count()

    if existing_count > 0 and not force_reindex:
        logger.info(f"ChromaDB already contains {existing_count} chunks. Skipping re-indexing.")
        return existing_count

    if force_reindex and existing_count > 0:
        logger.info("Clearing existing ChromaDB collection for fresh re-indexing...")
        try:
            vector_store.delete_collection()
        except Exception as e:
            logger.warning(f"Failed to delete Chroma collection: {e}")
        vector_store = get_vector_store()

    logger.info("Checking NIST documents in data directory...")
    download_all_documents()

    logger.info("Parsing documents and constructing specialized LangChain Documents...")
    all_chunks = build_all_documents()

    # Observability breakdown
    node_types: Dict[str, int] = {}
    doc_counts: Dict[str, int] = {}
    for d in all_chunks:
        nt = d.metadata.get("node_type", "unknown")
        dn = d.metadata.get("document_name", "unknown")
        node_types[nt] = node_types.get(nt, 0) + 1
        doc_counts[dn] = doc_counts.get(dn, 0) + 1

    subcats = [d for d in all_chunks if d.metadata.get("subcategory_id") and d.metadata.get("doc_id") == "ai_rmf_1.0"]
    actions = [d for d in all_chunks if d.metadata.get("action_id")]
    figures = [d for d in all_chunks if d.metadata.get("node_type") == "figure"]

    action_fn_counts: Dict[str, int] = {}
    for a in actions:
        fn = a.metadata.get("function", "UNKNOWN")
        action_fn_counts[fn] = action_fn_counts.get(fn, 0) + 1

    logger.info("=== INGESTION OBSERVABILITY REPORT ===")
    logger.info(f"Total searchable chunks generated: {len(all_chunks)}")
    logger.info(f"Chunks by Document: {doc_counts}")
    logger.info(f"Chunks by Node Type: {node_types}")
    logger.info(f"AI 100-1 Subcategories parsed: {len(subcats)}")
    logger.info(f"AI 600-1 Actions parsed: {len(actions)} (Breakdown: {action_fn_counts})")
    logger.info(f"Figures extracted & described: {len(figures)}")

    # Index into ChromaDB
    total_stored = store_documents(all_chunks)

    # Persist Document corpus for BM25Retriever
    corpus_file = PARSED_DATA_DIR / "corpus.pkl"
    with open(corpus_file, "wb") as f:
        pickle.dump(all_chunks, f)
    logger.info(f"Saved BM25 corpus to {corpus_file}")

    return total_stored


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingest NIST AI RMF documents into ChromaDB.")
    parser.add_argument("--force", action="store_true", help="Force re-indexing even if database exists")
    args = parser.parse_args()

    total = ingest_documents(force_reindex=args.force)
    logger.info(f"Ingestion completed. Total chunks in ChromaDB: {total}")
