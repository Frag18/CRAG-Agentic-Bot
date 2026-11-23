"""
Deterministic, High-Fidelity PDF Parsing Engine for NIST AI RMF Publications.
Uses PyMuPDF (fitz) for typographic layout, header/footer removal, heading detection,
and figure rendering, and pdfplumber for structured tabular parsing.

Outputs:
1. data/parsed/nist_ai_rmf_1.0.md
2. data/parsed/nist_ai_rmf_genai.md
3. data/parsed/figures.json
"""

import os
import sys
import re
import json
import base64
import logging
import unicodedata
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import pymupdf as fitz
import pdfplumber

from src.config import (
    DATA_DIR,
    PARSED_DATA_DIR,
    GEMINI_API_KEY,
    GEMINI_MODEL,
)

logger = logging.getLogger("rag.parser")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[-] [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def clean_text_typography(raw_text: str) -> str:
    """Normalize unicode, clean typography artifacts, and fix hyphenated line breaks."""
    if not raw_text:
        return ""
    normalized = unicodedata.normalize("NFKC", raw_text)
    normalized = normalized.replace("\u2010", "-").replace("\u2013", "-").replace("\u2014", "-")
    normalized = normalized.replace("\u2018", "'").replace("\u2019", "'")
    normalized = normalized.replace("\u201c", '"').replace("\u201d", '"')
    normalized = normalized.replace("\ufb01", "fi").replace("\ufb02", "fl")
    # Fix hyphenated words broken across lines: e.g. "inte-\ngrated" -> "integrated", "GOV-\n\nERN" -> "GOVERN"
    cleaned = re.sub(r"(\b[A-Za-z]+)-\s*\n+\s*([a-zA-Z]+\b)", r"\1\2", normalized)
    return cleaned


def describe_figure_with_llm(image_bytes: bytes, caption: str) -> Optional[str]:
    """Call vision-capable LLM to describe figure exhaustively."""
    if not GEMINI_API_KEY or GEMINI_API_KEY.startswith("your_"):
        return None
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import HumanMessage

        b64 = base64.b64encode(image_bytes).decode("utf-8")
        llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GEMINI_API_KEY,
            temperature=0.0
        )
        prompt = (
            f"Caption: {caption}\n\n"
            "Describe this figure exhaustively for retrieval: list every label, box, category, "
            "arrow/relationship and the caption. Do not add information not visible."
        )
        msg = HumanMessage(content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": f"data:image/png;base64,{b64}"}
        ])
        res = llm.invoke([msg])
        if hasattr(res, "content"):
            content = res.content
            if isinstance(content, list):
                parts = [p.get("text", "") if isinstance(p, dict) else str(p) for p in content]
                return "".join(parts).strip()
            return str(content).strip()
    except Exception as e:
        logger.warning(f"Vision LLM description failed for '{caption}': {e}")
    return None


def extract_and_describe_figures_ai100(doc: fitz.Document) -> Dict[str, Any]:
    """
    Extract, render at 200 DPI, and describe Figures 1-5 in AI 100-1.
    Fallback chain: figures.json cache -> Vision LLM -> caption fallback.
    """
    PARSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    figures_cache_file = PARSED_DATA_DIR / "figures.json"

    cached_figures = {}
    if figures_cache_file.exists():
        try:
            with open(figures_cache_file, "r", encoding="utf-8") as f:
                cached_figures = json.load(f)
        except Exception:
            cached_figures = {}

    figures_spec = [
        {"fig_id": "Fig. 1", "key": "fig_1", "pdf_page": 10, "printed_page": 5, "rotate": 0, "caption": "Fig. 1. Examples of potential harms related to AI systems."},
        {"fig_id": "Fig. 2", "key": "fig_2", "pdf_page": 15, "printed_page": 10, "rotate": 0, "caption": "Fig. 2. Lifecycle and Key Dimensions of an AI System."},
        {"fig_id": "Fig. 3", "key": "fig_3", "pdf_page": 16, "printed_page": 11, "rotate": 90, "caption": "Fig. 3. AI actors across AI lifecycle stages."},
        {"fig_id": "Fig. 4", "key": "fig_4", "pdf_page": 17, "printed_page": 12, "rotate": 0, "caption": "Fig. 4. Characteristics of trustworthy AI systems."},
        {"fig_id": "Fig. 5", "key": "fig_5", "pdf_page": 25, "printed_page": 20, "rotate": 0, "caption": "Fig. 5. Functions organize AI risk management activities at their highest level."},
    ]

    doc_fig_results = {}
    doc_cache = cached_figures.get("ai_rmf_1.0", {})

    for spec in figures_spec:
        fkey = spec["key"]
        if fkey in doc_cache and doc_cache[fkey].get("description"):
            doc_fig_results[fkey] = doc_cache[fkey]
            continue

        p_idx = spec["pdf_page"] - 1
        page = doc[p_idx]
        mat = fitz.Matrix(200 / 72, 200 / 72)
        if spec["rotate"] != 0:
            mat = mat.prerotate(spec["rotate"])

        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")

        # 1. Vision LLM
        desc = describe_figure_with_llm(img_bytes, spec["caption"])
        source = "vision_llm"

        # 2. Caption fallback
        if not desc:
            desc = spec["caption"]
            source = "caption_only"
            logger.warning(f"Could not generate visual description for {spec['fig_id']}. Using caption fallback.")

        doc_fig_results[fkey] = {
            "figure_id": spec["fig_id"],
            "caption": spec["caption"],
            "description": desc,
            "description_source": source,
            "pdf_page": spec["pdf_page"],
            "printed_page": spec["printed_page"]
        }

    cached_figures["ai_rmf_1.0"] = doc_fig_results
    cached_figures["ai_rmf_genai"] = {}
    with open(figures_cache_file, "w", encoding="utf-8") as f:
        json.dump(cached_figures, f, indent=2, ensure_ascii=False)

    return doc_fig_results


def parse_core_tables_ai100(doc: fitz.Document) -> List[Dict[str, Any]]:
    """
    Parse Tables 1-4 from AI 100-1 as logical tables with forward-filled category text.
    Handles page breaks, repeated headers, 'Continued on next page', and split text.
    Expected: GOVERN=19, MAP=18, MEASURE=22, MANAGE=13 (Total 72).
    """
    tables_spec = [
        {"function": "GOVERN", "pages": [27, 28, 29], "table_id": 1, "expected": 19},
        {"function": "MAP", "pages": [31, 32, 33], "table_id": 2, "expected": 18},
        {"function": "MEASURE", "pages": [34, 35, 36], "table_id": 3, "expected": 22},
        {"function": "MANAGE", "pages": [37, 38], "table_id": 4, "expected": 13},
    ]

    all_subcategories = []

    for t_info in tables_spec:
        fn = t_info["function"]
        pages = t_info["pages"]
        expected = t_info["expected"]

        active_cat_id = None
        active_cat_text = None
        subcats_for_table = []
        full_category_map = {}

        for p_num in pages:
            page = doc[p_num - 1]
            blocks = page.get_text("blocks")
            content_blocks = []
            for b in blocks:
                text = b[4].strip()
                if not text or b[1] < 55 or b[3] > 700:
                    continue
                if "Continued on next page" in text:
                    continue
                if re.match(r"^Table\s+\d+:", text):
                    continue
                if re.match(r"^Categories\s*Subcategories", text, re.I):
                    continue
                if re.match(r"^5\.[2-4]\s+[A-Za-z]+", text) or any(
                    phrase in text for phrase in ["The MAP function", "The MEASURE function", "The MANAGE function"]
                ):
                    continue
                content_blocks.append(b)

            left_col = []
            right_col = []
            for b in content_blocks:
                x0, y0, x1, y1, text = b[0], b[1], b[2], b[3], b[4].strip()
                if x1 < 205:
                    left_col.append({"y0": y0, "y1": y1, "text": text})
                elif x0 > 200:
                    right_col.append({"y0": y0, "y1": y1, "text": text})

            left_col.sort(key=lambda x: x["y0"])
            right_col.sort(key=lambda x: x["y0"])

            page_categories = []
            for item in left_col:
                norm_text = ' '.join(clean_text_typography(item["text"]).split())
                cat_match = re.match(r"^(" + fn + r"\s+\d+):?\s*(.*)", norm_text, re.I)
                if cat_match:
                    cid = cat_match.group(1).upper()
                    ctext = cat_match.group(2).strip()
                    page_categories.append({"y0": item["y0"], "cat_id": cid, "cat_text": ctext})
                    full_category_map[cid] = ctext
                else:
                    if page_categories:
                        page_categories[-1]["cat_text"] = (page_categories[-1]["cat_text"] + " " + norm_text).strip()
                        full_category_map[page_categories[-1]["cat_id"]] = page_categories[-1]["cat_text"]
                    elif active_cat_id is not None:
                        active_cat_text = (active_cat_text + " " + norm_text).strip()
                        full_category_map[active_cat_id] = active_cat_text

            for item in right_col:
                norm_text = ' '.join(clean_text_typography(item["text"]).split())
                sub_matches = list(re.finditer(r"(" + fn + r"\s+\d+\.\d+):?\s*", norm_text, re.I))
                if sub_matches:
                    for i, sm in enumerate(sub_matches):
                        sub_id = sm.group(1).upper()
                        start_pos = sm.end()
                        end_pos = sub_matches[i+1].start() if i + 1 < len(sub_matches) else len(norm_text)
                        sub_text = norm_text[start_pos:end_pos].strip()

                        matching_cats = [c for c in page_categories if c["y0"] <= item["y0"] + 15]
                        if matching_cats:
                            chosen_cat = matching_cats[-1]
                            cat_id = chosen_cat["cat_id"]
                            cat_text = chosen_cat["cat_text"]
                            active_cat_id = cat_id
                            active_cat_text = cat_text
                        else:
                            cat_id = active_cat_id
                            cat_text = active_cat_text

                        printed_p = p_num - 5 if p_num >= 6 else None
                        subcats_for_table.append({
                            "function": fn,
                            "category_id": cat_id,
                            "category_text": cat_text,
                            "subcategory_id": sub_id,
                            "subcategory_text": sub_text,
                            "pdf_page": p_num,
                            "printed_page": printed_p,
                            "table_id": t_info["table_id"],
                        })

            if page_categories:
                active_cat_id = page_categories[-1]["cat_id"]
                active_cat_text = page_categories[-1]["cat_text"]

        # Second pass: ensure every subcategory carries the complete category text
        for sc in subcats_for_table:
            cid = sc["category_id"]
            if cid in full_category_map:
                sc["category_text"] = full_category_map[cid]

        logger.info(f"Parsed {fn} Table {t_info['table_id']}: {len(subcats_for_table)} / {expected} subcategories.")
        all_subcategories.extend(subcats_for_table)

    return all_subcategories


def parse_action_tables_ai600(pdf_path: Path) -> List[Dict[str, Any]]:
    """
    Parse all 211 action rows from the action tables of NIST AI 600-1 (pages 17-50).
    Captures: action_id, suggested_action, gai_risks, ai_actor_tasks, function, subcategory.
    """
    fn_map = {"GV": "GOVERN", "MP": "MAP", "MS": "MEASURE", "MG": "MANAGE"}
    actions = []

    with pdfplumber.open(str(pdf_path)) as pdf:
        current_subcat_title = "GOVERN 1.1"
        current_actor_tasks = "Governance and Oversight"

        for p_idx in range(16, 50): # pages 17 to 50
            page = pdf.pages[p_idx]
            p_num = p_idx + 1
            printed_page = p_num - 4

            tables = page.extract_tables() or []
            for table in tables:
                for row in table:
                    cells = [' '.join(clean_text_typography(str(c or '')).split()) for c in row if str(c or '').strip()]
                    if not cells:
                        continue

                    row_str = " ".join(cells)

                    subcat_match = re.search(r"((?:GOVERN|MAP|MEASURE|MANAGE)\s+\d+\.\d+[:\s][^|]+)", row_str, re.I)
                    if subcat_match:
                        current_subcat_title = subcat_match.group(1).strip()

                    if "AI Actor Tasks:" in row_str:
                        tasks_part = row_str.split("AI Actor Tasks:")[-1].strip()
                        if tasks_part:
                            current_actor_tasks = tasks_part

                    aid_match = re.search(r"\b([A-Z]{2}-\d+\.\d+-\d{3})\b", row_str)
                    if aid_match:
                        aid = aid_match.group(1)
                        pref = aid[:2]
                        fn = fn_map.get(pref, "UNKNOWN")

                        aid_cell_idx = -1
                        for idx, c in enumerate(cells):
                            if aid in c:
                                aid_cell_idx = idx
                                break

                        action_text = ""
                        gai_risks = ""

                        remaining_cells = [
                            c for idx, c in enumerate(cells)
                            if idx != aid_cell_idx and c not in ('Action ID', 'Suggested Action', 'GAI Risks')
                        ]

                        if len(remaining_cells) >= 2:
                            action_text = remaining_cells[0]
                            gai_risks = remaining_cells[1]
                        elif len(remaining_cells) == 1:
                            action_text = remaining_cells[0]

                        actions.append({
                            "action_id": aid,
                            "function": fn,
                            "subcategory": current_subcat_title,
                            "suggested_action": action_text,
                            "gai_risks": gai_risks,
                            "ai_actor_tasks": current_actor_tasks,
                            "pdf_page": p_num,
                            "printed_page": printed_page,
                        })

    logger.info(f"Parsed NIST AI 600-1 Action Tables: {len(actions)} actions across GV, MP, MS, MG.")
    return actions


def parse_nist_ai_100_to_markdown() -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    """
    Parse NIST AI 100-1 (AI RMF 1.0) into clean Markdown with heading levels.
    Emits data/parsed/nist_ai_rmf_1.0.md, core subcategories, and figure blocks.
    """
    pdf_path = DATA_DIR / "nist_ai_rmf_1.0.pdf"
    doc = fitz.open(str(pdf_path))

    figures = extract_and_describe_figures_ai100(doc)
    subcategories = parse_core_tables_ai100(doc)

    md_lines = ["# NIST AI Risk Management Framework (AI RMF 1.0)\n"]
    current_part = "Front matter"
    current_sec = "Front matter"

    # Core table pages to skip raw prose for Table 1-4 (we insert clean markdown tables)
    table_pages = {27, 28, 29, 31, 32, 33, 34, 35, 36, 37, 38}

    # Group subcategories by function
    subcats_by_fn = {}
    for sc in subcategories:
        subcats_by_fn.setdefault(sc["function"], []).append(sc)

    inserted_tables = set()

    for p_idx in range(len(doc)):
        p_num = p_idx + 1
        page = doc[p_idx]

        # Determine printed page
        if p_num <= 4:
            printed_page = None
        elif p_num == 5:
            printed_page = "v"
        elif p_num == 48:
            printed_page = None
        else:
            printed_page = p_num - 5

        page_header_str = f"\n<!-- PAGE pdf_page={p_num} printed_page={printed_page or 'None'} -->\n"
        md_lines.append(page_header_str)

        if p_num == 48:
            md_lines.append("# Back matter\n\nThis publication is available free of charge from: https://doi.org/10.6028/NIST.AI.100-1\n")
            continue

        # Extract blocks, dropping running header (y < 55) and footer (y > 700)
        blocks = page.get_text("blocks")
        page_prose_lines = []

        # On front matter pages 1-5, keep section as Front matter and skip heading detection
        if p_num <= 5:
            for b in blocks:
                x0, y0, x1, y1, text = b[0], b[1], b[2], b[3], b[4].strip()
                if not text or y0 < 52 or y1 > 705:
                    continue
                page_prose_lines.append(clean_text_typography(text))
            md_lines.append("\n\n".join(page_prose_lines))
            continue

        for b in blocks:
            x0, y0, x1, y1, text = b[0], b[1], b[2], b[3], b[4].strip()
            if not text or y0 < 52 or y1 > 705:
                continue

            cleaned_block = clean_text_typography(text)

            # Skip table-specific content on table pages
            if p_num in table_pages:
                if re.match(r"^Table\s+\d+:", cleaned_block) or "Continued on next page" in cleaned_block:
                    continue
                if re.match(r"^(?:GOVERN|MAP|MEASURE|MANAGE)\s+\d", cleaned_block):
                    continue
                if "Categories Subcategories" in cleaned_block:
                    continue

            # Clean bullet glyphs
            cleaned_block = cleaned_block.replace("\ufffd", "- ")

            # Executive summary on printed page 1
            if re.match(r"^Executive\s+Summary\b", cleaned_block, re.I):
                current_part = "Executive Summary"
                current_sec = "Executive Summary"
                page_prose_lines.append(f"\n# Executive Summary\n")
                continue

            # Heading 1: Parts & Appendices
            part_match = re.match(r"^(Part\s+[12]:\s+[^\n]+|Appendix\s+[A-D]:\s+[^\n]+)", cleaned_block, re.I)
            if part_match:
                h1_text = part_match.group(1).strip()
                current_part = h1_text
                current_sec = h1_text
                page_prose_lines.append(f"\n# {h1_text}\n")
                continue

            # Heading 4: Three-level sections (e.g. "1.2.1 Risk Measurement")
            h4_match = re.match(r"^([1-6]\.[0-9]+\.[0-9]+)\.?\s*\n?\s*([A-Z][^\n]+)", cleaned_block)
            if h4_match and len(h4_match.group(2).strip()) < 90:
                h4_text = f"{h4_match.group(1)} {h4_match.group(2).strip()}"
                current_sec = f"{current_part} > {h4_text}"
                page_prose_lines.append(f"\n#### {h4_text}\n")
                continue

            # Heading 3: Two-level sections (e.g. "1.1 Understanding...", "5.1 Govern")
            h3_match = re.match(r"^([1-6]\.[0-9]+)\.?\s*\n?\s*([A-Z][^\n]+)", cleaned_block)
            if h3_match and len(h3_match.group(2).strip()) < 90:
                h3_text = f"{h3_match.group(1)} {h3_match.group(2).strip()}"
                current_sec = f"{current_part} > {h3_text}"
                page_prose_lines.append(f"\n### {h3_text}\n")
                continue

            # Heading 2: Major single-digit sections (e.g. "1. Framing Risk", "5. AI RMF Core")
            h2_match = re.match(r"^([1-6])\.?\s*\n\s*([A-Z][^\n]+)", cleaned_block)
            if h2_match and len(h2_match.group(2).strip()) < 80:
                h2_text = f"{h2_match.group(1)}. {h2_match.group(2).strip()}"
                current_sec = f"{current_part} > {h2_text}"
                page_prose_lines.append(f"\n## {h2_text}\n")
                continue

            # Check for Figure captions
            fig_match = re.match(r"^(Fig\.?\s*(\d+)[:\.\s][^\n]+)", cleaned_block, re.I)
            if fig_match:
                fnum = fig_match.group(2)
                fkey = f"fig_{fnum}"
                fig_info = figures.get(fkey, {})
                page_prose_lines.append(f"\n<!-- FIGURE_START figure_id={fkey} -->\n### Figure {fnum}: {fig_info.get('caption', cleaned_block)}\n{fig_info.get('description', '')}\n<!-- FIGURE_END -->\n")
                continue

            page_prose_lines.append(cleaned_block)

        # Insert table markdown where appropriate
        if p_num == 27 and "GOVERN" not in inserted_tables:
            inserted_tables.add("GOVERN")
            page_prose_lines.append("\n<!-- TABLE_START table_id=table_1 -->\n### Table 1: Categories and Subcategories for the GOVERN Function\n")
            page_prose_lines.append("| Subcategory ID | Subcategory Text | Category ID | Category Text |\n| --- | --- | --- | --- |")
            for sc in subcats_by_fn.get("GOVERN", []):
                page_prose_lines.append(f"| {sc['subcategory_id']} | {sc['subcategory_text']} | {sc['category_id']} | {sc['category_text']} |")
            page_prose_lines.append("<!-- TABLE_END -->\n")

        elif p_num == 31 and "MAP" not in inserted_tables:
            inserted_tables.add("MAP")
            page_prose_lines.append("\n<!-- TABLE_START table_id=table_2 -->\n### Table 2: Categories and Subcategories for the MAP Function\n")
            page_prose_lines.append("| Subcategory ID | Subcategory Text | Category ID | Category Text |\n| --- | --- | --- | --- |")
            for sc in subcats_by_fn.get("MAP", []):
                page_prose_lines.append(f"| {sc['subcategory_id']} | {sc['subcategory_text']} | {sc['category_id']} | {sc['category_text']} |")
            page_prose_lines.append("<!-- TABLE_END -->\n")

        elif p_num == 34 and "MEASURE" not in inserted_tables:
            inserted_tables.add("MEASURE")
            page_prose_lines.append("\n<!-- TABLE_START table_id=table_3 -->\n### Table 3: Categories and Subcategories for the MEASURE Function\n")
            page_prose_lines.append("| Subcategory ID | Subcategory Text | Category ID | Category Text |\n| --- | --- | --- | --- |")
            for sc in subcats_by_fn.get("MEASURE", []):
                page_prose_lines.append(f"| {sc['subcategory_id']} | {sc['subcategory_text']} | {sc['category_id']} | {sc['category_text']} |")
            page_prose_lines.append("<!-- TABLE_END -->\n")

        elif p_num == 37 and "MANAGE" not in inserted_tables:
            inserted_tables.add("MANAGE")
            page_prose_lines.append("\n<!-- TABLE_START table_id=table_4 -->\n### Table 4: Categories and Subcategories for the MANAGE Function\n")
            page_prose_lines.append("| Subcategory ID | Subcategory Text | Category ID | Category Text |\n| --- | --- | --- | --- |")
            for sc in subcats_by_fn.get("MANAGE", []):
                page_prose_lines.append(f"| {sc['subcategory_id']} | {sc['subcategory_text']} | {sc['category_id']} | {sc['category_text']} |")
            page_prose_lines.append("<!-- TABLE_END -->\n")

        md_lines.append("\n\n".join(page_prose_lines))

    full_md = "\n".join(md_lines)
    out_file = PARSED_DATA_DIR / "nist_ai_rmf_1.0.md"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(full_md)
    logger.info(f"Saved parsed Markdown to {out_file} ({len(full_md):,} chars).")

    return full_md, subcategories, figures


def parse_nist_ai_600_to_markdown() -> Tuple[str, List[Dict[str, Any]]]:
    """
    Parse NIST AI 600-1 (Generative AI Profile) into clean Markdown with heading levels.
    Emits data/parsed/nist_ai_rmf_genai.md and structured action rows.
    """
    pdf_path = DATA_DIR / "nist_ai_rmf_genai.pdf"
    doc = fitz.open(str(pdf_path))

    actions = parse_action_tables_ai600(pdf_path)

    md_lines = ["# NIST AI RMF Generative AI Profile (NIST AI 600-1)\n"]
    current_sec = "Front matter"
    actions_inserted = False

    for p_idx in range(len(doc)):
        p_num = p_idx + 1
        page = doc[p_idx]
        printed_page = p_num - 4 if p_num >= 5 else None

        page_header_str = f"\n<!-- PAGE pdf_page={p_num} printed_page={printed_page or 'None'} -->\n"
        md_lines.append(page_header_str)

        # On action table pages 17 to 50, we skip raw table fragments and insert the structured action table at page 17
        if 17 <= p_num <= 50:
            if not actions_inserted and p_num == 17:
                actions_inserted = True
                md_lines.append("\n<!-- TABLE_START table_id=action_table -->\n## 3. Generative AI Profile Suggested Actions\n")
                md_lines.append("| Action ID | Function | Subcategory | Suggested Action | GAI Risks | AI Actor Tasks | PDF Page | Printed Page |\n| --- | --- | --- | --- | --- | --- | --- | --- |")
                for act in actions:
                    md_lines.append(
                        f"| {act['action_id']} | {act['function']} | {act['subcategory']} | {act['suggested_action']} | {act['gai_risks']} | {act['ai_actor_tasks']} | {act['pdf_page']} | {act['printed_page']} |"
                    )
                md_lines.append("<!-- TABLE_END -->\n")
            continue

        blocks = page.get_text("blocks")
        page_prose_lines = []

        # On front matter pages 1-4, keep section as Front matter and skip heading detection
        if p_num <= 4:
            for b in blocks:
                x0, y0, x1, y1, text = b[0], b[1], b[2], b[3], b[4].strip()
                if not text or y0 < 50 or y1 > 730:
                    continue
                page_prose_lines.append(clean_text_typography(text))
            md_lines.append("\n\n".join(page_prose_lines))
            continue

        for b in blocks:
            x0, y0, x1, y1, text = b[0], b[1], b[2], b[3], b[4].strip()
            if not text or y0 < 50 or y1 > 730:
                continue

            cleaned_block = clean_text_typography(text)

            # Detect headings
            # Major Section e.g. "1. Introduction", "2. Overview of Risks...", "3. Suggested Actions..."
            sec1_match = re.match(r"^([1-4])\.\s*\n?\s*([A-Z][^\n]+)", cleaned_block)
            if sec1_match and len(sec1_match.group(2).strip()) < 80:
                h_text = f"{sec1_match.group(1)}. {sec1_match.group(2).strip()}"
                current_sec = h_text
                page_prose_lines.append(f"\n# {h_text}\n")
                continue

            # Sub-risk headings e.g. "2.2. Confabulation", "2.1. CBRN Information"
            risk_match = re.match(r"^(2\.\d+)\.?\s*\n?\s*([A-Z][^\n]+)", cleaned_block)
            if risk_match and len(risk_match.group(2).strip()) < 80:
                r_text = f"{risk_match.group(1)}. {risk_match.group(2).strip()}"
                current_sec = f"2. Overview of Risks Unique to or Exacerbated by GAI > {r_text}"
                page_prose_lines.append(f"\n## {r_text}\n")
                continue

            page_prose_lines.append(cleaned_block)

        md_lines.append("\n\n".join(page_prose_lines))

    full_md = "\n".join(md_lines)
    out_file = PARSED_DATA_DIR / "nist_ai_rmf_genai.md"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(full_md)
    logger.info(f"Saved parsed Markdown to {out_file} ({len(full_md):,} chars).")

    actions_file = PARSED_DATA_DIR / "genai_actions.json"
    with open(actions_file, "w", encoding="utf-8") as f:
        json.dump(actions, f, indent=2)
    logger.info(f"Saved {len(actions)} actions to {actions_file}.")

    return full_md, actions


if __name__ == "__main__":
    print("[-] Parsing both NIST documents into inspectable Markdown & structured tables...")
    md1, subcats, figs = parse_nist_ai_100_to_markdown()
    print(f"[OK] AI 100-1 parsed: {len(subcats)} subcategories, {len(figs)} figures.")
    md2, acts = parse_nist_ai_600_to_markdown()
    print(f"[OK] AI 600-1 parsed: {len(acts)} action rows.")
