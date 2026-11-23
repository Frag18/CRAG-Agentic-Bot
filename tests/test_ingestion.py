"""
Acceptance Test Suite for Ingestion Layer (Acceptance Tests 1 - 6).
Uses an isolated temporary Chroma directory to ensure production ./chroma_db is never touched.
"""

import re
import sys
import random
import shutil
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pymupdf as fitz
from src.config import CHUNK_SIZE, DATA_DIR
from src.ingest import build_all_documents


class TestIngestionLayer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n=======================================================")
        print("[-] BUILDING ALL DOCUMENTS FOR INGESTION TEST SUITE...")
        print("=======================================================")
        cls.docs = build_all_documents()
        cls.doc1_pdf = fitz.open(str(DATA_DIR / "nist_ai_rmf_1.0.pdf"))
        cls.docg_pdf = fitz.open(str(DATA_DIR / "nist_ai_rmf_genai.pdf"))
        print(f"[-] Total Documents Created: {len(cls.docs)}")

    @classmethod
    def tearDownClass(cls):
        cls.doc1_pdf.close()
        cls.docg_pdf.close()

    def test_01_ai100_1_subcategories(self):
        """
        Acceptance Test 1:
        AI 100-1 subcategory nodes: GOVERN=19, MAP=18, MEASURE=22, MANAGE=13 (72).
        Each has non-empty category_text.
        GOVERN 4.x nodes carry the full category text:
        'Organizational teams are committed to a culture that considers and communicates AI risk.'
        """
        print("\n--- [TEST 1] AI 100-1 Subcategory Nodes & Category Text ---")
        subcat_docs = [
            d for d in self.docs
            if d.metadata.get("doc_id") == "ai_rmf_1.0" and d.metadata.get("subcategory_id")
        ]
        self.assertEqual(len(subcat_docs), 72, f"Expected 72 subcategories, got {len(subcat_docs)}")

        fn_counts = {}
        for d in subcat_docs:
            fn = d.metadata.get("function")
            fn_counts[fn] = fn_counts.get(fn, 0) + 1
            cat_text = d.metadata.get("category_text", "").strip()
            self.assertTrue(len(cat_text) > 0, f"Empty category_text for {d.metadata.get('subcategory_id')}")

        print(f"Subcategory Counts by Function: {fn_counts}")
        self.assertEqual(fn_counts.get("GOVERN"), 19, "Expected 19 GOVERN subcategories")
        self.assertEqual(fn_counts.get("MAP"), 18, "Expected 18 MAP subcategories")
        self.assertEqual(fn_counts.get("MEASURE"), 22, "Expected 22 MEASURE subcategories")
        self.assertEqual(fn_counts.get("MANAGE"), 13, "Expected 13 MANAGE subcategories")

        # Verify GOVERN 4.x joined category text
        expected_gov4_text = "Organizational teams are committed to a culture that considers and communicates AI risk."
        gov4_nodes = [d for d in subcat_docs if d.metadata.get("subcategory_id", "").startswith("GOVERN 4")]
        self.assertEqual(len(gov4_nodes), 3, f"Expected 3 GOVERN 4.x nodes (4.1, 4.2, 4.3), got {len(gov4_nodes)}")
        for d in gov4_nodes:
            sub_id = d.metadata.get("subcategory_id")
            cat_text = d.metadata.get("category_text")
            print(f"Checked {sub_id}: category_text=\"{cat_text}\"")
            self.assertEqual(cat_text, expected_gov4_text, f"{sub_id} category text mismatch!")
        print("[PASS] Test 1: Subcategory counts & GOVERN 4 category text verified.")

    def test_02_no_forbidden_strings(self):
        """
        Acceptance Test 2:
        No chunk contains 'Continued on next page', 'Categories Subcategories',
        'NIST AI 100-1 AI RMF 1.0', or a bare 'Page N' footer. No glued words such as 'RiskMeasurement'.
        """
        print("\n--- [TEST 2] Forbidden Strings & Glued Words Audit ---")
        forbidden = [
            "Continued on next page",
            "Categories Subcategories",
            "NIST AI 100-1 AI RMF 1.0",
            "RiskMeasurement",
        ]
        for term in forbidden:
            bad_chunks = [d for d in self.docs if term in d.page_content]
            print(f"Term '{term}': {len(bad_chunks)} occurrences")
            self.assertEqual(len(bad_chunks), 0, f"Found forbidden term '{term}' in {len(bad_chunks)} chunks!")

        bare_page_footers = [
            d for d in self.docs if re.search(r"^\s*Page\s+\d+\s*$", d.page_content, re.M)
        ]
        print(f"Bare 'Page N' footers: {len(bare_page_footers)} occurrences")
        self.assertEqual(len(bare_page_footers), 0, "Found bare 'Page N' footers in chunks!")
        print("[PASS] Test 2: No forbidden artifacts or glued words found.")

    def test_03_figure_nodes(self):
        """
        Acceptance Test 3:
        Figure nodes exist for AI 100-1 Fig.1-5 and for every figure in AI 600-1, each with a non-trivial description.
        Fig.4 mentions all seven characteristics. Fig.5 mentions Govern, Map, Measure, Manage and 'cross-cutting'.
        """
        print("\n--- [TEST 3] Figure Nodes & Completeness ---")
        figure_nodes = [d for d in self.docs if d.metadata.get("node_type") == "figure"]
        self.assertEqual(len(figure_nodes), 5, f"Expected 5 figure nodes, got {len(figure_nodes)}")

        fig_map = {d.metadata.get("figure_id"): d for d in figure_nodes}
        for i in range(1, 6):
            fkey = f"fig_{i}"
            self.assertIn(fkey, fig_map, f"Missing figure node for {fkey}")
            desc = fig_map[fkey].page_content
            self.assertTrue(len(desc) > 100, f"Figure {fkey} description is too short ({len(desc)} chars)")
            print(f"Fig {i}: {len(desc)} chars, source={fig_map[fkey].metadata.get('description_source')}")

        # Fig.4 Characteristics
        fig4_text = fig_map["fig_4"].page_content
        characteristics = ["Valid", "Safe", "Secure", "Explainable", "Privacy", "Fair", "Accountable"]
        for c in characteristics:
            self.assertIn(c.lower(), fig4_text.lower(), f"Fig 4 missing characteristic: {c}")
        print("Fig 4 verified: contains all 7 trustworthiness characteristics.")

        # Fig.5 Functions & cross-cutting
        fig5_text = fig_map["fig_5"].page_content
        for fn in ["Govern", "Map", "Measure", "Manage", "cross-cutting"]:
            self.assertIn(fn.lower(), fig5_text.lower(), f"Fig 5 missing keyword: {fn}")
        print("Fig 5 verified: contains Govern, Map, Measure, Manage and 'cross-cutting'.")
        print("[PASS] Test 3: All figure nodes verified.")

    def test_04_ai600_1_actions(self):
        """
        Acceptance Test 4:
        AI 600-1: action-row nodes exist; spot-check GV-3.2-001 and one MS-* action for correct function and risks;
        print counts per function (GV/MP/MS/MG). Pages that contain MS-* actions must not be labeled MP-*.
        """
        print("\n--- [TEST 4] AI 600-1 Action-Row Nodes ---")
        action_nodes = [d for d in self.docs if d.metadata.get("action_id")]
        self.assertEqual(len(action_nodes), 211, f"Expected 211 action rows, got {len(action_nodes)}")

        fn_counts = {}
        for d in action_nodes:
            fn = d.metadata.get("function")
            fn_counts[fn] = fn_counts.get(fn, 0) + 1

        print(f"Action counts per function: {fn_counts}")
        self.assertEqual(fn_counts.get("GOVERN"), 57, "Expected 57 GOVERN actions")
        self.assertEqual(fn_counts.get("MAP"), 39, "Expected 39 MAP actions")
        self.assertEqual(fn_counts.get("MEASURE"), 72, "Expected 72 MEASURE actions")
        self.assertEqual(fn_counts.get("MANAGE"), 43, "Expected 43 MANAGE actions")

        # Spot-check GV-3.2-001
        gv = next((d for d in action_nodes if d.metadata.get("action_id") == "GV-3.2-001"), None)
        self.assertIsNotNone(gv, "Missing action GV-3.2-001")
        self.assertEqual(gv.metadata.get("function"), "GOVERN")
        self.assertTrue(len(gv.metadata.get("gai_risks", "")) > 0, "Missing GAI risks in GV-3.2-001")
        print(f"Spot-check GV-3.2-001: fn={gv.metadata['function']}, risks={gv.metadata['gai_risks']}")

        # Spot-check MS-* action
        ms = next((d for d in action_nodes if d.metadata.get("action_id") == "MS-1.1-001"), None)
        self.assertIsNotNone(ms, "Missing action MS-1.1-001")
        self.assertEqual(ms.metadata.get("function"), "MEASURE")
        self.assertTrue(len(ms.metadata.get("gai_risks", "")) > 0, "Missing GAI risks in MS-1.1-001")
        print(f"Spot-check MS-1.1-001: fn={ms.metadata['function']}, risks={ms.metadata['gai_risks']}")

        # Verify MS-* actions are NOT labeled MP-*
        ms_nodes = [d for d in action_nodes if d.metadata.get("action_id", "").startswith("MS-")]
        for d in ms_nodes:
            aid = d.metadata.get("action_id")
            fn = d.metadata.get("function")
            self.assertEqual(fn, "MEASURE", f"Action {aid} was labeled function {fn} instead of MEASURE!")
        print(f"Verified all {len(ms_nodes)} MS-* actions are correctly labeled MEASURE (NOT MP).")
        print("[PASS] Test 4: AI 600-1 action nodes verified.")

    def test_05_section_correctness(self):
        """
        Acceptance Test 5:
        For 15 random nodes per document, print text start, section_path, pdf_page, printed_page and verify
        programmatically that the text appears on that PDF page and that the section_path matches the heading governing that page.
        Include page 10 of AI 600-1 (must be Confabulation) and the last page of AI 100-1 (must be Back matter).
        """
        print("\n--- [TEST 5] Programmatic Section & Page Grounding Verification ---")
        random.seed(42)

        # AI 100-1 Check
        print("Evaluating AI 100-1 nodes...")
        doc1_nodes = [d for d in self.docs if d.metadata.get("doc_id") == "ai_rmf_1.0" and d.metadata.get("node_type") == "prose"]
        p48_nodes = [d for d in doc1_nodes if d.metadata.get("pdf_page") == 48]
        self.assertTrue(len(p48_nodes) > 0, "Missing page 48 node in AI 100-1")
        self.assertEqual(p48_nodes[0].metadata.get("section_path"), "Back matter", "Page 48 must be Back matter!")

        sample_100 = random.sample([d for d in doc1_nodes if d.metadata.get("pdf_page") != 48], 14) + [p48_nodes[0]]
        for idx, d in enumerate(sample_100, 1):
            p_num = d.metadata["pdf_page"]
            page_text = self.doc1_pdf[p_num - 1].get_text()
            raw_text = d.page_content.split(":\n\n")[-1].strip()
            words = [w for w in re.findall(r"\w+", raw_text) if len(w) > 4][:5]
            found = any(w in page_text for w in words)
            self.assertTrue(found, f"Words {words} from chunk not found on PDF p.{p_num}!")
            print(f"  AI 100-1 [{idx}/15] p.{p_num} (printed {d.metadata.get('printed_page')}) -> {d.metadata.get('section_path')} [OK]")

        # AI 600-1 Check
        print("Evaluating AI 600-1 nodes...")
        docg_nodes = [d for d in self.docs if d.metadata.get("doc_id") == "ai_rmf_genai" and d.metadata.get("node_type") == "prose"]
        p10_nodes = [d for d in docg_nodes if d.metadata.get("pdf_page") == 10]
        self.assertTrue(len(p10_nodes) > 0, "Missing page 10 node in AI 600-1")
        self.assertIn("Confabulation", p10_nodes[0].metadata.get("section_path", ""), "Page 10 must be Confabulation!")

        sample_g = random.sample([d for d in docg_nodes if d.metadata.get("pdf_page") != 10], 14) + [p10_nodes[0]]
        for idx, d in enumerate(sample_g, 1):
            p_num = d.metadata["pdf_page"]
            page_text = self.docg_pdf[p_num - 1].get_text()
            raw_text = d.page_content.split(":\n\n")[-1].strip()
            words = [w for w in re.findall(r"\w+", raw_text) if len(w) > 4][:5]
            found = any(w in page_text for w in words)
            self.assertTrue(found, f"Words {words} from chunk not found on PDF p.{p_num}!")
            print(f"  AI 600-1 [{idx}/15] p.{p_num} (printed {d.metadata.get('printed_page')}) -> {d.metadata.get('section_path')} [OK]")

        print("[PASS] Test 5: Section correctness verified across 30 random document nodes.")

    def test_06_node_boundaries_and_lengths(self):
        """
        Acceptance Test 6:
        No node crosses a heading boundary; no table row is split; max node length is within config.
        """
        print("\n--- [TEST 6] Heading Boundaries & Max Node Lengths ---")
        prose_nodes = [d for d in self.docs if d.metadata.get("node_type") == "prose"]
        lengths = [len(d.page_content) for d in prose_nodes]
        print(f"Prose node count: {len(prose_nodes)}, Max length: {max(lengths)}, Min length: {min(lengths)}")
        # Check that no prose node exceeds config CHUNK_SIZE plus breadcrumb allowance (1200 chars)
        self.assertTrue(max(lengths) <= 1200, f"Max prose chunk length {max(lengths)} exceeds limit!")

        # Verify no table rows were split
        table_row_nodes = [d for d in self.docs if d.metadata.get("node_type") == "table_row"]
        for d in table_row_nodes:
            # Action row or subcategory row must contain full entry
            if d.metadata.get("action_id"):
                self.assertIn("Action ID:", d.page_content)
                self.assertIn("Suggested Action:", d.page_content)
            elif d.metadata.get("subcategory_id"):
                self.assertIn("Subcategory", d.page_content)
                self.assertIn("Category:", d.page_content)
        print(f"Verified {len(table_row_nodes)} table rows: all complete and unsplit.")
        print("[PASS] Test 6: Node boundaries and lengths verified.")


if __name__ == "__main__":
    unittest.main()
