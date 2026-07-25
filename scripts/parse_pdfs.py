"""
parse_pdfs.py
Extracts text and tables from admission brochure PDFs, page by page.
Run this FIRST on your real PDF to see what you're working with before
building anything else.

Usage:
    python parse_pdfs.py path/to/brochure.pdf
"""

import sys
import json
import os
import pdfplumber


def extract_pdf(pdf_path: str, output_dir: str = "../data/raw/pdfs_extracted"):
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]

    pages_data = []

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        print(f"Opened '{pdf_path}' — {total_pages} pages found.\n")

        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            tables = page.extract_tables()

            # Flag pages that look mostly empty of text -> likely scanned/image-based
            is_likely_scanned = len(text.strip()) < 20

            page_info = {
                "page_number": i,
                "text": text.strip(),
                "tables": tables,
                "likely_scanned": is_likely_scanned,
            }
            pages_data.append(page_info)

            status = "SCANNED/IMAGE (needs OCR)" if is_likely_scanned else "OK"
            table_note = f", {len(tables)} table(s) found" if tables else ""
            print(f"Page {i:>3}/{total_pages}: {status}{table_note}")

    # Save full extraction as JSON for inspection
    out_path = os.path.join(output_dir, f"{base_name}_extracted.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(pages_data, f, indent=2, ensure_ascii=False)

    # Save plain text version too, for quick reading
    txt_path = os.path.join(output_dir, f"{base_name}_full_text.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for p in pages_data:
            f.write(f"\n--- PAGE {p['page_number']} ---\n")
            f.write(p["text"])
            f.write("\n")

    scanned_count = sum(1 for p in pages_data if p["likely_scanned"])
    print(f"\nDone. {scanned_count}/{total_pages} pages look scanned/image-based.")
    print(f"Extracted JSON:  {out_path}")
    print(f"Extracted text:  {txt_path}")

    if scanned_count > 0:
        print("\nNOTE: Some pages had little/no extractable text.")
        print("These likely need OCR (pytesseract) instead of pdfplumber.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python parse_pdfs.py path/to/brochure.pdf")
        sys.exit(1)

    extract_pdf(sys.argv[1])