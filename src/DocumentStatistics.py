"""Compute word, sentence, and token statistics across the PDF collection.

Run from the project root:
    python src/DocumentStatistics.py
"""
import re

import fitz  # PyMuPDF
import numpy as np

from config import STATS_INPUT


def compute_statistics(folder) -> None:
    word_counts     = []
    sentence_counts = []

    for pdf_path in folder.rglob("*.pdf"):
        doc  = fitz.open(pdf_path)
        text = "".join(page.get_text() for page in doc)
        doc.close()

        word_counts.append(len(re.findall(r"\w+", text)))
        sentence_counts.append(len([s for s in re.split(r"[.!?]", text) if s.strip()]))

    if not word_counts:
        print(f"No PDFs found in {folder}")
        return

    print(f"Documents analysed : {len(word_counts)}")
    print()
    print("Word counts")
    print(f"  Min    : {min(word_counts):,}")
    print(f"  Max    : {max(word_counts):,}")
    print(f"  Mean   : {int(np.mean(word_counts)):,}")
    print(f"  Median : {int(np.median(word_counts)):,}")
    print()
    print("Sentence counts")
    print(f"  Mean   : {int(np.mean(sentence_counts)):,}")
    print()
    print("Token estimates (words × 1.3)")
    print(f"  Mean   : {int(np.mean(word_counts) * 1.3):,}")


if __name__ == "__main__":
    compute_statistics(STATS_INPUT)
