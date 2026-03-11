import os
import re
import fitz
import numpy as np

folder = os.path.join("Data", "PDFs")

word_counts = []
sentence_counts = []
token_counts = []

# Walk through all subfolders recursively
for root, dirs, files in os.walk(folder):
    for file in files:
        if file.endswith(".pdf"):
            path = os.path.join(root, file)

        doc = fitz.open(path)
        text = ""

        for page in doc:
            text += page.get_text()

        words = re.findall(r"\w+", text)
        sentences = re.split(r"[.!?]", text)

        word_counts.append(len(words))
        sentence_counts.append(len([s for s in sentences if s.strip()]))

# Document statistics
print("Document Statistics")
print("Min words:", min(word_counts))
print("Max words:", max(word_counts))
print("Mean words:", int(np.mean(word_counts)))
print("Median words:", int(np.median(word_counts)))

print()

# Token statistics
print("Token Statistics")
print("Avg tokens per document:", int(np.mean(word_counts)))
print("Avg sentences per document:", int(np.mean(sentence_counts)))