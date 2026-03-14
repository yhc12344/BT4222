import os
import json
from pathlib import Path
import pdfplumber
from openai import OpenAI

# =========================
# CONFIG
# =========================

client = OpenAI(api_key=os.getenv("KEY"))

INPUT_FOLDER = Path("Data/Test")
OUTPUT_FOLDER = Path("Data/Processed")

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# Leakage words to filter from Facts
LEAK_WORDS = [
    "liable",
    "dismissed",
    "awarded",
    "convicted",
    "judgment",
    "ordered",
    "held that",
    "damages"
]

# =========================
# PROMPT
# =========================

PROMPT_TEMPLATE = """
You are a Senior Legal Data Engineer.

Extract structured data from Singapore legal judgment text.

STRICT OUTPUT RULES:
1. Return ONLY valid JSON.
2. No markdown.
3. No explanations.
4. Every object must contain all keys.
5. Missing values = null.
6. Do NOT infer facts not explicitly stated.
7. Create ONE object for EACH unique party.
8. Duplicate metadata across all party rows.

ANTI-DATA-LEAKAGE RULES:
9. Facts must contain only objective factual background.
10. Facts must NOT include judgment outcome, liability, conviction, damages, or legal conclusion language.
11. Issue must only state the legal question.
12. Rule must only state legal principles/statutes cited.
13. Application must explain reasoning only, without final decision wording.
14. Conclusion alone contains final judgment outcome.

VALID ROLE VALUES:
- Plaintiff
- Defendant

DATE FORMAT:
- YYYY-MM-DD only

LAWYERS:
- Always array
- Empty array if missing

SCHEMA:
{
  "results": [
    {
      "Metadata": {
        "Judge": "string or null",
        "Date": "YYYY-MM-DD or null",
        "Hearing_Duration": "string or null",
        "Tribunal_Court": "string or null",
        "Sector": "string or null",
        "Lawyers": []
      },
      "Party_Details": {
        "Role": "Plaintiff | Defendant",
        "Name": "string",
        "Facts": "objective facts only",
        "Issue": "legal issue only or null",
        "Rule": "legal rule only or null",
        "Application": "reasoning only, no conclusion",
        "Conclusion": "final outcome only or null"
      }
    }
  ]
}

JUDGMENT TEXT:
"""

# =========================
# PDF TEXT EXTRACTION
# =========================

def extract_pdf_text(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return " ".join(text.split())

# =========================
# GPT CALL
# =========================

def process_pdf(pdf_path):
    text = extract_pdf_text(pdf_path)

    response = client.chat.completions.create(
        model="gpt-5.1",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "You extract structured legal data strictly."
            },
            {
                "role": "user",
                "content": PROMPT_TEMPLATE + text
            }
        ]
    )

    return response.choices[0].message.content

# =========================
# LEAKAGE DETECTION
# =========================

def contains_leakage(facts):
    if not facts:
        return False
    facts_lower = facts.lower()
    return any(word in facts_lower for word in LEAK_WORDS)

# =========================
# MAIN LOOP
# =========================

def main():
    for pdf_file in INPUT_FOLDER.glob("*.pdf"):

        print(f"Processing {pdf_file.name}")

        try:
            result_json = process_pdf(pdf_file)

            parsed = json.loads(result_json)["results"]

            clean_rows = []

            for row in parsed:
                facts = row["Party_Details"]["Facts"]

                if contains_leakage(facts):
                    print(f"Leakage detected in {pdf_file.name} -> skipped row")
                    continue

                clean_rows.append(row)

            output_file = OUTPUT_FOLDER / f"{pdf_file.stem}.json"

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(clean_rows, f, indent=2, ensure_ascii=False)

            print(f"Saved -> {output_file}")

        except Exception as e:
            print(f"Error processing {pdf_file.name}: {e}")

# =========================
# RUN
# =========================

if __name__ == "__main__":
    main()