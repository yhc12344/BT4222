import os
import json
from pathlib import Path
import pdfplumber
from openai import OpenAI

# =========================
# 1. CONFIG
# =========================
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL_NAME = "gpt-5.1"

INPUT_FOLDER = Path("Data/PDFs/Test")
OUTPUT_FOLDER = Path("Data/Processed")

INPUT_FOLDER.mkdir(parents=True, exist_ok=True)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# =========================
# 2. PDF TEXT EXTRACTION
# =========================
def extract_pdf_text(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return " ".join(text.split())

# =========================
# 3. SAFE JSON LOAD
# =========================
def safe_json_load(content):
    try:
        return json.loads(content)
    except:
        import re
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise

# =========================
# 4. LABEL PROMPT
# =========================
LABEL_PROMPT = """
You are an expert legal judgment label extraction engine.

Your task:
Extract ALL unique Plaintiff–Defendant pairs from the provided judgment. 
For every unique pair, assign the correct legal outcome labels.

Plaintiff_Label Options:
- Claim Allowed
- Claim Dismissed
- Claim Allowed In-part

Defendant_Label Options:
- Liable
- Not Liable

STRICT RULES:
1. Pairwise Isolation: You must evaluate the verdict STRICTLY between the specific Plaintiff and Defendant in the pair. Ignore how the Plaintiff fared against other defendants.
2. Complete Success: If the Plaintiff completely won against this specific Defendant, assign:
   Plaintiff_Label = Claim Allowed
   Defendant_Label = Liable
3. Partial Success: If the Plaintiff won on some allegations but lost on others against this specific Defendant, assign:
   Plaintiff_Label = Claim Allowed In-part
   Defendant_Label = Liable
4. Complete Failure: If the court dismissed the claims entirely against this specific Defendant, or if they were merely a "nominal" defendant, assign:
   Plaintiff_Label = Claim Dismissed
   Defendant_Label = Not Liable
5. Exclusions (CRITICAL): Do NOT extract pairs for parties who settled out of court, withdrew their claims, or were struck out before trial. Only extract pairs where the judge rendered a final verdict on the merits.
6. Counterclaims: Treat Plaintiffs by Counterclaim as "Plaintiffs" and Defendants by Counterclaim as "Defendants" for the purpose of row creation.
7. Output format: Return STRICT JSON only, matching the exact schema provided. Do not include markdown code blocks, explanations, or conversational text.

JSON SCHEMA:
{
  "results": [
    {
      "Case_Number": "string",
      "Plaintiff": "string",
      "Defendant": "string",
      "Plaintiff_Label": "Claim Allowed | Claim Dismissed | Claim Allowed In-part",
      "Defendant_Label": "Liable | Not Liable"
    }
  ]
}
Return strict JSON only.
"""

# =========================
# 5. PROCESS ONE PDF
# =========================
def process_pdf(pdf_path):
    full_text = extract_pdf_text(pdf_path)

    try:
        response = client.responses.create(
            model=MODEL_NAME,
            temperature=0,
            input=[
                {
                    "role": "system",
                    "content": "Return strict JSON only."
                },
                {
                    "role": "user",
                    "content": LABEL_PROMPT + full_text[:100000]
                }
            ]
        )

        content = response.output_text
        data = safe_json_load(content)

        return data.get("results", [])

    except Exception as e:
        print(f"Error processing {pdf_path.name}: {e}")
        return []

# =========================
# 6. BATCH RUNNER
# =========================
def main():
    pdf_files = list(INPUT_FOLDER.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found")
        return

    for pdf_file in pdf_files:
        print(f"Processing {pdf_file.name}")

        rows = process_pdf(pdf_file)

        if rows:
            output_path = OUTPUT_FOLDER / f"{pdf_file.stem}_labels.json"

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(rows, f, indent=2, ensure_ascii=False)

            print(f"Saved {len(rows)} labels -> {output_path}")
        else:
            print(f"No labels extracted for {pdf_file.name}")

# =========================
# 7. ENTRY POINT
# =========================
if __name__ == "__main__":
    main()