"""Label extraction utility — extracts ground-truth Plaintiff/Defendant outcome
labels directly from a judgment PDF.

Standalone:
    python src/label_checker.py

Importable (used by audit_cases.py):
    from label_checker import extract_labels_from_pdf
"""
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import pdfplumber
from openai import OpenAI

from config import OPENAI_API_KEY, LABEL_MODEL, PDF_INPUT_ALL, EXTRACTION_OUTPUT

client = OpenAI(api_key=OPENAI_API_KEY)

# =============================================================================
# LABEL NORMALISATION
# Maps label_checker output → audit pipeline canonical labels
# =============================================================================
LABEL_NORMALISATION = {
    "claim allowed":         "Claim Allowed",
    "claim dismissed":       "Claim Dismissed",
    "claim allowed in-part": "Claim Allowed in Part",
    "claim allowed in part": "Claim Allowed in Part",
    "liable":                "Liable",
    "not liable":            "Not Liable",
}

# =============================================================================
# PROMPT
# =============================================================================
LABEL_PROMPT = """
You are an expert legal judgment label extraction engine.

Your task:
Extract ALL unique Plaintiff-Defendant pairs from the provided judgment.
For every unique pair, assign the correct legal outcome labels.

Plaintiff_Label Options:
- Claim Allowed
- Claim Dismissed
- Claim Allowed In-part

Defendant_Label Options:
- Liable
- Not Liable

STRICT RULES:
1. Pairwise Isolation: Evaluate the verdict STRICTLY between the specific Plaintiff and Defendant in the pair. Ignore how the Plaintiff fared against other defendants.
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

# =============================================================================
# HELPERS
# =============================================================================
def safe_json_load(content: str) -> Dict[str, Any]:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise


def read_pdf(pdf_path: Path) -> str:
    text_parts: List[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text_parts.append(page.extract_text() or "")
    return " ".join(" ".join(text_parts).split())


def normalise_label(raw: str) -> str:
    return LABEL_NORMALISATION.get(raw.strip().lower(), raw.strip())


# =============================================================================
# CORE — importable by audit_cases.py
# =============================================================================
def extract_labels_from_pdf(source_text: str) -> List[Dict[str, Any]]:
    """Call the model to extract Plaintiff/Defendant label pairs from judgment text.

    Args:
        source_text: Full text of the judgment PDF (already loaded).

    Returns:
        List of dicts with keys Case_Number, Plaintiff, Defendant,
        Plaintiff_Label, Defendant_Label.
    """
    try:
        response = client.chat.completions.create(
            model=LABEL_MODEL,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "Return strict JSON only."},
                {"role": "user",   "content": LABEL_PROMPT + source_text[:120_000]},
            ],
        )
        content = response.choices[0].message.content or "{}"
        data = safe_json_load(content)
        results = data.get("results", [])

        # Normalise label casing
        for row in results:
            if isinstance(row, dict):
                row["Plaintiff_Label"] = normalise_label(row.get("Plaintiff_Label", ""))
                row["Defendant_Label"] = normalise_label(row.get("Defendant_Label", ""))

        return results

    except Exception as exc:
        print(f"  Label check failed: {exc}")
        return []


# =============================================================================
# STANDALONE RUNNER
# =============================================================================
def main() -> None:
    pdf_files = list(PDF_INPUT_ALL.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {PDF_INPUT_ALL}")
        return

    for pdf_file in pdf_files:
        print(f"Processing {pdf_file.name}")
        source_text = read_pdf(pdf_file)
        results = extract_labels_from_pdf(source_text)

        if results:
            output_path = EXTRACTION_OUTPUT / f"{pdf_file.stem}_labels.json"
            output_path.write_text(
                json.dumps(results, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"  Saved {len(results)} label pairs -> {output_path.name}")
        else:
            print(f"  No labels extracted for {pdf_file.name}")


if __name__ == "__main__":
    main()
