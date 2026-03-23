import os
import json
import re
from pathlib import Path
import pdfplumber
from openai import OpenAI

# =========================
# 1. CONFIG & SYSTEM SETUP
# =========================
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL_NAME = "gpt-5.1"  # Optimized for your specific model choice

INPUT_FOLDER = Path("Data/Test")
OUTPUT_FOLDER = Path("Data/Processed")
INPUT_FOLDER.mkdir(parents=True, exist_ok=True)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# =========================
# 2. THE SANITIZATION ENGINE (The "Anti-Cheat" Layer)
# =========================
# Regex patterns to catch judicial conclusions that leak the outcome (y) into features (X)
STOP_PATTERNS = [
    r"\bfiduciary\b", r"\bbreach\b", r"\bliable\b", r"\bwrongful\b", 
    r"\bdishonest\b", r"\bbad faith\b", r"\bentitled\b", r"\bowed\b",
    r"\bthe court held\b", r"\bjudge found\b", r"\balleged\b", r"\basserted\b"
]

# Neutralizes outcome-indicative language into raw behavioral data
REWRITE_MAP = {
    "fiduciary position": "senior management role",
    "fiduciary duties": "management-level obligations",
    "breached his duties": "performed disputed conduct",
    "wrongfully transferred": "initiated transfer of",
    "failed to disclose": "did not provide record of",
    "liable for": "subject of dispute regarding",
    "asserted that": "stated that",
    "occupied a fiduciary position": "held a corporate management role"
}

def sanitize_legal_text(text):
    """Hard-scrubs the text to ensure zero target leakage."""
    if not text: return ""
    # 1. Apply Neutralization Map
    for bad, good in REWRITE_MAP.items():
        text = re.sub(re.escape(bad), good, text, flags=re.IGNORECASE)
    # 2. Hard Scrub remaining conclusion-words
    for pattern in STOP_PATTERNS:
        text = re.sub(pattern, "[SCRUBBED]", text, flags=re.IGNORECASE)
    return text.strip()

# =========================
# 3. HIGH-DENSITY PROMPT TEMPLATE
# =========================
PROMPT_TEMPLATE = """
You are a Senior Legal Data Engineer. Extract high-density, predictive data into strict JSON.

### EXTRACTION RULES:
1. MULTI-PARTY: Return one JSON object PER unique party (e.g. Plaintiff 1, Plaintiff 2, Defendant).
2. HIGH-DENSITY FACTS: DO NOT SUMMARIZE. Extract raw behavioral data including:
   - Specific dates (e.g., '15 January 2001')
   - Specific entity names (e.g., 'XIHARI', 'Huadian')
   - Communication counts (e.g., 'nine e-mails exchanged')
   - Specific job grades and titles (e.g., 'Grade M1', 'General Manager')
   - NO OUTCOMES: If the text says "P1 was liable for breach," extract ONLY the behavior "P1 sent emails to rival." 
   - No information gained during the trial in court (e.g. cross-examination, defendant denied that)
3. NEUTRALITY: Use observable conduct only. Move judicial inferences to 'Judicial_Reasoning_Log'.

VALID Fact_Type values: [PARTY_INFO, CHRONOLOGY, CORPORATE_ROLE, CONDUCT, COMMUNICATION, DOCUMENT, CONTRACT_EVENT, FINANCIAL_EVENT, RELATED_PARTY_EVENT, AUTHORITY_EVENT, DISCLOSURE_EVENT, BOARD_ACTION, RELATIONSHIP]

### JSON SCHEMA:
{
  "results": [
    {
      "Metadata": { "Judge": "string", "Date": "YYYY-MM-DD", "Tribunal_Court": "string" },
      "Party_Details": {
        "Role": "Plaintiff | Defendant",
        "Name": "string",
        "Facts": [ { "Fact_Type": "string", "Text": "string" } ],
        "Issue": "string",
        "Rule": "string",
        "Application": "string",
        "Conclusion": "string",
        "Judicial_Reasoning_Log": "string"
      }
    }
  ]
}

### FEW-SHOT ANCHOR:
Input: "The 1st Plaintiff (Holding Co) had its claim dismissed because he owed it no duty."
Output Row 1: {
  "Party_Details": {
    "Name": "Holding Co",
    "Facts": [{"Fact_Type": "PARTY_INFO", "Text": "The 1st Plaintiff is a non-operating holding company."}],
    "Issue": "Whether management duties extend to a non-operating parent entity.",
    "Conclusion": "Dismissed",
    "Judicial_Reasoning_Log": "The court held that the first plaintiff, as a non-operating holding company with no direct employment relationship with the defendant, was not owed fiduciary duties by him and that the first plaintiff’s claim against the defendant should be dismissed, with costs to be fixed after submissions on quantum."
  }
}

### JUDGMENT TEXT TO PROCESS:
"""

# =========================
# 4. EXECUTION PIPELINE
# =========================
def process_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        full_text = " ".join([p.extract_text() or "" for p in pdf.pages])
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME, 
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "Return high-density multi-party JSON. Facts must be arrays of objects."},
                {"role": "user", "content": PROMPT_TEMPLATE + full_text[:100000]}
            ]
        )
        data = json.loads(response.choices[0].message.content)
        results = data.get("results", [])

        processed_rows = []
        for row in results:
            party = row.get("Party_Details", {})
            
            # Neutralize and Scrub all feature fields (Issue, Rule, Application, Facts)
            party["Issue"] = sanitize_legal_text(party.get("Issue"))
            party["Rule"] = sanitize_legal_text(party.get("Rule"))
            party["Application"] = sanitize_legal_text(party.get("Application"))
            
            for f in party.get("Facts", []):
                f["Text"] = sanitize_legal_text(f.get("Text"))
            
            # Separate Label (y) from Features (X)
            row["Party_Details"] = party
            row["Label"] = party.pop("Conclusion", "Unknown")
            processed_rows.append(row)
            
        return processed_rows
    except Exception as e:
        print(f"Error processing {pdf_path.name}: {e}")
        return []

def main():
    pdf_files = list(INPUT_FOLDER.glob("*.pdf"))
    for pdf_file in pdf_files:
        print(f"--- Sanitizing: {pdf_file.name} ---")
        clean_data = process_pdf(pdf_file)
        
        if clean_data:
            output_path = OUTPUT_FOLDER / f"{pdf_file.stem}.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(clean_data, f, indent=2, ensure_ascii=False)
            print(f"Success: Generated {len(clean_data)} high-density rows.")

if __name__ == "__main__":
    main()