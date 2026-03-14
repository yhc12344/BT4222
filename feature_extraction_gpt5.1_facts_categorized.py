import os
import json
from pathlib import Path
import pdfplumber
from openai import OpenAI

# =========================
# CONFIG
# =========================

# Ensure your key is set in environment variables
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

INPUT_FOLDER = Path("Data/Test")
OUTPUT_FOLDER = Path("Data/Processed")

INPUT_FOLDER.mkdir(parents=True, exist_ok=True)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

LEAK_WORDS = [
    "liable",
    "dismissed",
    "convicted",
    "held that",
    "judgment entered",
    "court held",
    "ordered that"
]

# =========================
# PROMPT (Now with Few-Shot Anchoring)
# =========================

PROMPT_TEMPLATE = """
You are a Senior Legal Data Engineer. Return strict JSON only. Facts must always be an array of objects. Never return markdown. Never omit keys.
Extract structured data into the EXACT JSON format below.

### CRITICAL RULE: FACT ARRAY
The "Facts" field is NOT a string. It is an ARRAY of OBJECTS. 
You must split the factual narrative into individual sentences. 
Each sentence must be its own object with a "Fact_Type" and "Text".

### MULTI-PARTY RULE (CRITICAL)
- If a case has 3 Plaintiffs and 1 Defendant, you MUST return an array of 4 objects.
- Do NOT group multiple parties into one object.
- Every object must be a "Flat Row" containing both the shared Metadata and that specific party's Details.

### FACT CLASSIFICATION
- The "Facts" field is an ARRAY of objects.
- Split the factual narrative into individual sentences.
- Assign ONE Fact_Type per sentence: [PARTY_INFO, CHRONOLOGY, CONDUCT, DOCUMENT, CONTRACTUAL_BASE, FINANCIAL_FACT, CORPORATE_STRUCTURE, COMMUNICATION, REGULATORY_FACT, PROCEDURAL_FACT, EVIDENCE, DAMAGES_FACT, RELATIONSHIP, STATE_OF_MIND, BOARD_ACTION, SHAREHOLDER_ACTION, FIDUCIARY_CONDUCT, AUTHORITY_FACT, DISCLOSURE_FACT, CONFLICT_OF_INTEREST].

### EXAMPLE OF CORRECT FACT CLASSIFICATION:
Input: "The Plaintiff is a Singapore company. It signed a contract on 5 May 2023."
Output: 
"Facts": [
  {"Fact_Type": "PARTY_INFO", "Text": "The Plaintiff is a Singapore company."},
  {"Fact_Type": "CHRONOLOGY", "Text": "It signed a contract on 5 May 2023."}
]

### ANTI-LEAKAGE:
Do NOT include "The court held", "liable", or any outcomes in the Facts array. Move those to Application/Conclusion.

### SCHEMA:
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
        "Facts": [
          {
            "Fact_Type": "string",
            "Text": "string"
          }
        ],
        "Issue": "string or null",
        "Rule": "string or null",
        "Application": "string or null",
        "Conclusion": "string or null"
      }
    }
  ]
}

JUDGMENT TEXT:
"""

# =========================
# FUNCTIONS
# =========================

def extract_pdf_text(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return " ".join(text.split())

def contains_leakage(facts_list):
    """Fixed: Now correctly iterates through the list of fact objects."""
    if not isinstance(facts_list, list):
        return False
    
    # Merge all sentence texts into one searchable string
    full_factual_text = " ".join([item.get("Text", "") for item in facts_list]).lower()
    
    for word in LEAK_WORDS:
        if word in full_factual_text:
            return True
    return False

def process_pdf(pdf_path):
    text = extract_pdf_text(pdf_path)

    response = client.chat.completions.create(
        model="gpt-5.1", # Ensure this model string is supported by your provider
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a legal analyst. You ALWAYS return an array of objects for the Facts field. You never return a string for Facts."},
            {"role": "user", "content": PROMPT_TEMPLATE + text}
        ]
    )
    return response.choices[0].message.content

# =========================
# MAIN LOOP
# =========================

def main():
    pdf_files = list(INPUT_FOLDER.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {INPUT_FOLDER}")
        return

    for pdf_file in pdf_files:
        print(f"--- Processing: {pdf_file.name} ---")
        try:
            raw_json = process_pdf(pdf_file)
            data = json.loads(raw_json)
            
            # OpenAI sometimes wraps the array in a "results" key or similar
            parsed_rows = data.get("results", [])

            clean_rows = []
            for row in parsed_rows:
                facts = row.get("Party_Details", {}).get("Facts", [])
                
                if contains_leakage(facts):
                    print(f"  [!] Leakage detected in party {row['Party_Details']['Name']}. Skipping row.")
                    continue
                
                clean_rows.append(row)

            if clean_rows:
                output_path = OUTPUT_FOLDER / f"{pdf_file.stem}.json"
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(clean_rows, f, indent=2, ensure_ascii=False)
                print(f"  [+] Saved {len(clean_rows)} party rows.")
            else:
                print("  [?] No clean data extracted for this file.")

        except Exception as e:
            print(f"  [X] Error: {e}")

if __name__ == "__main__":
    main()