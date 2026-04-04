import os
import json
import re
from pathlib import Path
import pdfplumber
from openai import OpenAI

# =========================
# 1. CONFIG
# =========================
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL_NAME = "gpt-5.1"

INPUT_FOLDER = Path("Data/PDFs/ALL")
OUTPUT_FOLDER = Path("Data/Processed")

INPUT_FOLDER.mkdir(parents=True, exist_ok=True)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# =========================
# 2. HARD + SOFT LEAKAGE DETECTION
# =========================
STOP_PATTERNS = [
    r"\bfiduciary\b",
    r"\bbreach\b",
    r"\bliable\b",
    r"\bwrongful\b",
    r"\bdishonest\b",
    r"\bbad faith\b",
    r"\bthe court held\b",
    r"\bjudge found\b",
    r"\balleged\b",
    r"\basserted\b",
    r"\baccording to .*? evidence\b",
    r"\bhe testified that\b",
    r"\bcross-examination\b",
    r"\bthe witness stated\b",
    r"\bshow that\b",
    r"\bindicate that\b",
    r"\bdemonstrate that\b"
]

SOFT_PATTERNS = [
    r"\brelevant to\b",
    r"\brequired to\b",
    r"\bexpected to\b",
    r"\bought to\b",
    r"\bsubject to\b",
    r"\bmaterial to\b",
    r"\bshow that\b",
    r"\bindicate that\b",
    r"\bdemonstrate that\b",
    r"\bhe agreed that\b",
    r"\bshe agreed that\b"
]

ALL_PATTERNS = STOP_PATTERNS + SOFT_PATTERNS

# =========================
# 3. LEAKAGE CHECK
# =========================
def contains_leakage(text):
    if not text:
        return False
    return any(re.search(p, text, re.IGNORECASE) for p in ALL_PATTERNS)

# =========================
# 4. TEXT SCRUBBER
# =========================
def scrub_text(text):
    if not text:
        return ""

    for pattern in ALL_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    text = re.sub(r"\s+", " ", text)
    return text.strip()

# =========================
# 5. SAFE JSON LOADER
# =========================
def safe_json_load(content):
    try:
        return json.loads(content)
    except:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise

# =========================
# 6. PROMPT TEMPLATE
# =========================
PROMPT_TEMPLATE = """
### ROLE
You are a Senior Legal Data Engineer specializing in Predictive Analytics.

Your task is to extract:
1. Feature Set = strictly pre-dispute facts
2. Target Label = final outcome only

---

### CORE CONSTRAINT: PRE-DISPUTE RULE

Your feature extraction window closes the moment the dispute began.

Before finalizing each fact, ask:
"Could this fact exist before the judge wrote the judgment?"

If NO → move it to Judicial_Reasoning_Log.

---

### STRICTLY FORBIDDEN IN Facts / Issue / Rule / Application

1. NO JUDICIAL FINDINGS
Forbidden:
"The court found"
"The judge held"
"liable"
"fiduciary"

2. NO TRIAL EVIDENCE
Forbidden:
"cross-examination"
"affidavit"
"testimony"
"admitted"

3. NO PLEADINGS
Forbidden:
"plaintiffs alleged"
"statement of claim"
"asserted"

4. NO VERDICT VERBS
Forbidden:
"breached"
"violated"
"infringed"

5. NO LEGAL INFERENCE
Do not infer:
- motive
- dishonesty
- concealment
- unfairness
- duty
- breach

Describe only observable conduct.

---

### FACT EXTRACTION RULES

- MULTI-PARTY: create one object per unique party
- LEGAL TEAM: 
    - COUNSEL: Extract the full group of individual lawyers listed before the parentheses as an array.
    - LAW FIRM: Extract the firm name found in parentheses (e.g., "Lee & Lee").
- FACT DATING: For every fact, extract the specific ISO date (YYYY-MM-DD). 
    - If only a month is provided, use YYYY-MM-01. 
    - If only a year is provided, use YYYY-01-01.
- DATA HARDENING: Rewrite trial-phase evidence as raw conduct (e.g., "Ledger shows $70k" instead of "He admitted to $70k").
- FLATTEN facts
- Preserve:
  - dates
  - counts
  - titles
  - entity names
  - emails
  - contracts
  - board events

---

### VALID Fact_Type

[PARTY_INFO, CORPORATE_ROLE, CONDUCT, COMMUNICATION, DOCUMENT, CONTRACT_EVENT, FINANCIAL_EVENT, RELATED_PARTY_EVENT, BOARD_ACTION, RELATIONSHIP]

---

### FIELD RULES

Facts:
Only raw pre-dispute events.

Issue:
Neutral variable conflict only.

Rule:
General legal standard only.
No findings.

Application:
Neutral factual linkage only.
No inference.

Conclusion:
Only one of:
["Liable", "Not Liable", "Allowed", "Dismissed", "Unknown"]

Judicial_Reasoning_Log:
Store ALL:
- judicial findings
- reasoning
- verdict logic
- legal conclusions

---

### JSON SCHEMA:
{
  "results": [
    {
      "Metadata": {
        "Case_Number": "string",
        "Coram": "string",
        "Judge": "string",
        "Date": "YYYY-MM-DD",
        "Tribunal_Court": "string"
      },
      "Party_Details": {
        "Role": "Plaintiff | Defendant",
        "Name": "string",
        "Law_Firm": "string",
        "Counsel": ["string"],
        "Facts": [
          { "Fact_Type": "string", "Fact_Date": "YYYY-MM-DD", "Text": "string" }
        ],
        "Issue": "string",
        "Rule": "string",
        "Application": "string",
        "Conclusion": "string",
        "Judicial_Reasoning_Log": "string"
      }
    }
  ]
}

---

### FEW-SHOT ANCHOR (Transformation Example)
RAW TEXT: "Tan Tee Jim SC, Julian Tay and Jiang Ke-Yue (Lee & Lee) for the plaintiffs. In Suit 798/2007, the court found the defendant breached his duty by helping Huadian, which he admitted in cross-examination on 2009-03-12. Records show he joined the board on 1998-07-01." 

CLEAN EXTRACTION:
- Case_Number: "Suit 798/2007" 
- Law_Firm: "Lee & Lee" 
- Counsel: ["Tan Tee Jim SC", "Julian Tay", "Jiang Ke-Yue"] 
- Facts: [{"Fact_Type": "CORPORATE_ROLE", "Fact_Date": "1998-07-01", "Text": "The defendant was appointed as a director of the company."}]
- Issue: "Whether a Director's role requires disclosure of third-party business assistance."
- Conclusion: "Liable"
- Judicial_Reasoning_Log: "Defendant admitted to helping Huadian during cross-examination on 2009-03-12. Judge Prakash found this constituted a breach of fiduciary duty."

---

### JUDGMENT TEXT TO PROCESS:

"""

# =========================
# 7. PDF EXTRACTION
# =========================
def extract_pdf_text(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return " ".join(text.split())

# =========================
# 8. CLEAN OUTPUT ROW
# =========================
def clean_row(row):
    party = row.get("Party_Details", {})

    if not party.get("Name"):
        return None

    leakage_score = 0

    for field in ["Issue", "Rule", "Application"]:
        raw = party.get(field, "")
        leakage_score += contains_leakage(raw)
        party[field] = scrub_text(raw)

    clean_facts = []
    for fact in party.get("Facts", []):
        raw = fact.get("Text", "")
        leakage_score += contains_leakage(raw)

        txt = scrub_text(raw)
        if txt:
            fact["Text"] = txt
            clean_facts.append(fact)

    party["Facts"] = clean_facts

    row["Label"] = party.pop("Conclusion", "Unknown")
    row["Party_Details"] = party
    row["Leakage_Score"] = leakage_score

    return row

# =========================
# 9. MAIN GPT EXTRACTION
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
                    "content": "Return strict JSON only. No markdown. Follow schema exactly."
                },
                {
                    "role": "user",
                    "content": PROMPT_TEMPLATE + full_text[:100000]
                }
            ]
        )

        content = response.output_text
        data = safe_json_load(content)

        rows = data.get("results", [])

        cleaned = [clean_row(r) for r in rows]
        cleaned = [r for r in cleaned if r]

        return cleaned

    except Exception as e:
        print(f"Error processing {pdf_path.name}: {e}")
        return []

# =========================
# 10. BATCH RUNNER
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
            output_path = OUTPUT_FOLDER / f"{pdf_file.stem}.json"

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(rows, f, indent=2, ensure_ascii=False)

            print(f"Saved {len(rows)} rows -> {output_path}")
        else:
            print(f"No usable rows for {pdf_file.name}")

# =========================
# 11. ENTRY POINT
# =========================
if __name__ == "__main__":
    main()