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

INPUT_FOLDER = Path("Data/Test")
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
        "Judge": "string",
        "Date": "YYYY-MM-DD",
        "Tribunal_Court": "string"
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

RAW JUDGMENT TEXT: 
"The court found that the defendant, as a director, breached his fiduciary duty by failing to disclose he was helping a competitor, Huadian."

CLEAN EXTRACTION:
- Facts: [{"Fact_Type": "CORPORATE_ROLE", "Text": "The defendant held the position of Director from 1998 to 2003."}]
- Issue: "Whether a Director's role requires disclosure of third-party business assistance."
- Rule: "Corporate directors are subject to standards of loyalty regarding competitive activity."
- Application: "The defendant facilitated contact between an expert and a third-party entity while serving as Director."
- Conclusion: "Liable"
- Judicial_Reasoning_Log: "Judge Prakash found the fiduciary duty was breached because the assistance created a conflict with the director's role."

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
        print("No PDF files found in Data/Test")
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