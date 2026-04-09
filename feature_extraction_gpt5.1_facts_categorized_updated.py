import os
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import pdfplumber
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

# =========================
# 1. CONFIG & SYSTEM SETUP
# =========================

load_dotenv(find_dotenv())

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL_NAME = "gpt-4o"  # Fixed model name

INPUT_FOLDER = Path("Data/Testinput")
OUTPUT_FOLDER = Path("Data/Processed/testouput")
INPUT_FOLDER.mkdir(parents=True, exist_ok=True)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = (
    "Return only valid JSON with exactly two top-level arrays named "
    "'results' and 'counterclaims'. Do not add markdown fences or prose."
)

# =========================
# 2. HARD + SOFT LEAKAGE DETECTION & SCRUBBING
# =========================

BANNED_PATTERNS = [
    r"\bfiduciary\b",
    r"\bbreach(?:ed|es|ing)?\b",
    r"\bliable\b",
    r"\bwrongful(?:ly)?\b",
    r"\bdishonest(?:ly)?\b",
    r"\bbad faith\b",
    r"\b(?:the\s+)?(?:court|judge|tribunal|appellate\s+court|court\s+of\s+appeal|ca)\s+(?:held|found|concluded|ruled|observed|noted|decided|restructured)(?:\s+that)?\b",
    r"\b(?:upheld|affirmed|reversed|overturned|set\s+aside)\b",
    r"\balleged\b",
    r"\basserted\b",
    r"\bcross-?examination\b",
    r"\baffidavit\b",
    r"\btestified\b",
    r"\badmitted\b"
]

def contains_leakage(text: str) -> int:
    if not text:
        return 0
    score = 0
    for pattern in BANNED_PATTERNS:
        score += len(re.findall(pattern, text, flags=re.IGNORECASE))
    return score

def scrub_text(text: str) -> str:
    if not text:
        return ""
    
    # 1. Replace severe verdict/legal verbs with neutral synonyms
    replacements = [
        (r"\bbreach of\b", "contravention of"), # Catches grammatical awkwardness first
        (r"\bbreach(?:ed|es|ing)?\b", "contravened"),
        (r"\bliable\b", "legally responsible"),
        (r"\bwrongful(?:ly)?\b", "disputed"),
        (r"\bfiduciary\b", "management"),
        (r"\bdishonest(?:ly)?\b", "questionable"),
        (r"\bbad faith\b", "disputed motives"),
        (r"\b(?:the\s+)?(?:court|judge|tribunal|ca)\s+(?:held|found|concluded|ruled|observed|noted|decided)\b", "it was recorded"),
        (r"\balleged\b", "stated"),
        (r"\basserted\b", "stated")
    ]
    
    for pat, rep in replacements:
        text = re.sub(pat, rep, text, flags=re.IGNORECASE)
        
    # 2. Delete trial-evidence words completely
    deletions = [
        r"\baccording to .*? evidence\b",
        r"\bhe testified that\b",
        r"\bcross-?examination\b",
        r"\bthe witness stated\b",
        r"\badmitted\b",
        r"\baffidavit\b"
    ]
    
    for pat in deletions:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)
        
    # Clean up double spaces left by deletions
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# =========================
# 3. PROMPT TEMPLATE
# =========================

PROMPT_TEMPLATE = """
You are a Senior Legal Data Engineer specializing in Predictive Analytics. Extract predictive legal data into strict JSON.

### OBJECTIVE
Perform all tasks together in one pass:
1. Party-level fact extraction
2. Party-level outcome label assignment
3. Leakage control
4. Counterclaim handling

### STRICT PRE-DISPUTE RULE
- Stop extracting facts the moment the lawsuit is filed. Do NOT extract litigation procedure, statements of claim, counterclaims filed, or trial dates.
- Facts must be purely historical events leading up to the dispute.

### BANNED VOCABULARY
You must completely ban the following words from your vocabulary when writing the Issue, Rule, and Application:
'fiduciary', 'breach', 'breached', 'liable', 'alleged', 'asserted'. 
Use neutral synonyms like: 'management obligations', 'contravened', 'legally responsible', and 'stated'.

### EXTRACTION RULES
1. MULTI-PARTY:
   - Output one main-claim row per exact `(Case_Number, Party, Role)` combination under `results`.
   - CRITICAL: You MUST extract at least one Plaintiff row AND at least one Defendant row for EVERY Suit or Appeal listed in the document.
   - Ignore Third Party claims completely.
   - CRITICAL: You MUST extract a distinct Plaintiff and Defendant row for EVERY SINGLE plaintiff and defendant named in the main claim, even if they also appear in a counterclaim.
2. HIGH-DENSITY FACTS:
   - Extract observable data: dates, names, communications, roles, documents, payments, contracts.
   - FACT DATING RULES: 
    1. Extract the specific ISO date (YYYY-MM-DD). 
    2. PARTIAL DATES: If only a month/year, use YYYY-MM-01. If only a year, use YYYY-01-01.
    3. DATE RANGES: Use the START date.
   - CRITICAL: You must extract at least 4 to 8 highly specific facts per party. Do not be lazy.
3. LEAKAGE CONTROL & FIELDS:
   - Issue: MUST be a hypothetical legal question starting with 'Whether...'. STRICTLY FORBIDDEN to use the court's final conclusions.
   - Rule: General legal standard only. No findings.
   - Application: STRICTLY MAXIMUM 3 SENTENCES summarizing the competing arguments. Do NOT include what the Court ultimately decided.
   - Judicial_Reasoning_Log: Store all judicial findings, merits, and verdicts here.
4. LABEL ASSIGNMENT:
   - Plaintiff (including Appellants): `Claim Allowed`, `Claim Dismissed`, `Claim Allowed in Part`, `Appeal Allowed`, `Appeal Dismissed`, `Appeal Allowed in Part`, `Unknown`.
   - Defendant (including Respondents): `Liable`, `Not Liable`, `Partially Liable`, `Appeal Allowed`, `Appeal Dismissed`, `Appeal Allowed in Part`, `Unknown`.
5. COUNTERCLAIMS (CRITICAL RULE):
   - A Counterclaim is treated as an entirely separate lawsuit.
   - YOU MUST EXTRACT BOTH A PLAINTIFF ROW AND A DEFENDANT ROW FOR THE COUNTERCLAIM. 
   - CRITICAL: You MUST check the title page of the document. If it lists a "Plaintiff by counterclaim" and "Defendant(s) by counterclaim", you MUST extract them.
   - Even if the Defendant by Counterclaim is the same entity from the main claim, you MUST generate a separate, complete JSON object for them under the counterclaims array. Do not skip them!
   - CRITICAL FORMATTING: The `Case_Number` MUST exactly match the main case number, followed by exactly ` (Counterclaim)`. Example: `Suit No 418 of 2018 (Counterclaim)`.

VALID Fact_Type values:
[PARTY_INFO, CHRONOLOGY, CORPORATE_ROLE, CONDUCT, COMMUNICATION, DOCUMENT,
 CONTRACT_EVENT, FINANCIAL_EVENT, RELATED_PARTY_EVENT, AUTHORITY_EVENT,
 DISCLOSURE_EVENT, BOARD_ACTION, RELATIONSHIP]

### JSON SCHEMA
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
        "Application": "string (MAX 3 SENTENCES)",
        "Conclusion": "string",
        "Judicial_Reasoning_Log": "string"
      }
    }
  ],
  "counterclaims": [
    // EXACT SAME SCHEMA AS "results"
  ]
}

### FEW-SHOT ANCHOR (COUNTERCLAIM)
RAW TEXT SCENARIO:
"John Smith ... Plaintiff by counterclaim against Acme Corp ... Defendant by counterclaim. Smith stated Acme failed to pay $500k under an oral advisory agreement formed in 2012. Acme argued the formal 2011 written contract governed. The judge ultimately dismissed the counterclaim, finding no oral agreement existed."

CLEAN EXTRACTION:
{
  "counterclaims": [
    {
      "Metadata": {
        "Case_Number": "Suit 999 of 2018 (Counterclaim)",
        "Coram": "Jane Doe J",
        "Judge": "Jane Doe J",
        "Date": "2019-08-12",
        "Tribunal_Court": "High Court of Singapore"
      },
      "Party_Details": {
        "Role": "Plaintiff",
        "Name": "John Smith",
        "Law_Firm": "Unknown",
        "Counsel": [],
        "Facts": [
          {
            "Fact_Type": "CONTRACT_EVENT",
            "Fact_Date": "2011-01-01",
            "Text": "John Smith and Acme Corp executed a formal written advisory contract."
          },
          {
            "Fact_Type": "COMMUNICATION",
            "Fact_Date": "2012-01-01",
            "Text": "John Smith communicated with Acme Corp regarding advisory services and requested a $500,000 fee."
          }
        ],
        "Issue": "Whether an advisor can enforce a claim for unpaid fees based on a disputed oral variation of a written contract.",
        "Rule": "A party claiming a contravention of an oral variation must prove the exact terms of the variation and mutual consent to displace a prior written agreement.",
        "Application": "The Plaintiff by counterclaim stated that a subsequent oral agreement modified the payment terms, entitling him to immediate payment. The Defendant by counterclaim argued that the original written contract remained exclusively in force.",
        "Conclusion": "Claim Dismissed",
        "Judicial_Reasoning_Log": "The court held that the evidence did not support the existence of a binding oral variation. Because the conditions precedent in the original written contract were not met, the claim for unpaid fees was dismissed."
      }
    },
    {
      "Metadata": {
        "Case_Number": "Suit 999 of 2018 (Counterclaim)",
        "Coram": "Jane Doe J",
        "Judge": "Jane Doe J",
        "Date": "2019-08-12",
        "Tribunal_Court": "High Court of Singapore"
      },
      "Party_Details": {
        "Role": "Defendant",
        "Name": "Acme Corp",
        "Law_Firm": "Unknown",
        "Counsel": [],
        "Facts": [
          {
            "Fact_Type": "CONTRACT_EVENT",
            "Fact_Date": "2011-01-01",
            "Text": "Acme Corp executed a formal written advisory contract with John Smith."
          }
        ],
        "Issue": "Whether a company is legally responsible to pay advisory fees based on a disputed oral modification to a strict written contract.",
        "Rule": "A corporate entity is bound only by the formal terms of a written agreement unless mutual consent for an oral variation is definitively proven.",
        "Application": "The Defendant by counterclaim argued that the strict conditions of the 2011 written contract were not met, negating any payment obligation. The Plaintiff by counterclaim stated that the company was bound by a 2012 oral variation.",
        "Conclusion": "Not Liable",
        "Judicial_Reasoning_Log": "The court sided with the company, finding no oral agreement existed to override the written contract."
      }
    }
  ]
}

### JUDGMENT TEXT TO PROCESS:
"""

# =========================
# 4. NORMALIZATION HELPERS
# =========================

PLAINTIFF_LABEL_MAP = {
    "claim allowed": "Claim Allowed",
    "allowed": "Claim Allowed",
    "claim dismissed": "Claim Dismissed",
    "dismissed": "Claim Dismissed",
    "claim allowed in part": "Claim Allowed in Part",
    "allowed in part": "Claim Allowed in Part",
    "partially allowed": "Claim Allowed in Part",
    "appeal allowed": "Appeal Allowed",
    "appeal dismissed": "Appeal Dismissed",
    "appeal allowed in part": "Appeal Allowed in Part",
    "unknown": "Unknown",
}

DEFENDANT_LABEL_MAP = {
    "liable": "Liable",
    "not liable": "Not Liable",
    "partially liable": "Liable",
    "liable in part": "Liable",
    "allowed": "Liable",
    "claim allowed": "Liable",
    "dismissed": "Not Liable",
    "claim dismissed": "Not Liable",
    "allowed in part": "Liable",
    "claim allowed in part": "Liable",
    "appeal allowed": "Appeal Allowed",
    "appeal dismissed": "Appeal Dismissed",
    "appeal allowed in part": "Appeal Allowed in Part",
    "unknown": "Unknown",
}

def safe_json_load(content: str) -> Dict[str, Any]:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def normalize_role(value: Any) -> str:
    role = str(value or "").strip().title()
    if role in ["Plaintiff", "Appellant"]:
        return "Plaintiff"
    if role in ["Defendant", "Respondent"]:
        return "Defendant"
    return ""


def normalize_main_claim_label(role: str, value: Any) -> str:
    normalized = str(value or "Unknown").strip().lower()
    if role == "Plaintiff":
        return PLAINTIFF_LABEL_MAP.get(normalized, "Unknown")
    if role == "Defendant":
        return DEFENDANT_LABEL_MAP.get(normalized, "Unknown")
    return "Unknown"


def clean_row(row: Dict[str, Any]) -> Dict[str, Any]:
    party = row.get("Party_Details", {})
    
    if not party.get("Name"):
        return {}

    leakage_score = 0

    # 1. Scrub textual fields & calculate score ON THE SCRUBBED TEXT
    for field in ["Issue", "Rule", "Application"]:
        raw = party.get(field, "")
        scrubbed = scrub_text(raw)
        
        leakage_score += contains_leakage(scrubbed)
        party[field] = scrubbed

    # 2. Scrub facts
    clean_facts = []
    for fact in party.get("Facts", []):
        raw = fact.get("Text", "")
        scrubbed = scrub_text(raw)
        
        leakage_score += contains_leakage(scrubbed)
        
        if scrubbed:
            fact["Text"] = scrubbed
            clean_facts.append(fact)

    party["Facts"] = clean_facts

    # 3. Normalize Label
    role = normalize_role(party.get("Role", ""))
    party["Role"] = role
    row["Label"] = normalize_main_claim_label(role, party.pop("Conclusion", "Unknown"))
    
    row["Party_Details"] = party
    row["Leakage_Score"] = leakage_score

    return row


# =========================
# 5. EXECUTION PIPELINE
# =========================

def process_pdf(pdf_path: Path) -> List[Dict[str, Any]]:
    with pdfplumber.open(pdf_path) as pdf:
        full_text = "\n".join(page.extract_text() or "" for page in pdf.pages)

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": PROMPT_TEMPLATE + full_text[:100000]},
            ],
        )

        content = response.choices[0].message.content or "{}"
        data = safe_json_load(content)

        raw_results = data.get("results", []) or []
        raw_counterclaims = data.get("counterclaims", []) or []

        if isinstance(raw_results, dict):
            raw_results = [raw_results]
        if isinstance(raw_counterclaims, dict):
            raw_counterclaims = [raw_counterclaims]

        # BECAUSE RESULTS AND COUNTERCLAIMS USE THE EXACT SAME SCHEMA, WE PASS BOTH THROUGH CLEAN_ROW!
        processed_rows = [clean_row(row) for row in raw_results if isinstance(row, dict)]
        processed_counterclaims = [clean_row(row) for row in raw_counterclaims if isinstance(row, dict)]

        # Filter out empty rows and combine them into a single flat list
        flat_output = [r for r in (processed_rows + processed_counterclaims) if r]
        
        return flat_output

    except Exception as exc:
        print(f"Error processing {pdf_path.name}: {exc}")
        return []


def main() -> None:
    pdf_files = list(INPUT_FOLDER.glob("*.pdf"))
    if not pdf_files:
        print("No PDF files found.")
        return

    for pdf_file in pdf_files:
        print(f"--- Extracting: {pdf_file.name} ---")
        flat_output = process_pdf(pdf_file)

        if flat_output:
            output_path = OUTPUT_FOLDER / f"{pdf_file.stem}.json"
            with open(output_path, "w", encoding="utf-8") as handle:
                json.dump(flat_output, handle, indent=2, ensure_ascii=False)
            print(f"Success: Generated {len(flat_output)} total rows.")


if __name__ == "__main__":
    main()