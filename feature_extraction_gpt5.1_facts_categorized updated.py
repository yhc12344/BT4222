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

# find_dotenv() will automatically scan up through parent folders to find the .env file
load_dotenv(find_dotenv())

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL_NAME = "gpt-5.1"  # Updated to a valid OpenAI model

INPUT_FOLDER = Path("Data/Testinput")
OUTPUT_FOLDER = Path("Data/Processed/testouput")
INPUT_FOLDER.mkdir(parents=True, exist_ok=True)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = (
    "Return only valid JSON with exactly two top-level arrays named "
    "'results' and 'counterclaims'. Do not add markdown fences or prose."
)

# =========================
# 2. FEATURE SANITIZATION
# =========================

FEATURE_REWRITE_RULES = [
    (r"\b(?:the\s+)?(?:court|judge|tribunal)\s+(?:held|found|concluded|ruled|observed|noted)(?:\s+that)?\b", ""),
    (r"\bfiduciary duties\b", "management obligations"),
    (r"\bfiduciary duty\b", "management obligation"),
    (r"\bfiduciary position\b", "senior management role"),
    (r"\boccupied a fiduciary position\b", "held a corporate management role"),
    (r"\bbreached his duties\b", "engaged in disputed conduct"),
    (r"\bbreached her duties\b", "engaged in disputed conduct"),
    (r"\bbreached their duties\b", "engaged in disputed conduct"),
    (r"\bbreach of fiduciary duty\b", "disputed management conduct"),
    (r"\bwrongfully\b", ""),
    (r"\bdishonest(?:ly)?\b", "disputed"),
    (r"\bbad faith\b", "disputed motive"),
    (r"\bliable\b", "subject of dispute"),
    (r"\bowed\b", "had"),
    (r"\balleged that\b", "stated that"),
    (r"\basserted that\b", "stated that"),
    (r"\bcross-?examination\b", ""),
    (r"\baffidavit\b", ""),
    (r"\btestified\b", "stated"),
    (r"\badmitted\b", "stated"),
]

FEATURE_LEAKAGE_PATTERNS = [
    r"\b(?:the\s+)?(?:court|judge|tribunal)\s+(?:held|found|concluded|ruled|observed|noted)\b",
    r"\bcross-?examination\b",
    r"\baffidavit\b",
    r"\btestified\b",
    r"\badmitted\b",
    r"\bbreach(?:ed)?\b",
    r"\bfiduciary\b",
    r"\bliable\b",
    r"\bwrongful(?:ly)?\b",
]


def sanitize_feature_text(text: Any) -> str:
    if not text:
        return ""

    cleaned = str(text)
    for pattern, replacement in FEATURE_REWRITE_RULES:
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)

    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)
    return cleaned.strip(" ,;:-")


def count_feature_leakage(*texts: str) -> int:
    score = 0
    for text in texts:
        if not text:
            continue
        for pattern in FEATURE_LEAKAGE_PATTERNS:
            score += len(re.findall(pattern, text, flags=re.IGNORECASE))
    return score


# =========================
# 3. PROMPT TEMPLATE
# =========================

PROMPT_TEMPLATE = """
You are a Senior Legal Data Engineer. Extract predictive legal data into strict JSON.

### OBJECTIVE
Perform all four tasks together in one pass:
1. party-level fact extraction
2. party-level outcome label assignment
3. leakage control
4. counterclaim handling

### MULTI-SUIT CONTRACT
- Output one main-claim row per exact `(Case_Number, Party, Role)` combination under `results`.
- If a judgment covers multiple suits or appeals, split them. Never merge multiple cases into one `Case_Number` row.
- CRITICAL: Use the exact primary identifier for that row at the top of the judgment (e.g., `Suit No 418 of 2018`). If the document is an appellate judgment, you MUST extract the Appeal cases and you MUST format the `Case_Number` to include both the appeal number and the underlying suit number, like this: `Civil Appeal No 86 of 2017 (Suit No 1098 of 2013)`.
- The same named person may appear multiple times across different suits/appeals and/or roles.
- `results` is ONLY for the main claim of each case.
- If the judgment contains a counterclaim, put it under `counterclaims` only.
- Do not merge counterclaim roles into the main-claim rows.

### EXTRACTION RULES
1. MULTI-PARTY:
   - Return one JSON object per unique party-role within the main claim of each suit or appeal under `results`.
   - CRITICAL: You must extract a row for EVERY named party in the heading (including nominal defendants), even if their role is passive. 
   - FOR SIMPLICITY: Classify all Appellants as 'Plaintiff' and all Respondents as 'Defendant'. 
   - Do NOT extract rows for Third Parties. Completely ignore Third Party claims.
   - If the judgment contains a counterclaim / plaintiff by counterclaim / defendant by counterclaim, also return one compact object per counterclaim unit under `counterclaims`.
2. HIGH-DENSITY FACTS:
   - Do not summarize facts into broad conclusions.
   - Extract observable data: dates, names, communications, roles, documents, payments, contracts, board actions, and relationship events.
   - Facts must be concrete and tied to identifiable conduct or documents.
   - If a party is a nominal defendant with no active independent conduct, it is acceptable to leave their `Facts` array empty or include only a single `PARTY_INFO` fact.
3. LEAKAGE CONTROL:
   - Facts, Issue, Rule, and Application must stay neutral and must not contain judicial findings, post-trial conclusions, or verdict wording.
   - Do not use trial-only knowledge such as cross-examination admissions, credibility findings, or later evidential disputes as Facts.
   - `Application` must summarize how the parties ARGUED the rules apply to the facts (e.g., "The Plaintiff argued that...", "The Defendant relied on..."). Do NOT include the judge's final decision (e.g., "The court found") in this field.
   - Put judicial findings, merits reasoning, and final outcome logic into `Judicial_Reasoning_Log` only.
4. LABEL ASSIGNMENT:
   - For Plaintiff rows (including Appellants), `Conclusion` must be one of: `Claim Allowed`, `Claim Dismissed`, `Claim Allowed in Part`, `Appeal Allowed`, `Appeal Dismissed`, `Appeal Allowed in Part`, `Unknown`.
   - For Defendant rows (including Respondents), `Conclusion` must be one of: `Liable`, `Not Liable`, `Partially Liable`, `Appeal Allowed`, `Appeal Dismissed`, `Appeal Allowed in Part`, `Unknown`.
   - If a party is merely a nominal defendant, label their `Conclusion` as `Unknown`.
5. COUNTERCLAIMS:
   - Create `counterclaims` entries if the judgment expressly contains a counterclaim OR if an appellate judgment mentions a counterclaim that was decided by the lower court in its background facts.
   - CRITICAL FORMATTING: The `Case_Number` for any counterclaim MUST exactly match the main case number, followed by exactly the string ` (Counterclaim)`. Do NOT add any party names or other text inside the brackets. Example: `Suit No 418 of 2018 (Counterclaim)`.
   - Keep counterclaims compact.
   - Use exactly this compact schema:
     {
       "Case_Number": "string",
       "Plaintiff": "string",
       "Defendant": "string",
       "Plaintiff_Label": "string",
       "Defendant_Label": "string"
     }

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
          {
            "Fact_Type": "string",
            "Fact_Date": "YYYY-MM-DD | YYYY-MM | YYYY | ''",
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
  ],
  "counterclaims": [
    {
      "Case_Number": "string",
      "Plaintiff": "string",
      "Defendant": "string",
      "Plaintiff_Label": "Claim Allowed | Claim Dismissed | Claim Allowed in Part | Appeal Allowed | Appeal Dismissed | Appeal Allowed in Part | Unknown",
      "Defendant_Label": "Liable | Not Liable | Partially Liable | Appeal Allowed | Appeal Dismissed | Appeal Allowed in Part | Unknown"
    }
  ]
}

### FEW-SHOT ANCHOR (COUNTERCLAIM)
If the judgment says:
- `Danial Patrick Higgins ... Plaintiff by counterclaim`
- `(1) Philippe Emanuel Mulacek ... (6) Singapore Air Charter Pte Ltd ... Defendants by counterclaim`

then return a compact row like:
{
  "counterclaims": [
    {
      "Case_Number": "Suit 733 of 2014 (Counterclaim)",
      "Plaintiff": "Danial Patrick Higgins",
      "Defendant": "Philippe Emanuel Mulacek, Carlo Giuseppe Civelli, Nicholas Johnstone, Daniel Chance Walker, Stefan Wood, Singapore Air Charter Pte Ltd",
      "Plaintiff_Label": "Claim Allowed",
      "Defendant_Label": "Liable"
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
    "partially liable": "Partially Liable",
    "liable in part": "Partially Liable",
    "allowed": "Liable",
    "claim allowed": "Liable",
    "dismissed": "Not Liable",
    "claim dismissed": "Not Liable",
    "allowed in part": "Partially Liable",
    "claim allowed in part": "Partially Liable",
    "appeal allowed": "Appeal Allowed",
    "appeal dismissed": "Appeal Dismissed",
    "appeal allowed in part": "Appeal Allowed in Part",
    "unknown": "Unknown",
}

VALID_COUNTERCLAIM_PLAINTIFF_LABELS = {
    "Claim Allowed", "Claim Dismissed", "Claim Allowed in Part", 
    "Appeal Allowed", "Appeal Dismissed", "Appeal Allowed in Part", "Unknown",
}
VALID_COUNTERCLAIM_DEFENDANT_LABELS = {
    "Liable", "Not Liable", "Partially Liable", 
    "Appeal Allowed", "Appeal Dismissed", "Appeal Allowed in Part", "Unknown",
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
    # Safety net: Automatically map Appellant/Respondent back to Plaintiff/Defendant if the LLM slips up
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


def normalize_fact(fact: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "Fact_Type": str(fact.get("Fact_Type", "") or "").strip(),
        "Fact_Date": str(fact.get("Fact_Date", "") or "").strip(),
        "Text": sanitize_feature_text(fact.get("Text", "")),
    }


def normalize_result_row(row: Dict[str, Any]) -> Dict[str, Any]:
    metadata = row.get("Metadata", {}) or {}
    party = dict(row.get("Party_Details", {}) or {})

    role = normalize_role(party.get("Role", ""))
    facts = []
    for fact in party.get("Facts", []) or []:
        if not isinstance(fact, dict):
            continue
        normalized_fact = normalize_fact(fact)
        if normalized_fact["Text"]:
            facts.append(normalized_fact)

    issue = sanitize_feature_text(party.get("Issue", ""))
    rule = sanitize_feature_text(party.get("Rule", ""))
    application = sanitize_feature_text(party.get("Application", ""))
    reasoning = str(party.get("Judicial_Reasoning_Log", "") or "").strip()

    normalized_party = {
        "Role": role,
        "Name": str(party.get("Name", "") or "").strip(),
        "Law_Firm": str(party.get("Law_Firm", "") or "").strip(),
        "Counsel": [str(item).strip() for item in (party.get("Counsel", []) or []) if str(item).strip()],
        "Facts": facts,
        "Issue": issue,
        "Rule": rule,
        "Application": application,
        "Judicial_Reasoning_Log": reasoning,
    }

    leakage_score = count_feature_leakage(
        issue,
        rule,
        application,
        *(fact["Text"] for fact in facts),
    )

    return {
        "Metadata": {
            "Case_Number": str(metadata.get("Case_Number", "") or "").strip(),
            "Coram": str(metadata.get("Coram", "") or "").strip(),
            "Judge": str(metadata.get("Judge", "") or "").strip(),
            "Date": str(metadata.get("Date", "") or "").strip(),
            "Tribunal_Court": str(metadata.get("Tribunal_Court", "") or "").strip(),
        },
        "Party_Details": normalized_party,
        "Label": normalize_main_claim_label(role, party.get("Conclusion", "Unknown")),
        "Leakage_Score": leakage_score,
    }


def normalize_counterclaim(entry: Dict[str, Any]) -> Dict[str, Any]:
    plaintiff_label = str(entry.get("Plaintiff_Label", "Unknown") or "Unknown").strip()
    defendant_label = str(entry.get("Defendant_Label", "Unknown") or "Unknown").strip()

    if plaintiff_label not in VALID_COUNTERCLAIM_PLAINTIFF_LABELS:
        plaintiff_label = "Unknown"
    if defendant_label not in VALID_COUNTERCLAIM_DEFENDANT_LABELS:
        defendant_label = "Unknown"

    return {
        "Case_Number": str(entry.get("Case_Number", "") or "").strip(),
        "Plaintiff": str(entry.get("Plaintiff", "") or "").strip(),
        "Defendant": str(entry.get("Defendant", "") or "").strip(),
        "Plaintiff_Label": plaintiff_label,
        "Defendant_Label": defendant_label,
    }


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
                {"role": "user", "content": PROMPT_TEMPLATE + full_text},
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

        processed_rows = [normalize_result_row(row) for row in raw_results if isinstance(row, dict)]
        processed_counterclaims = [
            normalize_counterclaim(entry) for entry in raw_counterclaims if isinstance(entry, dict)
        ]

        # flat list, with compact counterclaim row appended at the bottom
        return processed_rows + processed_counterclaims

    except Exception as exc:
        print(f"Error processing {pdf_path.name}: {exc}")
        return []


def main() -> None:
    pdf_files = list(INPUT_FOLDER.glob("*.pdf"))
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