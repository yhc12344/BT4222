import os
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
import pdfplumber
from openai import OpenAI

# =========================
# 1. CONFIG
# =========================
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL_NAME = "gpt-5.4-mini"
INPUT_FOLDER = Path("Data/Processed/testouput")
OUTPUT_FOLDER = Path("Data/Processed/UpdatedFinalAuditedChecked")

INPUT_FOLDER.mkdir(parents=True, exist_ok=True)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# =========================
# 2. LABEL + FACT TYPE RULES
# =========================
PLAINTIFF_LABELS = {"Claim Allowed", "Claim Allowed in Part", "Claim Dismissed", "Unknown"}
DEFENDANT_LABELS = {"Liable", "Not Liable", "Unknown"}
GENERIC_LABELS = {"Unknown"}

ALLOWED_FACT_TYPES = {
    "PARTY_INFO",
    "CORPORATE_ROLE",
    "CONDUCT",
    "COMMUNICATION",
    "DOCUMENT",
    "CONTRACT_EVENT",
    "FINANCIAL_EVENT",
    "RELATED_PARTY_EVENT",
    "BOARD_ACTION",
    "RELATIONSHIP",
    "AUTHORITY_EVENT",
    "DISCLOSURE_EVENT",
    "CHRONOLOGY",
}

FACT_TYPE_ALIASES = {
    "PARTYINFO": "PARTY_INFO",
    "PARTY_INFO": "PARTY_INFO",
    "CORPORATEROLE": "CORPORATE_ROLE",
    "CORPORATE_ROLE": "CORPORATE_ROLE",
    "CONTRACTEVENT": "CONTRACT_EVENT",
    "CONTRACT_EVENT": "CONTRACT_EVENT",
    "FINANCIALEVENT": "FINANCIAL_EVENT",
    "FINANCIAL_EVENT": "FINANCIAL_EVENT",
    "RELATEDPARTYEVENT": "RELATED_PARTY_EVENT",
    "RELATED_PARTY_EVENT": "RELATED_PARTY_EVENT",
    "BOARDACTION": "BOARD_ACTION",
    "BOARD_ACTION": "BOARD_ACTION",
    "AUTHORITYEVENT": "AUTHORITY_EVENT",
    "AUTHORITY_EVENT": "AUTHORITY_EVENT",
    "DISCLOSUREEVENT": "DISCLOSURE_EVENT",
    "DISCLOSURE_EVENT": "DISCLOSURE_EVENT",
}

PROTECTED_PATHS = {
    "Metadata.Case_Number",
}

MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "sept": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

MONTH_PATTERN = (
    r"(?:January|February|March|April|May|June|July|August|September|October|"
    r"November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
)

DAY_MONTH_YEAR_RE = re.compile(
    rf"\b(\d{{1,2}})\s+({MONTH_PATTERN})\s+(\d{{4}})\b",
    re.IGNORECASE,
)
MONTH_DAY_YEAR_RE = re.compile(
    rf"\b({MONTH_PATTERN})\s+(\d{{1,2}}),\s*(\d{{4}})\b",
    re.IGNORECASE,
)
MONTH_YEAR_RE = re.compile(
    rf"\b({MONTH_PATTERN})\s+(\d{{4}})\b",
    re.IGNORECASE,
)

ISO_DATE_RE = re.compile(r"\b(19\d{2}|20\d{2})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b")
ISO_MONTH_RE = re.compile(r"\b(19\d{2}|20\d{2})-(0[1-9]|1[0-2])\b")
YEAR_RE = re.compile(r"(?<![-\d])(19\d{2}|20\d{2})(?![-\d])")

HEADER_COUNSEL_SECTION_RE = re.compile(
    r"Counsel Name\(s\)\s*:\s*(?P<section>.+?)\s*Parties\s*:",
    re.IGNORECASE | re.DOTALL,
)

HEADER_COUNSEL_BLOCK_RE = re.compile(
    r"(?P<names>.+?)\s*\((?P<firm>[^()]+)\)\s*for the\s+(?P<role_desc>.+)$",
    re.IGNORECASE,
)

TRAILING_COUNSEL_RE = re.compile(
    r"(?P<names>[A-Z][^.;]{0,700}?)\((?P<firm>[^()]+)\)\s*for the\s+"
    r"(?P<role_desc>plaintiff(?: and third parties)?|defendants?|defendant|third parties?)",
    re.IGNORECASE,
)

SUIT_ROLE_ENTRY_RE = re.compile(
    r"(?:the\s+)?(?P<side>plaintiff(?: and third parties)?|defendants?|defendant|third parties?)"
    r"(?:\s+in\s+(?P<case_no>Suit No \d+ of \d{4}))?",
    re.IGNORECASE,
)

APPEAL_ROLE_ENTRY_RE = re.compile(
    r"(?:the\s+)?"
    r"(?:(?P<ordinals>\d+(?:st|nd|rd|th)"
    r"(?:\s*,\s*\d+(?:st|nd|rd|th))*"
    r"(?:\s+and\s+\d+(?:st|nd|rd|th))?)\s+)?"
    r"(?P<side>appellant|respondent)s?\s+in\s+"
    r"(?P<appeal_no>Civil Appeal No \d+ of \d{4})",
    re.IGNORECASE,
)

ORDINAL_RE = re.compile(r"(\d+)(?:st|nd|rd|th)", re.IGNORECASE)

FILED_APPEAL_RE = re.compile(
    r"([A-Z][A-Za-z0-9&.,'()\- ]+?)\s+filed\s+(Civil Appeal No \d+ of \d{4})",
    re.IGNORECASE,
)

CROSS_APPEAL_RE = re.compile(
    r"([A-Z][A-Za-z0-9&.,'()\- ]+?)\s+brought\s+a\s+cross\s+appeal,\s+namely\s+"
    r"(Civil Appeal No \d+ of \d{4})",
    re.IGNORECASE,
)

# =========================
# 3. AUDIT PROMPTS
# =========================
SYSTEM_PROMPT = """
You are an expert legal ML data auditor.

Audit exactly one party row against the source judgment text. Your two goals are:
1. FACTUAL ACCURACY
2. TEMPORAL WALL SAFETY

Key principles:
- Be conservative.
- Prefer no change over speculation.
- Prefer rewriting over removal when a fact contains salvageable substance.
- Do not change Metadata.Case_Number.
- Never recommend role names such as Plaintiff or Defendant as labels.
- Every kept fact must be specifically supported by the source for this row's party.
- Every kept fact text must be a complete, grammatical sentence.

ROW SCOPE
- Audit only the provided row.
- Do not merge outcomes, facts, or party positions across different rows.
- If the row is a counterclaim row, assess facts and outcome only in relation to that counterclaim row.
- If the row is an appellate row, assess facts and outcome using the specific appeal number in that row.
- Do not assume a party has the same role or outcome across linked appeals, cross-appeals, or counterclaims.

TEMPORAL WALL
Definition:
- The Temporal Wall is the point when the dispute clearly crystallized between the parties, and in all events no later than commencement of legal proceedings.

How to classify information:
- SAFE: pre-dispute business events, contracts, communications, transactions, appointments, internal decisions, and dispute-forming events.
- BORDERLINE: demand letters, pre-suit negotiations, accusations, default notices, and other events showing the dispute emerging. Keep or rewrite if they describe a substantive event rather than court procedure.
- LEAKAGE: pleadings, filings, affidavits, witness testimony, cross-examination, submissions, hearings, court reasoning, judicial findings, appeals, costs awards, and judgment-stage outcomes.

REWRITE VS REMOVAL
- Prefer rewriting if a sentence contains a removable legal wrapper around a usable event.
- Remove only if the fact is mainly procedural, trial-stage, or judgment-stage and cannot be salvaged.
- Keep the underlying conduct only if it is explicitly supported by the source.

Examples:
- CURRENT: "The court found that the defendant breached fiduciary duties by pledging company assets."
  BETTER: "The defendant pledged company assets as security for personal loans." only if explicitly supported.
- CURRENT: "The plaintiff alleged that payment was not made by 30 March 2001."
  BETTER: "Payment due by 30 March 2001 was not made." only if explicitly supported.
- CURRENT: "The court held that the claim failed because there was no collateral agreement."
  BETTER: remove unless the source explicitly supports a usable non-judicial fact such as "The written agreement contained the payment schedule and no separate collateral agreement was documented."
- CURRENT: "The judge found that the director acted against the company's interests."
  BETTER: remove judicial language and keep only the underlying supported conduct, if any.

PARTY-SPECIFIC SUPPORT
- Every kept fact must be attributable to the row's party.
- Do not assign a fact to a party merely because that party appears in the same paragraph.
- If the paragraph describes conduct by another party, do not convert it into this row's fact.
- If attribution is unclear, prefer no change or remove the fact.

FIELDS
- Facts: only supported substantive events.
- Issue, Rule, Application: must be neutral and free of hindsight phrasing such as "the court found" or "the judge held".
- Judicial_Reasoning_Log may contain court reasoning and outcomes.

FACT TEXT QUALITY
- Every fact text must be a complete, grammatical sentence.
- No clipped fragments.
- No malformed rewrites.
- No mixed or corrupted dates/text.
- If a rewrite becomes fragmentary or ungrammatical, rewrite it cleanly from the source or remove it.

DATE RULES
Always prefer a full ISO date YYYY-MM-DD when support exists.

Fact_Date decision order:
1. Exact date explicitly supported by the source for this event -> use it.
2. Partial date explicitly supported by the source -> normalize:
   - June 1991 -> 1991-06-01
   - 1991 -> 1991-01-01
3. Date range -> choose the earliest point in the range:
   - June 1991 to April 2000 -> 1991-06-01
   - 1991 to 2000 -> 1991-01-01
4. If the fact text is vague, infer conservatively from the supporting PDF context for the same event.
5. If support is still insufficient, keep a supported existing ISO date if present; otherwise leave blank.

LINKED-EVENT DATE RULE
- If one sentence or paragraph mentions two or more different events with different dates, do not assign one date to all of them.
- Use the date for the specific event being kept.
- If necessary, split the content into separate facts.

Examples:
- "Settlement Agreement dated 12 July 2010; employment terminated with effect from 25 July 2010"
  -> agreement fact gets 2010-07-12
  -> termination fact gets 2010-07-25

If sources conflict:
- explicit full date > partial date
- explicit source support > vague inference
- if a supported range is given, use the earliest supported date in that range
- do not invent a day unless the source supports the day

LABEL RULES
- Party_Details.Role is metadata, not the label.
- Plaintiff labels: ["Claim Allowed", "Claim Allowed in Part", "Claim Dismissed", "Unknown"]
- Defendant labels: ["Liable", "Not Liable", "Unknown"]
- Third-party or procedural-only rows: use "Unknown" unless the judgment clearly determines a substantive outcome.
- For appellate rows, apply these same labels using the outcome of the specific appeal number in the row.
- Do not merge outcomes across different appeal numbers.

FACT TYPE RULES
Use only these Fact_Type values:
[PARTY_INFO, CORPORATE_ROLE, CONDUCT, COMMUNICATION, DOCUMENT, CONTRACT_EVENT, FINANCIAL_EVENT, RELATED_PARTY_EVENT, BOARD_ACTION, RELATIONSHIP, AUTHORITY_EVENT, DISCLOSURE_EVENT, CHRONOLOGY]
Do not invent new Fact_Type names.

Return only valid JSON with this exact shape:
{
  "case_name": "string",
  "party": "string",
  "status": "CLEAN" | "CHANGES",
  "leakage_score": 0,
  "label_change": null | {
    "current": "string",
    "recommended": "string",
    "evidence": "string",
    "recommended_change": "string"
  },
  "fact_changes": [
    {
      "path": "string",
      "issue_type": "ACCURACY_ERROR" | "TEMPORAL_WALL_LEAKAGE",
      "current_value": "string",
      "recommended_value": "string",
      "evidence": "string",
      "recommended_change": "string"
    }
  ],
  "facts_to_remove": [
    {
      "path": "string",
      "issue_type": "ACCURACY_ERROR" | "TEMPORAL_WALL_LEAKAGE",
      "current_value": "string",
      "evidence": "string",
      "recommended_change": "string"
    }
  ]
}
""".strip()

USER_PROMPT_TEMPLATE = """
Audit the following party row against the source judgment text.

Requirements:
- Audit only this row.
- Enforce factual accuracy and temporal wall safety.
- Prefer rewriting over removal when a fact can be salvaged into a supported substantive event.
- Remove a fact only if it is mainly procedural, trial-stage, or judgment-stage and cannot be salvaged.
- Keep Metadata.Case_Number unchanged.
- Set leakage_score to 0 in the model response. Python will recalculate the final Leakage_Score.
- Every kept fact must be specifically supported for this row's party.
- Every kept fact text must be a complete, grammatical sentence.
- If one sentence or paragraph contains different events with different dates, do not assign one date to all events.
- For appellate rows, use the specific appeal number in the row and do not merge outcomes across linked appeals.
- If uncertain, prefer no change.

PARTY JSON ROW:
<<<
{json_block}
>>>

SOURCE PDF TEXT:
<<<
{source_text}
>>>

If nothing needs changing, return:
{{
  "case_name": "",
  "party": "",
  "status": "CLEAN",
  "leakage_score": 0,
  "label_change": null,
  "fact_changes": [],
  "facts_to_remove": []
}}

Return only JSON matching the system schema.
""".strip()

# =========================
# 4. RULE-BASED SAFETY NETS
# =========================
LEAKAGE_PATTERNS = [
    r"\bthe\s+(?:trial\s+|high\s+|appeals?\s+)?(?:judge|court|tribunal)\s+(?:held|found|concluded|ruled|determined|decided|observed|noted)\b",
    r"\b(?:judgment|claim|action|suit|appeal)\s+(?:of\s+the\s+.*?)?(?:was|is|were|are)\s+(?:entered|dismissed|allowed|awarded)\b",
    r"\b(?:succeeded|failed)\s+in\s+(?:its|his|her|their)\s+(?:claim|action|suit|appeal)\b",
    r"\b(?:found|held\s+(?:to\s+be\s+)?)?liable\b",
    r"\b(?:costs?|damages?)\s+(?:were|was|are|is)?\s*(?:awarded|ordered|assessed)\b",
    r"\bwrit of summons\b",
    r"\bstatutory demand\b",
    r"\bcross-examination\b",
    r"\btestified\b",
    r"\baffidavit\b",
    r"\bsubmission[s]?\b",
    r"\bstatement of claim\b",
]

STRONG_FACT_REMOVAL_PATTERNS = [
    r"\bwrit of summons\b",
    r"\bstatement of claim\b",
    r"\bstatutory demand\b",
    r"\baffidavit\b",
    r"\bcross-examination\b",
    r"\b(?:he|she|they) testified\b",
    r"\bsubmission[s]?\b",
    r"\b(?:costs?|damages?)\s+(?:were|was|are|is)?\s*(?:awarded|ordered|assessed)\b",
    r"\bjudgment\s+(?:was|is)?\s*(?:entered|dismissed|allowed|awarded)\b",
]

FACT_TEXT_SCRUB_PATTERNS = [
    r"(?i)\s*(?:the\s+)?(?:judge|court|tribunal)\s+(?:held|found|concluded|determined|ruled|observed|noted)(?:\s+that)?\s*",
    r"(?i)\s*(?:the\s+)?(?:plaintiffs?|defendants?|appellants?|respondents?)(?:\s+by\s+counterclaim)?\s+(?:argued|contended|submitted|stated|claimed|maintained|denied|alleged)(?:\s+that)?\s*",
    r"(?i)\s*(?:Mr\.?|Mrs\.?|Ms\.?|Mdm\.?|Dr\.?)?\s*[A-Z][a-zA-Z0-9.\s&]+\s+(?:argued|contended|submitted|stated|claimed|maintained|denied|alleged)(?:\s+that)?\s*"
]

HINDSIGHT_FIELD_PATTERNS = [
    r"(?i)\b(?:the\s+)?(?:judge|court|tribunal)\s+(?:held|found|concluded|determined|ruled|observed|noted)(?:\s+that)?\b",
    r"(?i)\b(?:the\s+)?(?:plaintiffs?|defendants?|appellants?|respondents?)(?:\s+by\s+counterclaim)?\s+(?:argued|contended|submitted|stated|claimed|maintained|denied|alleged)(?:\s+that)?\b",
    r"(?i)\b(?:Mr\.?|Mrs\.?|Ms\.?|Mdm\.?|Dr\.?)?\s*[A-Z][a-zA-Z0-9.\s&]+\s+(?:argued|contended|submitted|stated|claimed|maintained|denied|alleged)(?:\s+that)?\b"
]

DATE_ISO_DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")
DATE_ISO_MONTH = re.compile(r"^\d{4}-\d{2}$")
DATE_YEAR = re.compile(r"^\d{4}$")
COUNTERCLAIM_SUFFIX_RE = re.compile(r"\s*\(\s*Counterclaim\s*\)\s*$", re.IGNORECASE)

STOPWORD_TOKENS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "were", "was", "are", "has",
    "have", "had", "been", "after", "before", "during", "between", "about", "around", "under",
    "onto", "through", "their", "there", "which", "while", "would", "could", "should", "stated",
    "argued", "contended", "company", "plaintiff", "defendant", "court", "judge"
}

# =========================
# 5. HELPERS
# =========================
def safe_json_load(content: str) -> Dict[str, Any]:
    if content is None:
        raise ValueError("Model returned empty content")

    text = content.strip()
    if not text:
        raise ValueError("Model returned empty text")

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise


def normalize_input_data(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, dict):
        if isinstance(data.get("results"), list):
            return data["results"]
        return [data]
    if isinstance(data, list):
        return data
    raise ValueError("Unsupported JSON structure")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_pdf(path: Path) -> str:
    text_parts: List[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts)


def build_prompt(source_text: str, row: Dict[str, Any]) -> str:
    return USER_PROMPT_TEMPLATE.format(
        source_text=source_text[:120000],
        json_block=json.dumps(row, ensure_ascii=False, indent=2),
    )


def contains_leakage(text: Any) -> bool:
    if not isinstance(text, str) or not text.strip():
        return False
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in LEAKAGE_PATTERNS)


def calculate_leakage_score(row: Dict[str, Any]) -> int:
    party = row.get("Party_Details", {})
    score = 0

    for field in ["Issue", "Rule", "Application"]:
        if contains_leakage(party.get(field, "")):
            score += 1

    for fact in party.get("Facts", []):
        if isinstance(fact, dict):
            fact_text = fact.get("Text", "")
        elif isinstance(fact, str):
            fact_text = fact
        else:
            continue

        if contains_leakage(fact_text):
            score += 1

    return score


def scrub_hindsight_language(text: Any) -> Any:
    if not isinstance(text, str) or not text.strip():
        return text
    cleaned = text
    for pattern in HINDSIGHT_FIELD_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.;:-")
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


def scrub_fact_text(text: Any) -> Any:
    if not isinstance(text, str) or not text.strip():
        return text
    cleaned = text
    for pattern in FACT_TEXT_SCRUB_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.;:-")
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


def allowed_labels_for_role(role: str) -> set:
    role_l = (role or "").strip().lower()
    if role_l == "plaintiff":
        return PLAINTIFF_LABELS
    if role_l == "defendant":
        return DEFENDANT_LABELS
    return GENERIC_LABELS


def normalize_fact_type(value: Any) -> Tuple[str, bool]:
    if not isinstance(value, str) or not value.strip():
        return "CONDUCT", True

    key = re.sub(r"[^A-Za-z]", "", value).upper()
    canonical = FACT_TYPE_ALIASES.get(key)
    if canonical:
        return canonical, canonical != value.strip()

    candidate = value.strip().upper().replace(" ", "_").replace("-", "_")
    if candidate in ALLOWED_FACT_TYPES:
        return candidate, candidate != value.strip()

    return "CONDUCT", True


def coerce_fact_entry(fact: Any) -> Optional[Dict[str, Any]]:
    """
    Ensure each fact is a dict with Fact_Type / Fact_Date / Text.
    If a malformed string fact appears, salvage it instead of crashing.
    """
    if isinstance(fact, dict):
        return fact

    if isinstance(fact, str) and fact.strip():
        return {
            "Fact_Type": "CONDUCT",
            "Fact_Date": "",
            "Text": fact.strip(),
        }

    return None


def month_to_num(token: str) -> int:
    return MONTHS[token.strip().lower()]


def normalize_partial_iso_to_full(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    value = value.strip()
    if not value:
        return ""
    if DATE_ISO_DAY.match(value):
        return value
    if DATE_ISO_MONTH.match(value):
        return f"{value}-01"
    if DATE_YEAR.match(value):
        return f"{value}-01-01"
    return ""


def extract_first_date_candidate(text: str) -> str:
    if not text:
        return ""

    candidates: List[Tuple[int, int, str]] = []
    occupied: List[Tuple[int, int]] = []

    def overlaps(span: Tuple[int, int]) -> bool:
        s1, e1 = span
        for s2, e2 in occupied:
            if s1 < e2 and s2 < e1:
                return True
        return False

    for match in ISO_DATE_RE.finditer(text):
        iso = match.group(0)
        candidates.append((match.start(), 5, iso))
        occupied.append(match.span())

    for match in DAY_MONTH_YEAR_RE.finditer(text):
        if overlaps(match.span()):
            continue
        day = int(match.group(1))
        month = month_to_num(match.group(2))
        year = int(match.group(3))
        iso = f"{year:04d}-{month:02d}-{day:02d}"
        candidates.append((match.start(), 5, iso))
        occupied.append(match.span())

    for match in MONTH_DAY_YEAR_RE.finditer(text):
        if overlaps(match.span()):
            continue
        month = month_to_num(match.group(1))
        day = int(match.group(2))
        year = int(match.group(3))
        iso = f"{year:04d}-{month:02d}-{day:02d}"
        candidates.append((match.start(), 5, iso))
        occupied.append(match.span())

    for match in ISO_MONTH_RE.finditer(text):
        if overlaps(match.span()):
            continue
        iso = f"{match.group(0)}-01"
        candidates.append((match.start(), 4, iso))
        occupied.append(match.span())

    for match in MONTH_YEAR_RE.finditer(text):
        if overlaps(match.span()):
            continue
        month = month_to_num(match.group(1))
        year = int(match.group(2))
        iso = f"{year:04d}-{month:02d}-01"
        candidates.append((match.start(), 4, iso))
        occupied.append(match.span())

    for match in YEAR_RE.finditer(text):
        if overlaps(match.span()):
            continue
        year = int(match.group(1))
        iso = f"{year:04d}-01-01"
        candidates.append((match.start(), 1, iso))

    if not candidates:
        return ""

    candidates.sort(key=lambda x: (x[0], -x[1]))
    return candidates[0][2]


def choose_supporting_context(fact_text: str, source_text: str) -> str:
    if not isinstance(fact_text, str) or not fact_text.strip() or not source_text:
        return ""

    bad_header_markers = (
        "Case Number :",
        "Decision Date :",
        "Tribunal/Court :",
        "Counsel Name(s) :",
        "Parties :",
        "Version No",
        "Judgment reserved.",
        "Copyright ©",
    )

    paragraphs = [p.strip() for p in re.split(r"\n+", source_text) if p.strip()]
    paragraphs = [p for p in paragraphs if not any(marker.lower() in p.lower() for marker in bad_header_markers)]

    tokens = [
        t for t in re.findall(r"[A-Za-z][A-Za-z'-]{2,}", fact_text.lower())
        if t not in STOPWORD_TOKENS and t not in MONTHS
    ]
    if not tokens:
        return ""

    tokens = tokens[:10]

    best_para = ""
    best_score = 0
    best_idx = -1

    for idx, para in enumerate(paragraphs):
        para_l = para.lower()
        score = sum(1 for tok in tokens if tok in para_l)
        if score > best_score:
            best_score = score
            best_para = para
            best_idx = idx

    if best_score >= 3 and best_idx != -1:
        parts = []
        if best_idx > 0:
            parts.append(paragraphs[best_idx - 1])
        parts.append(paragraphs[best_idx])
        if best_idx + 1 < len(paragraphs):
            parts.append(paragraphs[best_idx + 1])
        return " ".join(parts)

    return ""


def choose_best_fact_date(
    current_value: Any,
    original_text: str,
    cleaned_text: str,
    source_text: str,
) -> str:
    """
    Conservative date selection:
    1. Prefer an explicit date in the original fact text.
    2. Then prefer an explicit date in the cleaned fact text.
    3. Then keep an existing valid ISO date.
    4. Only then fall back to supporting-context inference.
    """
    original_text = original_text if isinstance(original_text, str) else ""
    cleaned_text = cleaned_text if isinstance(cleaned_text, str) else ""

    original_text_date = extract_first_date_candidate(original_text)
    if original_text_date:
        return original_text_date

    cleaned_text_date = extract_first_date_candidate(cleaned_text)
    if cleaned_text_date:
        return cleaned_text_date

    current_full = normalize_partial_iso_to_full(current_value)
    if current_full:
        return current_full

    context = choose_supporting_context(original_text, source_text)
    context_date = extract_first_date_candidate(context)
    if context_date:
        return context_date

    return ""


def clean_single_fact(
    raw_fact: Any,
    source_text: str,
    allow_strong_leakage: bool = False,
) -> Optional[Dict[str, Any]]:
    fact = coerce_fact_entry(raw_fact)
    if not fact:
        return None

    text = fact.get("Text", "")
    if not isinstance(text, str) or not text.strip():
        return None

    original_text = text
    revised_text = scrub_fact_text(original_text)
    final_text = revised_text if isinstance(revised_text, str) and revised_text.strip() else original_text

    if not allow_strong_leakage:
        strong_leakage = any(
            re.search(pattern, final_text, re.IGNORECASE)
            for pattern in STRONG_FACT_REMOVAL_PATTERNS
        )
        if strong_leakage:
            return None

    fact["Text"] = final_text
    fact["Fact_Date"] = choose_best_fact_date(
        fact.get("Fact_Date", ""),
        original_text,
        final_text,
        source_text,
    )

    normalized_type, changed = normalize_fact_type(fact.get("Fact_Type", ""))
    if changed:
        fact["Fact_Type"] = normalized_type

    return fact


def get_value_by_path(obj: Any, path: str) -> Any:
    current = obj
    for part in path.split("."):
        if "[" in part and "]" in part:
            key = part[:part.index("[")]
            idx = int(part[part.index("[") + 1:part.index("]")])
            if key:
                current = current[key]
            current = current[idx]
        else:
            current = current[part]
    return current


def set_value_by_path(obj: Any, path: str, new_value: Any) -> None:
    parts = path.split(".")
    current = obj
    for part in parts[:-1]:
        if "[" in part and "]" in part:
            key = part[:part.index("[")]
            idx = int(part[part.index("[") + 1:part.index("]")])
            if key:
                current = current[key]
            current = current[idx]
        else:
            current = current[part]

    last = parts[-1]
    if "[" in last and "]" in last:
        key = last[:last.index("[")]
        idx = int(last[last.index("[") + 1:last.index("]")])
        if key:
            current = current[key]
        current[idx] = new_value
    else:
        current[last] = new_value


def parse_fact_index(path: str) -> Optional[int]:
    marker = "Party_Details.Facts["
    if marker not in path:
        return None
    start = path.index(marker) + len(marker)
    end = path.index("]", start)
    return int(path[start:end])


def split_case_bucket(case_number: str) -> Tuple[str, str]:
    raw = (case_number or "").strip()
    if COUNTERCLAIM_SUFFIX_RE.search(raw):
        base = COUNTERCLAIM_SUFFIX_RE.sub("", raw).strip()
        return base, "counterclaim"
    return raw, "main"


def normalize_name(value: str) -> str:
    value = (value or "").strip().lower()
    value = re.sub(r"\s+", " ", value)
    return value


def base_case_number(case_number: str) -> str:
    return split_case_bucket(case_number)[0]


def is_counterclaim_case(case_number: str) -> bool:
    return bool(COUNTERCLAIM_SUFFIX_RE.search((case_number or "").strip()))


def is_appeal_case(case_number: str) -> bool:
    return (case_number or "").strip().lower().startswith("civil appeal no")


def should_fill_representation(party: Dict[str, Any]) -> bool:
    law_firm = str(party.get("Law_Firm", "")).strip()
    counsel = party.get("Counsel", [])
    return law_firm in {"", "Unknown"} or not counsel


def parse_names_list(names_text: str) -> List[str]:
    text = (names_text or "").strip()
    text = re.sub(r"\s+and\s+", ", ", text)
    names = [n.strip(" ,;") for n in text.split(",") if n.strip(" ,;")]
    return names


def extract_first_page_text(pdf_path: Path) -> str:
    with pdfplumber.open(pdf_path) as pdf:
        if not pdf.pages:
            return ""
        return pdf.pages[0].extract_text() or ""


def extract_header_counsel_blocks(pdf_path: Path) -> List[Dict[str, Any]]:
    first_page_text = extract_first_page_text(pdf_path)
    if not first_page_text:
        return []

    match = HEADER_COUNSEL_SECTION_RE.search(first_page_text)
    if not match:
        return []

    section = re.sub(r"\s+", " ", match.group("section")).strip()
    parts = [p.strip(" ;") for p in re.split(r"\s*;\s*", section) if p.strip(" ;")]

    blocks: List[Dict[str, Any]] = []
    for part in parts:
        m = HEADER_COUNSEL_BLOCK_RE.match(part)
        if not m:
            continue
        blocks.append({
            "law_firm": m.group("firm").strip(),
            "counsel": parse_names_list(m.group("names")),
            "role_desc": m.group("role_desc").strip(),
            "source": "header",
        })
    return blocks


def extract_trailing_counsel_blocks(source_text: str) -> List[Dict[str, Any]]:
    if not source_text:
        return []

    tail = re.sub(r"\s+", " ", source_text[-12000:]).strip()
    blocks: List[Dict[str, Any]] = []

    for match in TRAILING_COUNSEL_RE.finditer(tail):
        blocks.append({
            "law_firm": match.group("firm").strip(),
            "counsel": parse_names_list(match.group("names")),
            "role_desc": match.group("role_desc").strip(),
            "source": "trailing",
        })

    return blocks


def normalize_suit_side(side: str) -> Optional[str]:
    s = (side or "").strip().lower()
    if s.startswith("plaintiff"):
        return "plaintiff"
    if s.startswith("defendant"):
        return "defendant"
    return None


def parse_suit_role_desc(role_desc: str) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for m in SUIT_ROLE_ENTRY_RE.finditer(role_desc or ""):
        side = normalize_suit_side(m.group("side"))
        if not side:
            continue
        results.append({
            "side": side,
            "case_no": (m.group("case_no") or "").strip() or None,
        })
    return results


def parse_appeal_role_desc(role_desc: str) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for m in APPEAL_ROLE_ENTRY_RE.finditer(role_desc or ""):
        ordinals_text = m.group("ordinals") or ""
        ordinals = [int(x) for x in ORDINAL_RE.findall(ordinals_text)] if ordinals_text else [None]
        for ordinal in ordinals:
            results.append({
                "side": m.group("side").strip().lower(),
                "appeal_no": m.group("appeal_no").strip(),
                "ordinal": ordinal,
            })
    return results


def build_original_side_name_sets(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, set]]:
    grouped: Dict[str, Dict[str, set]] = {}

    for row in rows:
        case_number = row.get("Metadata", {}).get("Case_Number", "")
        if is_counterclaim_case(case_number):
            continue

        base = base_case_number(case_number)
        grouped.setdefault(base, {"plaintiff": set(), "defendant": set()})

        role = str(row.get("Party_Details", {}).get("Role", "")).strip().lower()
        name = normalize_name(str(row.get("Party_Details", {}).get("Name", "")))
        if not name:
            continue

        if role == "plaintiff":
            grouped[base]["plaintiff"].add(name)
        elif role == "defendant":
            grouped[base]["defendant"].add(name)

    return grouped


def parse_appellants_from_judgment_text(source_text: str) -> Dict[str, str]:
    results: Dict[str, str] = {}

    for m in FILED_APPEAL_RE.finditer(source_text or ""):
        party_name = m.group(1).strip(" ,.;:")
        appeal_no = m.group(2).strip()
        results[appeal_no] = party_name

    for m in CROSS_APPEAL_RE.finditer(source_text or ""):
        party_name = m.group(1).strip(" ,.;:")
        appeal_no = m.group(2).strip()
        results[appeal_no] = party_name

    return results


def fill_row_representation(row: Dict[str, Any], block: Dict[str, Any]) -> None:
    party = row.get("Party_Details", {})
    if str(party.get("Law_Firm", "")).strip() in {"", "Unknown"}:
        party["Law_Firm"] = block["law_firm"]
    if not party.get("Counsel"):
        party["Counsel"] = list(block["counsel"])


def fill_representation_for_suits(
    rows: List[Dict[str, Any]],
    header_blocks: List[Dict[str, Any]],
    trailing_blocks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    raw_blocks = header_blocks if header_blocks else trailing_blocks
    if not raw_blocks:
        return rows

    entries: List[Dict[str, Any]] = []
    for block in raw_blocks:
        for entry in parse_suit_role_desc(block["role_desc"]):
            entries.append({
                "case_no": entry["case_no"],
                "side": entry["side"],
                "law_firm": block["law_firm"],
                "counsel": block["counsel"],
            })

    if not entries:
        return rows

    original_side_sets = build_original_side_name_sets(rows)

    for row in rows:
        party = row.get("Party_Details", {})
        if not should_fill_representation(party):
            continue

        case_number = row.get("Metadata", {}).get("Case_Number", "")
        base = base_case_number(case_number)
        name = normalize_name(str(party.get("Name", "")))

        original_side = None
        if name in original_side_sets.get(base, {}).get("plaintiff", set()):
            original_side = "plaintiff"
        elif name in original_side_sets.get(base, {}).get("defendant", set()):
            original_side = "defendant"
        elif not is_counterclaim_case(case_number):
            role = str(party.get("Role", "")).strip().lower()
            if role in {"plaintiff", "defendant"}:
                original_side = role

        if not original_side:
            continue

        match = next(
            (
                e for e in entries
                if e["side"] == original_side
                and (e["case_no"] is None or e["case_no"].lower() == base.lower())
            ),
            None,
        )
        if match:
            fill_row_representation(row, match)

    return rows


def fill_representation_for_appeals(
    rows: List[Dict[str, Any]],
    header_blocks: List[Dict[str, Any]],
    source_text: str,
) -> List[Dict[str, Any]]:
    if not header_blocks:
        return rows

    entries: List[Dict[str, Any]] = []
    for block in header_blocks:
        for entry in parse_appeal_role_desc(block["role_desc"]):
            entries.append({
                "appeal_no": entry["appeal_no"],
                "side": entry["side"],
                "ordinal": entry["ordinal"],
                "law_firm": block["law_firm"],
                "counsel": block["counsel"],
            })

    if not entries:
        return rows

    appellant_by_appeal = parse_appellants_from_judgment_text(source_text)
    grouped: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {}

    for idx, row in enumerate(rows):
        appeal_no = base_case_number(row.get("Metadata", {}).get("Case_Number", ""))
        grouped.setdefault(appeal_no, []).append((idx, row))

    for appeal_no, indexed_rows in grouped.items():
        blocks_for_appeal = [b for b in entries if b["appeal_no"].lower() == appeal_no.lower()]
        if not blocks_for_appeal:
            continue

        appellant_name = normalize_name(appellant_by_appeal.get(appeal_no, ""))
        appellant_idx = None

        if appellant_name:
            for idx, row in indexed_rows:
                row_name = normalize_name(str(row.get("Party_Details", {}).get("Name", "")))
                if row_name == appellant_name:
                    appellant_idx = idx
                    break

        appellant_block = next((b for b in blocks_for_appeal if b["side"] == "appellant"), None)
        if appellant_block is not None and appellant_idx is not None:
            row = rows[appellant_idx]
            if should_fill_representation(row.get("Party_Details", {})):
                fill_row_representation(row, appellant_block)

        respondent_candidates = [(idx, row) for idx, row in indexed_rows if idx != appellant_idx]
        respondent_blocks = sorted(
            [b for b in blocks_for_appeal if b["side"] == "respondent"],
            key=lambda x: (x["ordinal"] is None, x["ordinal"] or 999),
        )

        for block in respondent_blocks:
            if block["ordinal"] is None:
                if len(respondent_candidates) == 1:
                    idx, row = respondent_candidates[0]
                    if should_fill_representation(row.get("Party_Details", {})):
                        fill_row_representation(row, block)
                continue

            pos = block["ordinal"] - 1
            if 0 <= pos < len(respondent_candidates):
                idx, row = respondent_candidates[pos]
                if should_fill_representation(row.get("Party_Details", {})):
                    fill_row_representation(row, block)

    return rows


def fill_missing_representation(
    rows: List[Dict[str, Any]],
    pdf_path: Path,
    source_text: str,
) -> List[Dict[str, Any]]:
    if not any(should_fill_representation(r.get("Party_Details", {})) for r in rows):
        return rows

    first_case = rows[0].get("Metadata", {}).get("Case_Number", "")
    header_blocks = extract_header_counsel_blocks(pdf_path)
    trailing_blocks = extract_trailing_counsel_blocks(source_text)

    if is_appeal_case(first_case):
        return fill_representation_for_appeals(rows, header_blocks, source_text)

    return fill_representation_for_suits(rows, header_blocks, trailing_blocks)


def plaintiff_to_defendant_label(label: str) -> Optional[str]:
    mapping = {
        "Claim Allowed": "Liable",
        "Claim Allowed in Part": "Liable",
        "Claim Dismissed": "Not Liable",
    }
    return mapping.get(label)


def defendant_to_plaintiff_label(label: str) -> Optional[str]:
    mapping = {
        "Liable": "Claim Allowed",
        "Not Liable": "Claim Dismissed",
    }
    return mapping.get(label)


def normalize_audit_result(result: Dict[str, Any], row: Dict[str, Any]) -> Dict[str, Any]:
    metadata = row.get("Metadata", {})
    party_details = row.get("Party_Details", {})

    result["case_name"] = (
        result.get("case_name")
        or metadata.get("Case_Name")
        or metadata.get("Case_Number")
        or ""
    )
    result["party"] = party_details.get("Name", "")
    result.setdefault("status", "CLEAN")
    result.setdefault("leakage_score", 0)
    result.setdefault("label_change", None)
    result.setdefault("fact_changes", [])
    result.setdefault("facts_to_remove", [])

    if not isinstance(result["fact_changes"], list):
        result["fact_changes"] = []
    if not isinstance(result["facts_to_remove"], list):
        result["facts_to_remove"] = []
    if result["label_change"] is not None and not isinstance(result["label_change"], dict):
        result["label_change"] = None

    if result["label_change"] or result["fact_changes"] or result["facts_to_remove"]:
        result["status"] = "CHANGES"

    banned_role_labels = {"plaintiff", "defendant", "third party", "third-party"}
    if result.get("label_change"):
        recommended = str(result["label_change"].get("recommended", "")).strip().lower()
        if recommended in banned_role_labels:
            result["label_change"] = None

    return result

# =========================
# 6. MODEL CALL
# =========================
def audit_row_with_model(source_text: str, row: Dict[str, Any]) -> Dict[str, Any]:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        reasoning_effort="high",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(source_text, row)},
        ],
    )

    content = response.choices[0].message.content
    result = safe_json_load(content)
    return normalize_audit_result(result, row)


def apply_audit_result(row: Dict[str, Any], audit: Dict[str, Any]) -> Dict[str, Any]:
    updated = deepcopy(row)

    label_change = audit.get("label_change")
    if isinstance(label_change, dict) and label_change.get("recommended") is not None:
        old_label = updated.get("Label", "Unknown")
        new_label = label_change["recommended"]
        if old_label != new_label:
            updated["Label"] = new_label

    fact_changes = audit.get("fact_changes", [])
    if isinstance(fact_changes, list):
        for item in fact_changes:
            if not isinstance(item, dict):
                continue

            path = item.get("path")
            new_value = item.get("recommended_value")

            if not path or path in PROTECTED_PATHS:
                continue

            if path.startswith("Party_Details.Facts["):
                allowed_suffixes = (".Text", ".Fact_Date", ".Fact_Type")
                if not path.endswith(allowed_suffixes):
                    continue

            try:
                old_value = get_value_by_path(updated, path)
                if old_value != new_value:
                    set_value_by_path(updated, path, new_value)
            except Exception:
                continue

    facts_to_remove = audit.get("facts_to_remove", [])
    if isinstance(facts_to_remove, list):
        indices = []
        for item in facts_to_remove:
            if not isinstance(item, dict):
                continue
            idx = parse_fact_index(item.get("path", ""))
            if idx is not None:
                indices.append(idx)

        facts = updated.get("Party_Details", {}).get("Facts", [])
        for idx in sorted(set(indices), reverse=True):
            if 0 <= idx < len(facts):
                del facts[idx]

    return updated

# =========================
# 7. RULE-BASED CLEANUP
# =========================
def apply_rule_based_cleanup(row: Dict[str, Any], source_text: str) -> Dict[str, Any]:
    updated = row
    party = updated.get("Party_Details", {})

    role = party.get("Role", "")
    label = updated.get("Label", "Unknown")
    allowed = allowed_labels_for_role(role)
    if label not in allowed:
        updated["Label"] = "Unknown"

    for field in ["Issue", "Rule", "Application"]:
        party[field] = scrub_hindsight_language(party.get(field, ""))

    original_facts = party.get("Facts", [])

    clean_facts = []
    for raw_fact in original_facts:
        cleaned = clean_single_fact(raw_fact, source_text, allow_strong_leakage=False)
        if cleaned:
            clean_facts.append(cleaned)

    if original_facts and not clean_facts:
        fallback_facts = []
        for raw_fact in original_facts:
            cleaned = clean_single_fact(raw_fact, source_text, allow_strong_leakage=True)
            if cleaned:
                fallback_facts.append(cleaned)
        clean_facts = fallback_facts

    party["Facts"] = clean_facts
    updated["Leakage_Score"] = calculate_leakage_score(updated)
    return updated

# =========================
# 8. CASE-LEVEL CONSISTENCY CHECKS
# =========================
def apply_case_level_consistency(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        case_number = row.get("Metadata", {}).get("Case_Number", "")
        base_case, bucket = split_case_bucket(case_number)
        grouped.setdefault((base_case, bucket), []).append(row)

    for (_base_case, _bucket), bucket_rows in grouped.items():
        plaintiffs = [r for r in bucket_rows if r.get("Party_Details", {}).get("Role", "").strip().lower() == "plaintiff"]
        defendants = [r for r in bucket_rows if r.get("Party_Details", {}).get("Role", "").strip().lower() == "defendant"]

        plaintiff_labels = [r.get("Label", "Unknown") for r in plaintiffs]
        defendant_labels = [r.get("Label", "Unknown") for r in defendants]

        if len(plaintiffs) == 1 and len(defendants) == 1:
            p_row = plaintiffs[0]
            d_row = defendants[0]
            p_label = p_row.get("Label", "Unknown")
            d_label = d_row.get("Label", "Unknown")

            expected_d = plaintiff_to_defendant_label(p_label)
            expected_p = defendant_to_plaintiff_label(d_label)

            if p_label != "Unknown" and d_label == "Unknown" and expected_d:
                d_row["Label"] = expected_d
            elif d_label != "Unknown" and p_label == "Unknown" and expected_p:
                p_row["Label"] = expected_p
            elif p_label != "Unknown" and d_label != "Unknown" and expected_d and d_label != expected_d:
                d_row["Label"] = expected_d

        if plaintiffs and defendants:
            all_plaintiffs_dismissed = all(label == "Claim Dismissed" for label in plaintiff_labels)
            all_defendants_not_liable = (
                all(label == "Not Liable" for label in defendant_labels if label != "Unknown")
                and any(label != "Unknown" for label in defendant_labels)
            )

            if all_plaintiffs_dismissed:
                for row in defendants:
                    if row.get("Label") in {"Liable", "Unknown"}:
                        row["Label"] = "Not Liable"

            if all_defendants_not_liable:
                for row in plaintiffs:
                    if row.get("Label") in {"Claim Allowed", "Claim Allowed in Part", "Unknown"}:
                        row["Label"] = "Claim Dismissed"

    return rows

# =========================
# 9. FILE PROCESSING
# =========================
def process_case(json_path: Path) -> None:
    pdf_path = json_path.with_suffix(".pdf")
    if not pdf_path.exists():
        print(f"Skipping {json_path.name}: no matching PDF found")
        return

    print(f"Auditing {json_path.name}")
    raw_data = read_json(json_path)
    rows = normalize_input_data(raw_data)
    if not rows:
        print(f"  No rows found in {json_path.name}")
        return

    pdf_text = read_pdf(pdf_path)
    corrected_rows: List[Dict[str, Any]] = []

    for i, row in enumerate(rows, start=1):
        party_name = row.get("Party_Details", {}).get("Name", f"row_{i}")
        print(f"  Auditing row {i}/{len(rows)}: {party_name}")

        try:
            audit_result = audit_row_with_model(pdf_text, row)
        except Exception as e:
            print(f"    Model audit failed for {party_name}: {e}")
            audit_result = {
                "case_name": row.get("Metadata", {}).get("Case_Number", ""),
                "party": party_name,
                "status": "CLEAN",
                "leakage_score": 0,
                "label_change": None,
                "fact_changes": [],
                "facts_to_remove": [],
            }

        final_row = apply_audit_result(row, audit_result)
        final_row = apply_rule_based_cleanup(final_row, pdf_text)
        corrected_rows.append(final_row)

    corrected_rows = apply_case_level_consistency(corrected_rows)
    corrected_rows = fill_missing_representation(corrected_rows, pdf_path, pdf_text)

    for row in corrected_rows:
        row["Leakage_Score"] = calculate_leakage_score(row)

    corrected_path = OUTPUT_FOLDER / f"{json_path.stem}.correct.json"
    with open(corrected_path, "w", encoding="utf-8") as f:
        json.dump(corrected_rows, f, indent=2, ensure_ascii=False)

    print(f"  Saved: {corrected_path.name}")

# =========================
# 10. BATCH RUNNER
# =========================
def main() -> None:
    json_files = [
        p for p in INPUT_FOLDER.glob("*.json")
        if not p.name.endswith(".corrected.json")
        and not p.name.endswith(".audit.json")
        and not p.name.endswith(".review.json")
        and not p.name.endswith(".correct.json")
    ]

    if not json_files:
        print("No JSON files found")
        return

    for json_file in json_files:
        process_case(json_file)

    print("DONE")


if __name__ == "__main__":
    main()