import json
import re
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

load_dotenv()
client = OpenAI()

MODEL_NAME = "gpt-5.4-mini"

SYSTEM_PROMPT = """
You are an expert legal ML data auditor.

Your job is to audit one party row against the source judgment text and strictly enforce two concepts: ACCURACY and the TEMPORAL WALL.

1. FACTUAL ACCURACY
- Every fact must be explicitly supported by the source text.
- Correct hallucinations, including wrong dates, wrong amounts, wrong parties, wrong roles, or misleading wording.
- If the judgment only gives a year, month, range, or approximate period, do not invent an exact day.

2. THE TEMPORAL WALL
- Predictive ML models must only use facts that would have been available before the substantive dispute outcome was determined.
- The Temporal Wall is the earliest point at which the dispute had clearly arisen between the parties, and no later than the commencement of legal proceedings.
- The "Issue", "Rule", and "Application" fields MUST be completely neutral. You are strictly forbidden from using phrases like "The judge held", "The court found", or "The court concluded" in the Application or Issue fields. Rewrite these statements as neutral, pre-trial facts.
- If uncertain, treat the following as leakage:
  - litigation filings
  - procedural applications
  - hearings
  - trial evidence
  - witness testimony
  - submissions
  - court reasoning
  - judicial findings
  - appeals
  - judgment-stage outcomes

3. OUTCOME LABEL
- Party_Details.Role is metadata, not the label.
- The top-level "Label" must be the substantive outcome for that row.
- For plaintiff rows, valid labels are: "Claim Allowed", "Claim Dismissed", "Unknown".
- For defendant rows, valid labels are: "Liable", "Not Liable", "Partially Liable", "Unknown".
- For third-party or procedural-only rows, use "Unknown" unless the judgment clearly determines a substantive outcome.
- Do NOT recommend "Defendant", "Plaintiff", or "Third Party" as labels.

Return only valid JSON with this shape:
{
  "case_name": string,
  "party": string,
  "status": "CLEAN" | "CHANGES",
  "label_change": null | {
    "current": string,
    "recommended": string,
    "evidence": string,
    "recommended_change": string
  },
  "fact_changes": [
    {
      "path": string,
      "issue_type": "ACCURACY_ERROR" | "TEMPORAL_WALL_LEAKAGE",
      "current_value": string,
      "recommended_value": string,
      "evidence": string,
      "recommended_change": string
    }
  ],
  "facts_to_remove": [
    {
      "path": string,
      "issue_type": "ACCURACY_ERROR" | "TEMPORAL_WALL_LEAKAGE",
      "current_value": string,
      "evidence": string,
      "recommended_change": string
    }
  ]
}
""".strip()

USER_PROMPT_TEMPLATE = """
SOURCE PDF TEXT:
<<<
{source_text}
>>>

PARTY JSON ROW:
<<<
{json_block}
>>>

INSTRUCTIONS:
- Audit only this row.
- Enforce ACCURACY and TEMPORAL WALL.
- Remove facts that contain litigation-stage actions, court findings, or judgment reasoning.
- Fix incorrect dates and values to match the source exactly.
- The "Label" must represent the outcome for this party, not their role.
- If no clear outcome exists, keep "Unknown".
- Be conservative. If uncertain, prefer no change over a speculative change.

If nothing needs changing, return:
{{
  "case_name": "",
  "party": "",
  "status": "CLEAN",
  "label_change": null,
  "fact_changes": [],
  "facts_to_remove": []
}}

Return only JSON.
""".strip()

PLAINTIFF_LABELS = {"Claim Allowed", "Claim Dismissed", "Unknown"}
DEFENDANT_LABELS = {"Liable", "Not Liable", "Partially Liable", "Unknown"}
GENERIC_LABELS = {"Unknown"}

LEAKAGE_PATTERNS = [
    r"\bthe\s+(?:trial\s+|high\s+|appeals?\s+)?(?:judge|court|tribunal)\s+(?:held|found|concluded|ruled|determined|decided|observed|noted)\b",
    r"\b(?:judgment|claim|action|suit|appeal)\s+(?:of\s+the\s+.*?)?(?:was|is|were|are)\s+(?:entered|dismissed|allowed|awarded)\b",
    r"\b(?:succeeded|failed)\s+in\s+(?:its|his|her|their)\s+(?:claim|action|suit|appeal)\b",
    r"\b(?:found|held\s+(?:to\s+be\s+)?)?liable\b",
    r"\b(?:costs?|damages?)\s+(?:were|was|are|is)?\s*(?:awarded|ordered|assessed)\b",
    r"\bletter of demand\b",
    r"\bdemand from .*? lawyers\b",
    r"\blawyers for\b",
    r"\bsolicitors for\b",
    r"\bwrit of summons\b",
    r"\bstatutory demand\b",
    r"\bletter in response to .*? demand\b",
    r"\blater (?:medical |hospital )?records?\b",
    r"\baccording to later\b",
    r"\blater account\b",
    r"\btestified later\b",
    r"\bproduced during discovery\b",
    r"\b(?:later|subsequently)\s+(?:seen|examined|diagnosed)\s+by\b",
    r"\bpsychiatri(?:c|st)\b",
    r"\bpolyclinic\b"
]

SUCCESS_PATTERNS_PLAINTIFF = [
    r"\bjudgment\s+(?:was\s+|is\s+)?(?:entered|given|awarded)\s+in\s+favour\s+of\b",
    r"\bjudgment\s+against\b",
    r"\b(?:claim|action|suit)\s+(?:succeeded|succeeds|is\s+allowed|was\s+allowed)\b",
    r"\bthe\s+(?:court|judge|tribunal)\s+awarded\b",
    r"\b(?:holds|held|finds|found|concludes|concluded|determines|determined)\s+that\s+(?:the\s+defendant|he|she|it|they)\s+(?:is|was|were)\s+(?:in\s+breach|liable)\b",
    r"\b(?:liability|breach\s+of\s+(?:contract|fiduciary|duty|obligations?))[^.]*?(?:is|was|has\s+been)\s+(?:established|proven|made\s+out|found)\b"
]

FAILURE_PATTERNS_PLAINTIFF = [
    r"\bclaim(?: of the.*?)? (?:is|was) dismissed\b",
    r"\bclaims? (?:are|were) dismissed\b",
    r"\baction (?:is|was) dismissed\b"
]

SUCCESS_PATTERNS_DEFENDANT = [
    r"\bclaim was dismissed\b",
    r"\bcompany'?s claim was dismissed\b",
    r"\bplaintiff'?s claim was dismissed\b",
    r"\bjudgment was entered in favour of\b",
    r"\bsucceeded on (?:his|her|their) counterclaim\b",
]

DATE_ISO_DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")
DATE_ISO_MONTH = re.compile(r"^\d{4}-\d{2}$")
DATE_YEAR = re.compile(r"^\d{4}$")


def extract_json_from_text(text: str) -> Dict[str, Any]:
    if text is None:
        raise ValueError("Model returned empty content (None).")

    text = text.strip()
    if not text:
        raise ValueError("Model returned empty text.")

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"Model output did not contain valid JSON.\nRaw output:\n{text[:1000]}")
        snippet = text[start:end + 1].strip()
        return json.loads(snippet)


def build_prompt(source_text: str, row: Dict[str, Any]) -> str:
    return USER_PROMPT_TEMPLATE.format(
        source_text=source_text,
        json_block=json.dumps(row, ensure_ascii=False, indent=2),
    )


def normalize_input_data(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, dict):
        if "results" in data and isinstance(data["results"], list):
            return data["results"]
        return [data]
    if isinstance(data, list):
        return data
    raise ValueError("Unsupported JSON format")


def get_all_rows(data: Any) -> List[Dict[str, Any]]:
    return normalize_input_data(data)


def normalize_result(result: Dict[str, Any], row: Dict[str, Any]) -> Dict[str, Any]:
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
    result.setdefault("label_change", None)
    result.setdefault("fact_changes", [])
    result.setdefault("facts_to_remove", [])

    # sanitize malformed model output
    if not isinstance(result["fact_changes"], list):
        result["fact_changes"] = []
    else:
        result["fact_changes"] = [x for x in result["fact_changes"] if isinstance(x, dict)]

    if not isinstance(result["facts_to_remove"], list):
        result["facts_to_remove"] = []
    else:
        result["facts_to_remove"] = [x for x in result["facts_to_remove"] if isinstance(x, dict)]

    if result["label_change"] is not None and not isinstance(result["label_change"], dict):
        result["label_change"] = None

    if result["label_change"] or result["fact_changes"] or result["facts_to_remove"]:
        result["status"] = "CHANGES"

    bad_labels = {"defendant", "plaintiff", "third party", "third-party"}
    if result.get("label_change"):
        recommended = str(result["label_change"].get("recommended", "")).strip().lower()
        if recommended in bad_labels:
            result["label_change"]["recommended"] = row.get("Label", "Unknown")
            result["label_change"]["recommended_change"] = (
                "Rejected role-based label recommendation. Label must be an outcome label."
            )

    return result


def read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join([p.extract_text() or "" for p in reader.pages])


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def call_model(source_text: str, row: Dict[str, Any], retries: int = 2, backoff: int = 2):
    prompt = build_prompt(source_text, row)
    last_error = None

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            if content is None or not content.strip():
                raise ValueError("Model returned empty text.")
            return extract_json_from_text(content), response.usage

        except Exception as e:
            last_error = e
            if attempt < retries - 1:
                time.sleep(backoff * (2 ** attempt))
            else:
                raise RuntimeError(f"OpenAI call failed after {retries} attempts: {e}") from e

    raise RuntimeError(f"OpenAI call failed: {last_error}")


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


def get_value_by_path(obj: Any, path: str) -> Any:
    parts = path.split(".")
    current = obj

    for part in parts:
        if "[" in part and "]" in part:
            key = part[:part.index("[")]
            idx = int(part[part.index("[") + 1:part.index("]")])
            if key:
                current = current[key]
            current = current[idx]
        else:
            current = current[part]

    return current


def parse_fact_index(path: str) -> Optional[int]:
    marker = "Party_Details.Facts["
    if marker not in path:
        return None
    start = path.index(marker) + len(marker)
    end = path.index("]", start)
    return int(path[start:end])


def safe_search_any(patterns: List[str], text: str) -> bool:
    if not text:
        return False
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in patterns)


def is_nullish_text(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def is_valid_date_format(value: Any) -> bool:
    if value is None:
        return True
    if not isinstance(value, str):
        return False
    value = value.strip()
    if not value:
        return True
    return bool(DATE_ISO_DAY.match(value) or DATE_ISO_MONTH.match(value) or DATE_YEAR.match(value))


def normalize_date_or_blank(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def allowed_labels_for_role(role: str) -> set:
    role_l = (role or "").strip().lower()
    if role_l == "plaintiff":
        return PLAINTIFF_LABELS
    if role_l == "defendant":
        return DEFENDANT_LABELS
    return GENERIC_LABELS


def apply_audit_and_build_review(input_data: Any, audit_data: List[Dict[str, Any]]):
    corrected_rows = deepcopy(normalize_input_data(input_data))
    review_log = []

    row_lookup: Dict[str, Dict[str, Any]] = {}
    for row in corrected_rows:
        party_name = row.get("Party_Details", {}).get("Name", "")
        if party_name:
            row_lookup[party_name] = row

    for audit_row in audit_data:
        if not isinstance(audit_row, dict):
            print(f"  [WARN] Skipping malformed audit row: {audit_row}")
            continue

        party = audit_row.get("party", "")
        base_review = {
            "case_name": audit_row.get("case_name", ""),
            "party": party,
            "status": "UNMATCHED",
            "audit_suggestion": audit_row,
            "applied_changes": {
                "label_change": None,
                "fact_changes": [],
                "facts_removed": [],
                "rule_changes": [],
            },
        }

        if party not in row_lookup:
            review_log.append(base_review)
            continue

        target_row = row_lookup[party]
        applied_changes = {
            "label_change": None,
            "fact_changes": [],
            "facts_removed": [],
            "rule_changes": [],
        }

        # 1. Apply GPT label change first
        label_change = audit_row.get("label_change")
        if isinstance(label_change, dict) and label_change.get("recommended") is not None:
            old_label = target_row.get("Label")
            new_label = label_change["recommended"]
            if old_label != new_label:
                target_row["Label"] = new_label
                applied_changes["label_change"] = {
                    "from": old_label,
                    "to": new_label,
                    "source": "MODEL",
                }

        # 2. Apply GPT fact changes safely
        fact_changes = audit_row.get("fact_changes", [])
        if not isinstance(fact_changes, list):
            fact_changes = []

        for fc in fact_changes:
            if not isinstance(fc, dict):
                print(f"  [WARN] Skipping malformed fact_change for party '{party}': {fc}")
                continue

            path = fc.get("path")
            new_value = fc.get("recommended_value")

            if path and "Party_Details.Facts[" in path:
                try:
                    old_value = get_value_by_path(target_row, path)
                    if old_value != new_value:
                        set_value_by_path(target_row, path, new_value)
                        applied_changes["fact_changes"].append({
                            "path": path,
                            "from": old_value,
                            "to": new_value,
                            "source": "MODEL",
                        })
                except Exception as e:
                    print(f"  [WARN] Failed to apply fact_change for party '{party}', path='{path}': {e}")
                    continue

            elif path == "Label" and new_value is not None:
                old_value = target_row.get("Label")
                if old_value != new_value:
                    target_row["Label"] = new_value
                    applied_changes["label_change"] = {
                        "from": old_value,
                        "to": new_value,
                        "source": "MODEL",
                    }

        # 3. Apply GPT fact removals safely
        facts_to_remove = audit_row.get("facts_to_remove", [])
        if not isinstance(facts_to_remove, list):
            facts_to_remove = []

        indices_to_remove = []
        removal_text_by_index = {}
        facts = target_row.get("Party_Details", {}).get("Facts", [])

        for fr in facts_to_remove:
            if not isinstance(fr, dict):
                print(f"  [WARN] Skipping malformed facts_to_remove for party '{party}': {fr}")
                continue

            idx = parse_fact_index(fr.get("path", ""))
            if idx is not None:
                indices_to_remove.append(idx)
                if 0 <= idx < len(facts):
                    removal_text_by_index[idx] = facts[idx].get("Text")

        indices_to_remove = sorted(set(indices_to_remove), reverse=True)

        for idx in indices_to_remove:
            if 0 <= idx < len(facts):
                removed_text = facts[idx].get("Text")
                del facts[idx]
                applied_changes["facts_removed"].append({
                    "path": f"Party_Details.Facts[{idx}]",
                    "text_removed": removed_text,
                    "source": "MODEL",
                })

        review_log.append({
            "case_name": audit_row.get("case_name", ""),
            "party": party,
            "status": "CHANGED" if (
                applied_changes["label_change"]
                or applied_changes["fact_changes"]
                or applied_changes["facts_removed"]
            ) else "CLEAN",
            "audit_suggestion": audit_row,
            "applied_changes": applied_changes,
        })

    return corrected_rows, review_log


def append_rule_change(
    review_item: Dict[str, Any],
    rule_name: str,
    field: str,
    old_value: Any,
    new_value: Any,
    note: str,
):
    review_item["applied_changes"]["rule_changes"].append({
        "rule": rule_name,
        "field": field,
        "from": old_value,
        "to": new_value,
        "note": note,
        "source": "RULE",
    })
    if review_item["status"] == "CLEAN":
        review_item["status"] = "AUTO_FIXED"


def enforce_row_level_rules(
    corrected_rows: List[Dict[str, Any]],
    review_log: List[Dict[str, Any]],
):
    review_lookup = {item["party"]: item for item in review_log if item.get("party")}

    for row in corrected_rows:
        party = row.get("Party_Details", {}).get("Name", "")
        role = row.get("Party_Details", {}).get("Role", "")
        label = row.get("Label", "Unknown")
        review_item = review_lookup.setdefault(
            party,
            {
                "case_name": row.get("Metadata", {}).get("Case_Number", ""),
                "party": party,
                "status": "CLEAN",
                "audit_suggestion": None,
                "applied_changes": {
                    "label_change": None,
                    "fact_changes": [],
                    "facts_removed": [],
                    "rule_changes": [],
                },
            },
        )

        allowed = allowed_labels_for_role(role)
        if label not in allowed:
            old_label = label
            row["Label"] = "Unknown"
            if review_item["applied_changes"]["label_change"] is None:
                review_item["applied_changes"]["label_change"] = {
                    "from": old_label,
                    "to": "Unknown",
                    "source": "RULE",
                }
            append_rule_change(
                review_item,
                "INVALID_LABEL_BY_ROLE",
                "Label",
                old_label,
                "Unknown",
                f"Role '{role}' does not permit label '{old_label}'.",
            )

        facts = row.get("Party_Details", {}).get("Facts", [])
        to_delete = []
        hallucinated_deletes = {"remove this fact.", "remove this fact", "delete this fact.", "delete this fact"}

        for idx, fact in enumerate(facts):
            text = fact.get("Text", "")
            if is_nullish_text(text) or (isinstance(text, str) and text.strip().lower() in hallucinated_deletes):
                to_delete.append(idx)

        for idx in reversed(to_delete):
            del facts[idx]
            review_item["applied_changes"]["facts_removed"].append({
                "path": f"Party_Details.Facts[{idx}]",
                "source": "RULE",
            })
            append_rule_change(
                review_item,
                "NULL_OR_HALLUCINATED_FACT_TEXT",
                f"Party_Details.Facts[{idx}]",
                "null/empty/hallucination",
                "removed",
                "Removed fact with null text or literal LLM deletion command.",
            )

        facts = row.get("Party_Details", {}).get("Facts", [])
        leak_delete = []
        for idx, fact in enumerate(facts):
            text = fact.get("Text")
            if isinstance(text, str) and safe_search_any(LEAKAGE_PATTERNS, text):
                leak_delete.append(idx)
        for idx in reversed(leak_delete):
            old_text = facts[idx].get("Text")
            del facts[idx]
            review_item["applied_changes"]["facts_removed"].append({
                "path": f"Party_Details.Facts[{idx}]",
                "source": "RULE",
            })
            append_rule_change(
                review_item,
                "OBVIOUS_LEAKAGE_FACT",
                f"Party_Details.Facts[{idx}]",
                old_text,
                "removed",
                "Removed fact containing obvious judgment-stage leakage phrase.",
            )

        facts = row.get("Party_Details", {}).get("Facts", [])
        for idx, fact in enumerate(facts):
            old_date = fact.get("Fact_Date", "")
            if not is_valid_date_format(old_date):
                if old_date != "":
                    fact["Fact_Date"] = ""
                    review_item["applied_changes"]["fact_changes"].append({
                        "path": f"Party_Details.Facts[{idx}].Fact_Date",
                        "from": old_date,
                        "to": "",
                        "source": "RULE",
                    })
                    append_rule_change(
                        review_item,
                        "INVALID_DATE_FORMAT",
                        f"Party_Details.Facts[{idx}].Fact_Date",
                        old_date,
                        "",
                        "Normalized unsupported date format to blank.",
                    )

        leakage_regex = r"(?i)\b(?:the\s+)?(?:judge|court)\s+(?:held|found|concluded|determined|ruled|observed)(?:\s+that)?\s*"
        for field in ["Issue", "Rule", "Application"]:
            raw_text = row.get("Party_Details", {}).get(field, "")
            if raw_text and isinstance(raw_text, str):
                cleaned_text = re.sub(leakage_regex, "", raw_text)
                if cleaned_text and cleaned_text != raw_text:
                    cleaned_text = cleaned_text[0].upper() + cleaned_text[1:] if len(cleaned_text) > 0 else ""
                    row["Party_Details"][field] = cleaned_text

                    append_rule_change(
                        review_item,
                        "NARRATIVE_SCRUBBER",
                        f"Party_Details.{field}",
                        raw_text,
                        cleaned_text,
                        f"Scrubbed hindsight judicial language from {field}.",
                    )
    return corrected_rows, review_log


def group_case_keys(corrected_rows: List[Dict[str, Any]]) -> Dict[str, List[Tuple[int, Dict[str, Any]]]]:
    grouped = defaultdict(list)
    for idx, row in enumerate(corrected_rows):
        case_key = row.get("Metadata", {}).get("Case_Number", "").strip()
        grouped[case_key].append((idx, row))
    return grouped


def enforce_case_level_rules(
    corrected_rows: List[Dict[str, Any]],
    review_log: List[Dict[str, Any]],
):
    review_lookup = {item["party"]: item for item in review_log if item.get("party")}
    grouped = group_case_keys(corrected_rows)

    for case_key, members in grouped.items():
        plaintiffs = []
        defendants = []

        for _, row in members:
            role = row.get("Party_Details", {}).get("Role", "").strip().lower()
            if role == "plaintiff":
                plaintiffs.append(row)
            elif role == "defendant":
                defendants.append(row)

        plaintiff_labels = {row.get("Label", "Unknown") for row in plaintiffs}
        defendant_labels = {row.get("Label", "Unknown") for row in defendants}

        if plaintiffs and defendants:
            if plaintiff_labels == {"Claim Dismissed"} and defendant_labels == {"Liable"}:
                for row in defendants:
                    combined_text = " ".join(
                        str(row.get(k, "") or "") for k in ("Issue", "Rule", "Application", "Judicial_Reasoning_Log")
                    ).lower()

                    if "counterclaim" in combined_text:
                        print(f"  [Rule 5 Guard] Skipping '{row.get('Party_Details', {}).get('Name')}' due to potential counterclaim.")
                        continue

                    old_label = row.get("Label")
                    row["Label"] = "Not Liable"
                    review_item = review_lookup[row.get("Party_Details", {}).get("Name", "")]
                    review_item["applied_changes"]["label_change"] = {
                        "from": old_label,
                        "to": "Not Liable",
                        "source": "RULE",
                    }
                    append_rule_change(
                        review_item,
                        "UNANIMOUS_PLAINTIFF_DISMISSED",
                        "Label",
                        old_label,
                        "Not Liable",
                        "All plaintiffs were Claim Dismissed in this case group.",
                    )

        for row in plaintiffs:
            if row.get("Label") == "Unknown":
                combined_text = " ".join(
                    str(row.get(k, "") or "") for k in ("Issue", "Rule", "Application", "Judicial_Reasoning_Log")
                )
                if safe_search_any(SUCCESS_PATTERNS_PLAINTIFF, combined_text):
                    old_label = row["Label"]
                    row["Label"] = "Claim Allowed"
                    review_item = review_lookup[row.get("Party_Details", {}).get("Name", "")]
                    review_item["applied_changes"]["label_change"] = {
                        "from": old_label,
                        "to": "Claim Allowed",
                        "source": "RULE",
                    }
                    append_rule_change(
                        review_item,
                        "STRONG_PLAINTIFF_SUCCESS_SIGNAL",
                        "Label",
                        old_label,
                        "Claim Allowed",
                        "Plaintiff row contains strong success signal.",
                    )

        for row in defendants:
            if row.get("Label") == "Liable":
                combined_text = " ".join(
                    str(row.get(k, "") or "") for k in ("Issue", "Rule", "Application", "Judicial_Reasoning_Log")
                )
                if safe_search_any(SUCCESS_PATTERNS_DEFENDANT, combined_text):
                    old_label = row["Label"]
                    row["Label"] = "Not Liable"
                    review_item = review_lookup[row.get("Party_Details", {}).get("Name", "")]
                    review_item["applied_changes"]["label_change"] = {
                        "from": old_label,
                        "to": "Not Liable",
                        "source": "RULE",
                    }
                    append_rule_change(
                        review_item,
                        "STRONG_DEFENDANT_SUCCESS_SIGNAL",
                        "Label",
                        old_label,
                        "Not Liable",
                        "Defendant row contains strong success signal.",
                    )

        for row in plaintiffs:
            combined_text = " ".join(
                str(row.get(k, "") or "") for k in ("Issue", "Rule", "Application", "Judicial_Reasoning_Log")
            ).lower()

            if safe_search_any(FAILURE_PATTERNS_PLAINTIFF, combined_text):
                old_label = row.get("Label")
                if old_label in {"Unknown", "Claim Allowed"}:
                    row["Label"] = "Claim Dismissed"
                    review_item = review_lookup[row.get("Party_Details", {}).get("Name", "")]
                    review_item["applied_changes"]["label_change"] = {
                        "from": old_label,
                        "to": "Claim Dismissed",
                        "source": "RULE",
                    }
                    append_rule_change(
                        review_item,
                        "STRONG_PLAINTIFF_DISMISSAL_SIGNAL",
                        "Label",
                        old_label,
                        "Claim Dismissed",
                        "Overriding label due to explicit dismissal text in judgment.",
                    )

        case_leaky_texts = set()
        for _, row in members:
            party_name = row.get("Party_Details", {}).get("Name", "")
            review_item = review_lookup.get(party_name, {})
            for fr in review_item.get("applied_changes", {}).get("facts_removed", []):
                if "text_removed" in fr and fr["text_removed"]:
                    case_leaky_texts.add(fr["text_removed"])

        for _, row in members:
            party_name = row.get("Party_Details", {}).get("Name", "")
            review_item = review_lookup.get(party_name, {})
            facts = row.get("Party_Details", {}).get("Facts", [])
            to_delete = []

            for idx, fact in enumerate(facts):
                if fact.get("Text") in case_leaky_texts:
                    to_delete.append(idx)

            for idx in reversed(to_delete):
                old_text = facts[idx].get("Text")
                del facts[idx]
                review_item["applied_changes"]["facts_removed"].append({
                    "path": f"Party_Details.Facts[{idx}]",
                    "source": "RULE",
                })
                append_rule_change(
                    review_item,
                    "CROSS_PARTY_LEAKAGE_SYNC",
                    f"Party_Details.Facts[{idx}]",
                    old_text,
                    "removed",
                    "Fact was flagged as leakage in another party's row and purged globally.",
                )
    return corrected_rows, review_log


def finalize_review_status(review_log: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for item in review_log:
        applied = item.get("applied_changes", {})
        if item.get("status") == "UNMATCHED":
            continue
        changed = bool(
            applied.get("label_change")
            or applied.get("fact_changes")
            or applied.get("facts_removed")
            or applied.get("rule_changes")
        )
        item["status"] = "AUTO_FIXED" if changed else "UNCHANGED"
    return review_log


def process_batch(input_dir: Path, output_dir: Path):
    json_files = [
        p for p in input_dir.rglob("*.json")
        if not p.name.endswith(".corrected.json")
        and not p.name.endswith(".review.json")
        and not p.name.endswith(".audit.json")
        and not p.name.endswith(".changes.json")
    ]

    if not json_files:
        raise Exception(f"No JSON files found in {input_dir}")

    total_tokens = 0
    output_dir.mkdir(parents=True, exist_ok=True)

    for json_path in json_files:
        corrected_path = output_dir / f"{json_path.stem}.corrected.json"
        if corrected_path.exists():
            print(f"Skipping {json_path.name} (already processed)")
            continue

        pdf_path = json_path.with_suffix(".pdf")
        if not pdf_path.exists():
            print(f"Skipping {json_path.name} (no PDF)")
            continue

        print(f"\nProcessing {json_path.name}...")

        raw_data = read_json(json_path)
        rows = get_all_rows(raw_data)
        if not rows:
            print(f"  No rows found in {json_path.name}")
            continue

        pdf_text = read_pdf(pdf_path)
        audit_results = []

        for i, row in enumerate(rows, start=1):
            try:
                party_name = row.get("Party_Details", {}).get("Name", f"row_{i}")
                print(f"  Auditing row {i}/{len(rows)}: {party_name}...")
                result, usage = call_model(pdf_text, row)
                result = normalize_result(result, row)
                audit_results.append(result)
                total_tokens += getattr(usage, "total_tokens", 0) or 0
            except Exception as e:
                print(f"  Skipping row {i}/{len(rows)} due to error: {e}")

        corrected_data, review_log = apply_audit_and_build_review(raw_data, audit_results)
        corrected_data, review_log = enforce_row_level_rules(corrected_data, review_log)
        corrected_data, review_log = enforce_case_level_rules(corrected_data, review_log)

        with open(corrected_path, "w", encoding="utf-8") as f:
            json.dump(corrected_data, f, indent=2, ensure_ascii=False)

        print(f"  Saved: {corrected_path.name}")

    print("\nDONE")
    print(f"Total tokens used: {total_tokens}")


def main():
    print(" Running audit + correction mode")
    print(f" Model: {MODEL_NAME}")

    INPUT_FOLDER = Path("Data/PDFsandJsonInput")
    OUTPUT_FOLDER = Path("Data/Processed/FinalAudited")

    process_batch(INPUT_FOLDER, OUTPUT_FOLDER)


if __name__ == "__main__":
    main()