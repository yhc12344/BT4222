import json
import re
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

load_dotenv()
client = OpenAI()

MODEL_NAME = "gpt-5.4-mini"

SYSTEM_PROMPT = """
You are a legal ML audit engine.

Audit one party row against the source judgment text.

Check only:
1. factual inaccuracies,
2. facts that contain post-dispute, litigation-stage, or judgment-stage leakage,
3. whether the OUTCOME LABEL is correct.

IMPORTANT:
- Party_Details.Role is metadata, not the prediction label.
- The label must be the OUTCOME for that row, such as:
  "Liable", "Not Liable", "Partially Liable", "Claim Allowed", "Claim Dismissed", or "Unknown".
- Do NOT recommend "Defendant", "Plaintiff", or "Third Party" as a label.
- If the judgment does not clearly determine the row's final outcome, keep the label as "Unknown".
- Check whether Fact_Date values are supported by the judgment.
- If the judgment gives only a year or approximate period, do not invent an exact day or month.

Return only valid JSON with this shape:
{{
  "case_name": string,
  "party": string,
  "status": "CLEAN" | "CHANGES",
  "label_change": null | {{
    "current": string,
    "recommended": string,
    "evidence": string,
    "recommended_change": string
  }},
  "fact_changes": [
    {{
      "path": string,
      "issue_type": string,
      "current_value": string,
      "recommended_value": string,
      "evidence": string,
      "recommended_change": string
    }}
  ],
  "facts_to_remove": [
    {{
      "path": string,
      "issue_type": string,
      "current_value": string,
      "evidence": string,
      "recommended_change": string
    }}
  ]
}}
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
- Party_Details.Role is metadata, not the prediction label.
- The label to audit is the top-level "Label" field.
- The label must refer to the row's OUTCOME, not litigation role.
- Do not recommend "Defendant", "Plaintiff", or "Third Party" as label values.
- If the judgment only concerns procedural permission, appeals, amendment applications, or third-party proceedings without deciding this row's substantive outcome, keep Label as "Unknown".
- Return only changed items.

If nothing needs changing, return:
{{
  "case_name": "",
  "party": "",
  "status": "CLEAN",
  "label_change": null,
  "fact_changes": [],
  "facts_to_remove": []
}}
""".strip()


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
        if not snippet:
            raise ValueError(f"Extracted JSON snippet was empty.\nRaw output:\n{text[:1000]}")

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

    result["case_name"] = result.get("case_name") or metadata.get("Case_Name") or metadata.get("Case_Number") or ""
    result["party"] = party_details.get("Name", "")
    result.setdefault("status", "CLEAN")
    result.setdefault("label_change", None)
    result.setdefault("fact_changes", [])
    result.setdefault("facts_to_remove", [])

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
    text_parts = []

    for page in reader.pages:
        t = page.extract_text()
        if t:
            text_parts.append(t)

    return "\n".join(text_parts)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def call_model(source_text: str, row: Dict[str, Any], retries: int = 3, backoff: int = 2):
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

            choice = response.choices[0]
            msg = choice.message
            content = msg.content

            if content is None or not content.strip():
                raise ValueError(
                    f"Model returned empty text. "
                    f"finish_reason={getattr(choice, 'finish_reason', None)}, "
                    f"refusal={repr(getattr(msg, 'refusal', None))}"
                )

            return extract_json_from_text(content), response.usage

        except Exception as e:
            last_error = e

            if attempt == retries - 1:
                try:
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0,
                    )

                    choice = response.choices[0]
                    msg = choice.message
                    content = msg.content

                    if content is None or not content.strip():
                        raise ValueError(
                            f"Model returned empty text. "
                            f"finish_reason={getattr(choice, 'finish_reason', None)}, "
                            f"refusal={repr(getattr(msg, 'refusal', None))}"
                        )

                    return extract_json_from_text(content), response.usage

                except Exception as fallback_error:
                    raise RuntimeError(
                        f"OpenAI call failed after {retries} attempts: {fallback_error}"
                    ) from fallback_error

            time.sleep(backoff * (2 ** attempt))

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


def apply_audit_with_tracking(input_data: Any, audit_data: List[Dict[str, Any]]):
    corrected_rows = deepcopy(normalize_input_data(input_data))
    changes_log = []

    row_lookup = {}
    for row in corrected_rows:
        party_name = row.get("Party_Details", {}).get("Name", "")
        if party_name:
            row_lookup[party_name] = row

    for audit_row in audit_data:
        party = audit_row.get("party", "")
        if party not in row_lookup:
            change_entry = {
                "case_name": audit_row.get("case_name", ""),
                "party": party,
                "status": "UNMATCHED",
                "label_change": None,
                "fact_changes": [],
                "facts_removed": []
            }
            changes_log.append(change_entry)
            continue

        target_row = row_lookup[party]

        change_entry = {
            "case_name": audit_row.get("case_name", ""),
            "party": party,
            "label_change": None,
            "fact_changes": [],
            "facts_removed": []
        }

        label_change = audit_row.get("label_change")
        if label_change and label_change.get("recommended") is not None:
            old_label = target_row.get("Label")
            new_label = label_change["recommended"]

            if old_label != new_label:
                target_row["Label"] = new_label
                change_entry["label_change"] = {
                    "from": old_label,
                    "to": new_label
                }

        for fc in audit_row.get("fact_changes", []):
            path = fc.get("path")
            new_value = fc.get("recommended_value")

            if path and "Party_Details.Facts[" in path:
                try:
                    old_value = get_value_by_path(target_row, path)
                    if old_value != new_value:
                        set_value_by_path(target_row, path, new_value)
                        change_entry["fact_changes"].append({
                            "path": path,
                            "from": old_value,
                            "to": new_value
                        })
                except Exception:
                    continue
            elif path == "Label" and new_value is not None:
                old_value = target_row.get("Label")
                if old_value != new_value:
                    target_row["Label"] = new_value
                    change_entry["label_change"] = {
                        "from": old_value,
                        "to": new_value
                    }

        indices_to_remove = []
        for fr in audit_row.get("facts_to_remove", []):
            idx = parse_fact_index(fr.get("path", ""))
            if idx is not None:
                indices_to_remove.append(idx)

        indices_to_remove = sorted(set(indices_to_remove), reverse=True)

        facts = target_row.get("Party_Details", {}).get("Facts", [])
        for idx in indices_to_remove:
            if 0 <= idx < len(facts):
                del facts[idx]
                change_entry["facts_removed"].append(f"Party_Details.Facts[{idx}]")

        change_entry["status"] = (
            "CHANGED" if (
                change_entry["label_change"] or
                change_entry["fact_changes"] or
                change_entry["facts_removed"]
            ) else "CLEAN"
        )

        changes_log.append(change_entry)

    return corrected_rows, changes_log


def process_batch(input_dir: Path, output_dir: Path):
    json_files = [
        p for p in input_dir.rglob("*.json")
        if not p.name.endswith(".corrected.json")
        and not p.name.endswith(".changes.json")
        and not p.name.endswith(".audit.json")
    ]

    if not json_files:
        raise Exception(f"No JSON files found in {input_dir}")

    total_tokens = 0

    for json_path in json_files:
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

        corrected_data, changes_log = apply_audit_with_tracking(raw_data, audit_results)

        output_path = output_dir / f"{json_path.stem}.corrected.json"
        changes_path = output_dir / f"{json_path.stem}.changes.json"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(corrected_data, f, indent=2, ensure_ascii=False)

        with open(changes_path, "w", encoding="utf-8") as f:
            json.dump(changes_log, f, indent=2, ensure_ascii=False)

        print(f"  Saved: {output_path.name}")
        print(f"  Saved: {changes_path.name}")

    print("\n✅ DONE")
    print(f"Total tokens used: {total_tokens}")


def main():
    print("🚀 Running correction mode")
    print(f"🤖 Model: {MODEL_NAME}")

    INPUT_FOLDER = Path("Data/PDFsandJsonInput")
    OUTPUT_FOLDER = Path("Data/Processed/FinalAudited")

    INPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    process_batch(INPUT_FOLDER, OUTPUT_FOLDER)


if __name__ == "__main__":
    main()