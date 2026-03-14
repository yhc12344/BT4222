import os
import json
from pathlib import Path

import pdfplumber
from dotenv import load_dotenv
from openai import OpenAI

# =========================
# CONFIG
# =========================

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

INPUT_FOLDER = Path("Data/Test")
OUTPUT_FOLDER = Path("Data/Processed/MultiPrompts_Final")
DEBUG_FOLDER = Path("Data/Debug")

INPUT_FOLDER.mkdir(parents=True, exist_ok=True)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
DEBUG_FOLDER.mkdir(parents=True, exist_ok=True)

LEAK_WORDS = [
    "liable",
    "not liable",
    "appeal allowed",
    "appeal dismissed",
    "convicted",
    "sentence imposed",
    "damages awarded",
    "the court held",
    "the court found",
    "the judge concluded",
    "judgment entered",
    "ordered that",
    "breach established",
    "claim dismissed"
]

# =========================
# PROMPTS
# =========================

PARTY_PROMPT = """
You are a legal data extraction engine.

Your job is to identify ALL individual parties from a Singapore judgment.

CRITICAL RULES:
1. If the case title contains words like "and Others" or "and another", you MUST look at the party listing in the judgment header and recover the full individual party names.
2. Do NOT merge multiple plaintiffs into one row.
3. Do NOT merge multiple defendants into one row.
4. Normalize roles as:
   - Plaintiff = plaintiff / appellant / applicant / claimant
   - Defendant = defendant / respondent

Return JSON only in this exact format:

{
  "parties": [
    {
      "name": "string",
      "role": "Plaintiff | Defendant"
    }
  ]
}

Judgment text:
"""

EXTRACTION_PROMPT = """
You are a Senior Legal Data Engineer specializing in extracting structured legal datasets from Singapore court judgments.

Your task is to convert a Singapore Judgment Summary into structured JSON following the exact schema below.

The dataset will be used for machine learning research that predicts legal outcomes. Therefore the factual narrative must remain verdict-blind except for the dedicated Conclusion field.

--------------------------------------------------

OUTPUT FORMAT

Return ONLY valid JSON.

The output must be a JSON array of objects.

Do NOT include explanations, markdown, comments, or text outside JSON.

--------------------------------------------------

PARTY ROW GENERATION

Create a NEW JSON object for every party whose conduct is discussed in the judgment summary.

If there are multiple Plaintiffs or multiple Defendants, generate one object per party.

Example:

ABB Holdings Pte Ltd
ABB Installation Materials (East Asia) Pte Ltd
ABB Industry Pte Ltd

These must produce THREE separate rows.

Do NOT merge multiple parties into a single object.

If the case title contains "and Others" or "and another", identify the full party list from the judgment header and create separate objects for each party.

Each object must contain BOTH:

Metadata
Party_Details

--------------------------------------------------

FACT DISTILLATION PROCESS

Judgments often contain long narrative paragraphs. You must convert them into short factual statements suitable for machine learning.

Internally follow these steps:

Step 1 - Identify Raw Facts
Locate sentences describing actions, roles, transactions, relationships, employment, communications, or events.

Step 2 - Distill Facts
Rewrite the facts into short, clean, atomic sentences.

Rules for distillation:

- Each fact must contain one event or relationship only
- Remove legal commentary or reasoning
- Remove references to court opinions
- Convert complex sentences into simple factual statements
- Attribute each fact to the specific party in that row

Example transformation:

Original text:
"The defendant, who had been employed by the plaintiff for several years, contacted a competitor and disclosed confidential pricing information."

Distilled facts:

Fact 1:
"The defendant was employed by the plaintiff."

Fact 2:
"The defendant contacted a competitor."

Fact 3:
"The defendant disclosed confidential pricing information."

Only output the distilled facts.

--------------------------------------------------

FACT ISOLATION

The "Facts" field must contain ONLY:

- observable actions
- events
- factual conduct
- employment roles
- business relationships
- corporate structures
- communications
- transactions
- responsibilities

Facts must describe events attributable to the specific party represented in that row.

Facts must NOT contain:

- judicial reasoning
- court analysis
- interpretations of law
- conclusions about liability
- references to what the court held

Facts must NOT include wording such as:

held
found
concluded
liable
not liable
judgment
appeal allowed
appeal dismissed
damages awarded

Facts must contain ONLY observable factual information.

--------------------------------------------------

FACT STRUCTURE

The "Facts" field must be an ARRAY of OBJECTS.

Each fact must contain:

Fact_Type
Text

Example:

{
  "Fact_Type": "EMPLOYMENT",
  "Text": "The defendant was employed by the plaintiff as a regional sales manager."
}

Each fact must be a single sentence.

Extract between 4 and 12 facts if the judgment contains sufficient information.

Facts must NOT be empty unless absolutely impossible.

--------------------------------------------------

ISSUE

The Issue field should describe the legal question or dispute involving this party.

It must not contain the final judgment.

Example:

"Whether the defendant breached fiduciary duties owed to the plaintiff."

--------------------------------------------------

RULE

The Rule field should describe the legal rule or doctrine referenced by the court that applies to the issue.

Examples:

"Directors owe fiduciary duties of loyalty and good faith to their company."

"Employees must not misuse confidential information obtained during employment."

--------------------------------------------------

APPLICATION

The Application field should summarize how the legal rule was applied to the factual circumstances involving the party.

It may reference conduct but must NOT reveal the final judgment.

Example:

"The court examined whether the defendant's communication with a competitor and use of confidential information breached fiduciary obligations."

--------------------------------------------------

CONCLUSION

The Conclusion field MUST contain the final decision relating to the party.

Examples:

"Appeal dismissed."

"Judgment entered for the plaintiff."

"Defendant found liable for breach of fiduciary duty."

"Claim dismissed."

Conclusion must contain the court's outcome.

--------------------------------------------------

METADATA

Metadata fields describe the case itself.

Extract if present in the judgment summary.

Fields:

Case_ID
Case_Name
Judge
Date
Tribunal_Court

If information is unavailable, return null.

--------------------------------------------------

JSON SCHEMA

Return JSON exactly in this structure:

[
  {
    "Metadata": {
      "Case_ID": "string",
      "Case_Name": "string",
      "Judge": "string or null",
      "Date": "YYYY-MM-DD or null",
      "Tribunal_Court": "string or null"
    },
    "Party_Details": {
      "Role": "Plaintiff | Defendant",
      "Name": "string",
      "Facts": [
        {
          "Fact_Type": "PARTY_INFO | EMPLOYMENT | BUSINESS_ACTIVITY | RELATIONSHIP | CONDUCT | COMMUNICATION | TRANSACTION | CORPORATE_STRUCTURE | DUTY | OTHER",
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

--------------------------------------------------

FINAL INSTRUCTIONS

Facts must remain verdict-blind.

Conclusion must contain the judgment outcome.

Each party must produce a separate JSON object.

Return only valid JSON.

--------------------------------------------------

Case_ID: __CASE_ID__
Case_Name: __CASE_NAME__
Target Party Name: __PARTY_NAME__
Target Party Role: __PARTY_ROLE__

Judgment Summary:
"""

# =========================
# HELPERS
# =========================

def extract_pdf_text(pdf_path: Path) -> str:
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += (page.extract_text() or "") + "\n"
    return " ".join(text.split())


def remove_verdict_sections(text: str) -> str:
    stop_markers = [
        "copyright © government of singapore"
    ]

    lower_text = text.lower()
    cut_positions = []

    for marker in stop_markers:
        idx = lower_text.find(marker)
        if idx != -1:
            cut_positions.append(idx)

    if cut_positions:
        return text[:min(cut_positions)]

    return text


def safe_json_parse(raw_output: str):
    raw_output = raw_output.strip()

    try:
        return json.loads(raw_output)
    except Exception:
        pass

    start_obj = raw_output.find("{")
    end_obj = raw_output.rfind("}") + 1

    start_arr = raw_output.find("[")
    end_arr = raw_output.rfind("]") + 1

    candidates = []
    if start_obj != -1 and end_obj > start_obj:
        candidates.append(raw_output[start_obj:end_obj])
    if start_arr != -1 and end_arr > start_arr:
        candidates.append(raw_output[start_arr:end_arr])

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception:
            continue

    raise ValueError(f"Could not parse JSON. Raw output starts with:\n{raw_output[:1200]}")


def contains_leakage(facts_list) -> bool:
    if not isinstance(facts_list, list):
        return False

    combined_text = " ".join(
        fact.get("Text", "").lower()
        for fact in facts_list
        if isinstance(fact, dict)
    )

    return any(word in combined_text for word in LEAK_WORDS)


def facts_too_weak(facts_list) -> bool:
    if not isinstance(facts_list, list):
        return True

    usable = [
        fact for fact in facts_list
        if isinstance(fact, dict) and fact.get("Text", "").strip()
    ]

    return len(usable) == 0


# =========================
# STAGE 1: PARTY EXTRACTION
# =========================

def detect_parties(text: str):
    response = client.chat.completions.create(
        model="gpt-5.1",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "developer",
                "content": "Extract all individual parties from the judgment header. Output valid JSON only."
            },
            {
                "role": "user",
                "content": PARTY_PROMPT + text
            }
        ]
    )

    raw_output = response.choices[0].message.content

    with open(DEBUG_FOLDER / "party_outputs.txt", "a", encoding="utf-8") as f:
        f.write(raw_output + "\n\n")

    data = safe_json_parse(raw_output)
    parties = data.get("parties", [])

    seen = set()
    clean_parties = []
    for party in parties:
        name = party.get("name", "").strip()
        role = party.get("role", "").strip()
        key = (name, role)

        if name and key not in seen:
            seen.add(key)
            clean_parties.append({
                "name": name,
                "role": role
            })

    return clean_parties


# =========================
# STAGE 2: PARTY-SPECIFIC EXTRACTION
# =========================

def extract_for_party(text: str, party: dict, case_name: str, case_id: str):
    prompt = (
        EXTRACTION_PROMPT
        .replace("__CASE_ID__", case_id)
        .replace("__CASE_NAME__", case_name)
        .replace("__PARTY_NAME__", party["name"])
        .replace("__PARTY_ROLE__", party["role"])
    )

    response = client.chat.completions.create(
        model="gpt-5.1",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "developer",
                "content": "Extract structured legal data for exactly one target party. Output valid JSON only."
            },
            {
                "role": "user",
                "content": prompt + text
            }
        ]
    )

    raw_output = response.choices[0].message.content

    with open(DEBUG_FOLDER / "extraction_outputs.txt", "a", encoding="utf-8") as f:
        f.write(raw_output + "\n\n")

    data = safe_json_parse(raw_output)
    return data


def retry_if_facts_empty(text: str, party: dict, case_name: str, case_id: str, first_result):
    if isinstance(first_result, list) and len(first_result) > 0:
        obj = first_result[0]
    else:
        obj = first_result

    facts = obj.get("Party_Details", {}).get("Facts", [])

    if not facts_too_weak(facts):
        return first_result

    retry_prompt = f"""
The previous extraction for this target party returned empty or weak facts.

Retry the extraction and focus on distilling factual sentences attributable to this target party.

You must return at least 4 fact objects if enough factual material exists in the judgment.

Keep Facts verdict-blind.
Keep Conclusion as the judgment outcome.

Return valid JSON only in the same schema.

Case_ID: {case_id}
Case_Name: {case_name}
Target Party Name: {party['name']}
Target Party Role: {party['role']}

Judgment Summary:
{text}
"""

    response = client.chat.completions.create(
        model="gpt-5.1",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "developer",
                "content": "Retry the extraction. Facts must not be empty if the source contains enough facts."
            },
            {
                "role": "user",
                "content": retry_prompt
            }
        ]
    )

    raw_output = response.choices[0].message.content

    with open(DEBUG_FOLDER / "retry_outputs.txt", "a", encoding="utf-8") as f:
        f.write(raw_output + "\n\n")

    return safe_json_parse(raw_output)


# =========================
# NORMALIZATION
# =========================

def normalize_party_result(result, party, case_name, case_id):
    if isinstance(result, list):
        if not result:
            raise ValueError("Empty list returned")
        obj = result[0]
    else:
        obj = result

    if "Metadata" not in obj:
        obj["Metadata"] = {}

    if "Party_Details" not in obj:
        obj["Party_Details"] = {}

    obj["Metadata"]["Case_ID"] = case_id
    obj["Metadata"]["Case_Name"] = case_name

    obj["Party_Details"]["Role"] = party["role"]
    obj["Party_Details"]["Name"] = party["name"]

    obj["Party_Details"].setdefault("Facts", [])
    obj["Party_Details"].setdefault("Issue", None)
    obj["Party_Details"].setdefault("Rule", None)
    obj["Party_Details"].setdefault("Application", None)
    obj["Party_Details"].setdefault("Conclusion", None)

    return obj


# =========================
# MAIN PIPELINE
# =========================

def process_pdf(pdf_path: Path):
    print(f"Processing {pdf_path.name}")

    text = extract_pdf_text(pdf_path)
    text = remove_verdict_sections(text)

    case_name = pdf_path.stem
    case_id = pdf_path.stem.replace(" ", "_")

    parties = detect_parties(text)
    print("Detected parties:", parties)

    if not parties:
        print("No parties detected.")
        return []

    rows = []

    for party in parties:
        try:
            result = extract_for_party(text, party, case_name, case_id)
            result = retry_if_facts_empty(text, party, case_name, case_id, result)
            obj = normalize_party_result(result, party, case_name, case_id)

            facts = obj.get("Party_Details", {}).get("Facts", [])

            if contains_leakage(facts):
                print(f"Leakage detected in facts for {party['name']}. Skipping.")
                continue

            if facts_too_weak(facts):
                print(f"Weak or empty facts for {party['name']}. Skipping.")
                continue

            rows.append(obj)

        except Exception as e:
            print(f"Extraction error for {party['name']}: {e}")

    return rows


# =========================
# MAIN LOOP
# =========================

def main():
    pdf_files = list(INPUT_FOLDER.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found.")
        return

    for pdf_file in pdf_files:
        rows = process_pdf(pdf_file)

        if rows:
            output_path = OUTPUT_FOLDER / f"{pdf_file.stem}.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(rows, f, indent=2, ensure_ascii=False)
            print(f"Saved {len(rows)} party rows.")
        else:
            print("No clean rows extracted.")


if __name__ == "__main__":
    main()