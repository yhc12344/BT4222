import os
import json
from pathlib import Path
import pdfplumber
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("KEY"))

# Folder paths
INPUT_FOLDER = Path("Data/Test")
OUTPUT_FOLDER = Path("Data/Processed")

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# Your extraction prompt
PROMPT_TEMPLATE = """
You are a Senior Legal Data Engineer specializing in extracting structured data from Singapore legal judgments.

Your task is to convert a Singapore Judgment Summary into structured JSON following the schema below.

Return ONLY valid JSON.

Schema:
[
  {
    "Metadata": {
      "Judge": "string | null",
      "Date": "YYYY-MM-DD | null",
      "Hearing_Duration": "string | null",
      "Tribunal_Court": "string | null",
      "Sector": "string | null",
      "Lawyers": ["string"]
    },
    "Party_Details": {
      "Role": "Plaintiff | Defendant",
      "Name": "string",
      "Facts": "string",
      "Issue": "string | null",
      "Rule": "string | null",
      "Application": "string | null",
      "Conclusion": "string | null"
    }
  }
]

Judgment Summary:
"""

def extract_pdf_text(pdf_path):
    """Extract text from PDF"""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text


def process_pdf(pdf_path):
    """Send PDF text to GPT-4o and get JSON"""
    
    text = extract_pdf_text(pdf_path)

    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        messages=[
            {"role": "system", "content": "You extract structured legal data."},
            {"role": "user", "content": PROMPT_TEMPLATE + text}
        ],
    )

    return response.choices[0].message.content


def main():
    for pdf_file in INPUT_FOLDER.glob("*.pdf"):
        
        print(f"Processing {pdf_file.name}")

        try:
            result_json = process_pdf(pdf_file)

            output_file = OUTPUT_FOLDER / (pdf_file.stem + ".json")

            # Validate JSON before saving
            parsed = json.loads(result_json)

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(parsed, f, indent=2)

            print(f"Saved -> {output_file}")

        except Exception as e:
            print(f"Error processing {pdf_file.name}: {e}")


if __name__ == "__main__":
    main()