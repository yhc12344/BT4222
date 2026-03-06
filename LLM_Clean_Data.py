import pdfplumber
import re
import os
import json
import pandas as pd
# import openai 
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_irac_from_llm(text_chunk):
    """Sends text to an LLM to extract IRAC components in JSON format."""
    prompt = f"""
    You are a legal analyst specializing in Singapore Corporate Law. 
    Analyze the following judgment excerpt and extract the IRAC components:
    - ISSUE: The core legal question (e.g., breach of fiduciary duty).
    - RULE: The specific statutes (e.g., Section 157 CA) or legal tests used.
    - APPLICATION: How the facts of the director's behavior were applied to the rule.
    - CONCLUSION: The final ruling (Success/Dismissed and the remedy).

    Return ONLY a valid JSON object with the keys: "Issue", "Rule", "Application", "Conclusion".

    TEXT:
    {text_chunk}
    """
    
    # --- OPENAI APPROACH (Commented out) ---
    # response = openai.chat.completions.create(
    #     model="gpt-4o",
    #     messages=[{"role": "user", "content": prompt}],
    #     response_format={ "type": "json_object" }
    # )
    # return json.loads(response.choices[0].message.content)
    
    # --- GEMINI APPROACH ---
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not found. Please add it to your .env file.")
        return {"Issue": "Error", "Rule": "Error", "Application": "Error", "Conclusion": "Error"}
        
    genai.configure(api_key=api_key)
    
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            temperature=0.1
        )
    )
    
    try:
        return json.loads(response.text)
    except json.JSONDecodeError:
        print("Failed to decode JSON from Gemini. Raw response:")
        print(response.text)
        return {"Issue": "Error", "Rule": "Error", "Application": "Error", "Conclusion": "Error"}

def process_judgments_with_irac(data_folder, output_file):
    results = []
    
    for filename in os.listdir(data_folder):
        if filename.endswith(".pdf"):
            path = os.path.join(data_folder, filename)
            print(f"Processing: {filename}")
            
            with pdfplumber.open(path) as pdf:
                # To save tokens, we extract the first 3 pages (Issue/Facts) 
                # and the last 2 pages (Conclusion)
                pages = pdf.pages
                start_text = "\n".join([p.extract_text() for p in pages[:3] if p.extract_text()])
                end_text = "\n".join([p.extract_text() for p in pages[-2:] if p.extract_text()])
                
                combined_context = start_text + "\n" + end_text
                
                try:
                    # Extract IRAC metadata via LLM
                    irac_metadata = get_irac_from_llm(combined_context)
                    
                    # Merge with basic file info
                    record = {
                        "File_Name": filename,
                        **irac_metadata
                    }
                    results.append(record)
                    print(f"SUCCESS! Extracted Data:\n{json.dumps(irac_metadata, indent=2)}")
                except Exception as e:
                    print(f"Error calling LLM for {filename}: {e}")
                    
            # Stop after 1 document for testing purposes
            print("\nTest complete! Breaking early.")
            break

    # Save as CSV for your BT4222 interim report dataset [cite: 3461]
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"Saved {len(results)} IRAC-encoded records to {output_file}")

if __name__ == '__main__':
    os.makedirs("Data/Cleaned_Data", exist_ok=True)
    process_judgments_with_irac("Data/PDFs", "Data/Cleaned_Data/LLM_extracted_features.csv")