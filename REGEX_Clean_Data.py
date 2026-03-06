import pdfplumber
import re
import pandas as pd
import os

def extract_judgment_sections(pdf_path):
    extracted_data = {
        "File_Name": os.path.basename(pdf_path),
        "CatchWords_PDF": "Not Found",
        "Conclusion_Text": "Not Found",
        "Full_Text_Length": 0
    }
    
    try:
        # 1. Extract all text from the PDF
        full_text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
                    
        extracted_data["Full_Text_Length"] = len(full_text)
        
        # 2. Extract CatchWords
        # Singapore judgments typically put Catchwords in square brackets near the top.
        # This regex avoids greediness by matching any character EXCEPT a closing bracket
        catchwords_match = re.search(r'\[\s*(Companies|Contract|Tort|Insolvency)[^\]]*\]', full_text, re.IGNORECASE)
        if catchwords_match:
            # Clean up newlines within the catchwords
            clean_catchwords = re.sub(r'\s+', ' ', catchwords_match.group(0))
            extracted_data["CatchWords_PDF"] = clean_catchwords
            
        # 3. Extract the Conclusion
        # Looks for the word "Conclusion" (often preceded by a number like "V." or "12.") 
        # and captures everything after it until common sign-offs or the end of the document.
        conclusion_pattern = re.compile(
            r'(?:^|\n)\s*(?:[IVX]+|\d+)?\.?\s*Conclusion\s*\n(.*?)(?=\n\s*Dated this|\n\s*Judge|\n\s*Judicial Commissioner|\n\s*Chief Justice|\Z)', 
            re.IGNORECASE | re.DOTALL
        )
        conclusion_match = conclusion_pattern.search(full_text)
        
        if conclusion_match:
            clean_conclusion = re.sub(r'\s+', ' ', conclusion_match.group(1)).strip()
            extracted_data["Conclusion_Text"] = clean_conclusion
            
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        
    return extracted_data

def process_all_judgments(data_folder="Data/PDFs", output_csv="Data/Cleaned_Data/REGEX_extracted_features.csv"):
    results = []
    print(f"Scanning directory: {data_folder}...")
    
    if not os.path.exists(data_folder):
        print(f"Directory {data_folder} not found.")
        return
        
    for filename in os.listdir(data_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(data_folder, filename)
            print(f"Extracting features from: {filename}")
            extracted_data = extract_judgment_sections(pdf_path)
            results.append(extracted_data)
            
    if results:
        df = pd.DataFrame(results)
        
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        
        df.to_csv(output_csv, index=False)
        print(f"\nSuccessfully extracted features from {len(results)} documents.")
        print(f"Saved to: {output_csv}")
    else:
        print("No PDFs found to process.")

if __name__ == '__main__':
    process_all_judgments()