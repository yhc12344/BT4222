import json
import pandas as pd
from itertools import product
from pathlib import Path

# 1. Define the target directory containing your JSON files
# Using raw string (r"...") to handle Windows backslashes safely
data_dir = Path(r"data\processed\FinalAudited")

rows = []

# 2. Iterate through every .json file in the directory
for filepath in data_dir.glob("*.json"):
    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"Skipping {filepath.name}: Invalid JSON format.")
            continue

    # 3. Separate plaintiffs and defendants for the current file
    plaintiffs = [item for item in data if item.get('Party_Details', {}).get('Role') == 'Plaintiff']
    defendants = [item for item in data if item.get('Party_Details', {}).get('Role') == 'Defendant']

    # 4. Generate every combination for this specific case/file
    for p, d in product(plaintiffs, defendants):
        # Safely extract dictionaries
        meta = p.get('Metadata', {})
        p_details = p.get('Party_Details', {})
        d_details = d.get('Party_Details', {})
        
        # Build the flat row dictionary
        row = {
            # Metadata
            'Case_Number': meta.get('Case_Number'),
            'Coram': meta.get('Coram'),
            'Judge': meta.get('Judge'),
            'Date': meta.get('Date'),
            'Tribunal_Court': meta.get('Tribunal_Court'),
            
            # Party Names
            'Plaintiff_Name': p_details.get('Name'),
            'Defendant_Name': d_details.get('Name'),
            
            # Combined Text Fields (Arrays of Arrays)
            'Combined_Facts': [p_details.get('Facts', []), d_details.get('Facts', [])],
            'Combined_Issue': [p_details.get('Issue', ''), d_details.get('Issue', '')],
            'Combined_Rule': [p_details.get('Rule', ''), d_details.get('Rule', '')],
            'Combined_Application': [p_details.get('Application', ''), d_details.get('Application', '')],
            
            # Split Labels
            'plaintiff_label': p.get('Label'),
            'defendant_label': d.get('Label')
        }
        
        rows.append(row)

# 5. Create the master flat DataFrame from all files
df = pd.DataFrame(rows)

# View the total number of rows and the first few entries
print(f"Total combinations processed: {len(df)}")
print(df.head())

# Export the master DataFrame to a CSV for easy viewing
df.to_csv('court_cases.csv', index=False)