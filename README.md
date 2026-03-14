# ⚖️ Predicting Outcomes of Singapore Corporate Law Appeals

An end-to-end legal analytics pipeline that extracts structured legal features from Singapore court judgments using large language models and applies machine learning models to predict appeal outcomes.

## 🏗 Project Architecture

PDF Judgments
      │
      ▼
1. Document Ingestion
   - Load Singapore court judgment PDFs
   - Keep case ID / filename / source link
      │
      ▼
2. LLM Extraction
   - Send judgment text or PDF to GPT-4.1
   - Extract:
       • Metadata
         - Judge
         - Date
         - Tribunal/Court
         - Lawyers
         - Sector
       • IRAC
         - Facts
         - Issue
         - Rule
         - Application
         - Conclusion
       • Party-specific rows
         - one row per plaintiff / defendant discussed
      │
      ▼
3. Structured JSON Dataset
   - Store each judgment in normalized JSON
   - Example fields:
       • case_id
       • metadata
       • party_details
       • facts[]
       • issue
       • rule
       • application
       • conclusion
      │
      ▼
4. Dataset Validation / Cleaning
   - Check JSON validity
   - Remove duplicates
   - Standardize nulls
   - Verify facts do not contain verdict leakage
   - Normalize conclusion labels
      │
      ▼
5. Feature Engineering
   - Text features
       • Facts embeddings
       • Issue embeddings
       • Rule embeddings
       • Application embeddings
   - Categorical features
       • Court level
       • Judge
       • Sector
       • Lawyer / law firm
   - Legal citation features
       • number of precedents cited
       • statute categories
   - Target label
       • Liable / Not Liable
      │
      ▼
6. ML Training Dataset
   - Convert structured JSON into tabular format
   - One row = one party-case record
   - Split into train / validation / test
      │
      ▼
7. ML Model
   - Baselines:
       • Logistic Regression
       • Random Forest
       • XGBoost
   - Text-based models:
       • TF-IDF + classifier
       • Legal-BERT embeddings + classifier
      │
      ▼
8. Prediction Output
   - Predicted case outcome
   - Probability score
   - Important contributing features

## Prerequisites

- Chrome browser installed.
- Python 3.8+ installed and accessible via command line.
- Git (optional).

## Setup Instructions

1. **Open your Terminal (or Command Prompt / PowerShell)**
   Navigate to the directory containing `Scrapper.py`:
   ```bash
   cd c:\Users\temp\Documents\GitHub\BT4222-
   ```

2. **Create a Virtual Environment**
   Run the following command to create a folder named `venv` where all the required Python libraries will be isolated:
   ```bash
   # On Windows
   python -m venv venv
   # Or if 'python' isn't recognized, try:
   py -m venv venv
   ```

3. **Activate the Virtual Environment**
   Whenever you want to run the scraper or install dependencies, you must activate the virtual environment:
   ```bash
   # On Windows PowerShell
   .\venv\Scripts\Activate.ps1
   
   # On Windows Command Prompt
   .\venv\Scripts\activate
   ```
   *You'll know it's activated when you see `(venv)` at the beginning of your terminal prompt.*

4. **Install Requirements**
   Install all the required Python libraries with `pip`:
   ```bash
   pip install -r requirements.txt
   ```

## Pipeline Usage

This project is split into a **2-step pipeline**: Scraping the raw documents, and then extracting the NLP features.

### Step 1: Running the Scraper
To begin downloading cases, execute the scraper script:
```bash
python Scrapper.py
```

**Scraper Workflow:**
- The script uses `selenium` to search the `elitigation.sg` judgments portal.
- It iterates through the configured `target_years` and `target_catchwords`.
- For each case found, it immediately downloads the **PDF file** to the `Data/PDFs/` folder and runs a linguistic scan.
- **NLP Quality Control:** Only PDFs containing at least two valid corporate law keywords (e.g., "derivative action", "oppression") are kept. Irrelevant cases are automatically deleted to ensure the richness of the final dataset.

### Step 2: Extracting Machine Learning Features
Once your `Data/PDFs/` folder is populated with 100 high-quality cases, run the cleaning script:
```bash
python Clean_Data.py
```

**Extraction Workflow:**
- This script uses `pdfplumber` to analyze the raw text of every PDF in your `Data/PDFs` folder.
- It uses Regex to specifically target and extract the **CatchWords** and the **Conclusion** paragraphs (vital for determining case outcomes).
- The final processed dataset is automatically saved as `extracted_features.csv` in the `Data/Cleaned_Data/` folder, ready for your NLP model.

## Configuration
Inside `Scrapper.py`, you can change the target years and the specific domain of law you want to scrape:
```python
target_years = [str(year) for year in range(2000, 2027)][::-1]
target_catchwords = [
    "Companies — Directors — Duties"
]
# Adjust target_count=100 in the main loop to limit results
```
