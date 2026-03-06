# Judiciary Scraper

This is a Python script used to automatically scrape judgments from the Singapore Judiciary website based on specified catchwords and years. The newest update downloads the actual PDF files for the cases directly into a local folder.

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

## Running the Scraper

To run the scraper, simply execute the script within your activated virtual environment:

```bash
python Scrapper.py
```

### What happens when you run it?
- The script uses `selenium` to open a Chrome window and navigates the Judiciary search page.
- It iterates through the configured `target_years` and `target_catchwords`.
- For each case found, it extracts case metadata (title, URL, catchwords).
- It attempts to find the associated **PDF file link** on the judgment page and downloads it directly to the local `Data/` folder.
- All scraped metadata (including the local path to the saved PDF in `Data/`) is exported automatically to a CSV file named `refined_corporate_judgments.csv`.

## Configuration
Inside `Scrapper.py`, you can change the target years and catchwords you want to scrape:
```python
target_years = ["2024", "2025", "2026"]
target_catchwords = [
    "Companies - Incorporation of companies - Lifting corporate veil",
    ...
]
# Adjust target_count=100 in the main loop to limit results per exact search.
```
