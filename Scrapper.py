import os
import re
import requests
import time
import pandas as pd
import PyPDF2
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

def is_valid_corporate_pdf(pdf_path):
    """
    Validates if the downloaded PDF is relevant to corporate law by scanning the first 
    few pages for domain-specific NLP terms ("shareholder", "director", "fiduciary", etc.).
    """
    if pdf_path == "N/A" or not os.path.exists(pdf_path):
        return False
    try:
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            # Scanning up to first 5 pages for efficiency
            for i in range(min(5, len(reader.pages))): 
                extracted = reader.pages[i].extract_text()
                if extracted:
                    text += extracted
            
            text = text.lower()
            keywords = [
                "director", "shareholder", "fiduciary", "oppression", 
                "incorporation", "winding up", "company", "dividend", 
                "board of directors", "derivative action", "corporate veil"
            ]
            
            match_count = sum(1 for kw in keywords if kw in text)
            # Threshold: must contain at least 2 distinct corporate keywords
            if match_count >= 2: 
                return True
    except Exception as e:
        print(f"Error validating PDF {pdf_path}: {e}")
    return False

def scrape_by_catchwords(target_years, catchwords_list, target_count=100):
    # Ensure Data folder exists
    os.makedirs("Data", exist_ok=True)
    
    options = webdriver.ChromeOptions()
    # options.add_argument('--headless') # Commented out so you can watch it interact with the dropdown
    options.add_argument('--disable-gpu')
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    url = "https://www.elitigation.sg/gd"
    
    scraped_data = []
    
    try:
        driver.get(url)
        wait = WebDriverWait(driver, 15)
        
        for year in target_years:
            for catchword in catchwords_list:
                if len(scraped_data) >= target_count:
                    break
                    
                # 1. Format the Exact Phrase Search
                search_query = f'CatchWords:"{catchword}"'
                
                # 2. Input Search Query
                search_box = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "input[placeholder='Type your search terms here']")))
                search_box.clear()
                search_box.send_keys(search_query)
                
                # 3. Select Year from Dropdown
                year_dropdown = Select(driver.find_element(By.ID, "ddlYears"))
                try:
                    year_dropdown.select_by_value(year)
                except Exception as e:
                    print(f"Year {year} not found in dropdown. Skipping.")
                    continue
                
                # 4. Execute Search
                search_button = driver.find_element(By.CSS_SELECTOR, "button.in-input-group")
                search_button.click()
                time.sleep(5) # Wait for JS to load results
                
                # 5. Extract Results
                while len(scraped_data) < target_count:
                    soup = BeautifulSoup(driver.page_source, 'html.parser')
                    results = soup.find_all('div', class_='gd-card-body') # Refined selector
                    
                    if not results:
                        break # Move to next catchword/year if no results
                        
                    for result in results:
                        if len(scraped_data) >= target_count:
                            break
                            
                        # Extract Title and URL
                        title_elem = result.find('a', class_='gd-heardertext')
                        title = title_elem.text.strip() if title_elem else "N/A"
                        link = "https://www.elitigation.sg" + title_elem['href'] if title_elem else "N/A"
                        
                        # --- Extract corporate-law CatchWords from the case page ---
                        corporate_catchwords = "N/A"
                        pdf_path = "N/A"
                        if link != "N/A":
                            try:
                                # The case href is /gd/s/<ID>  →  PDF is /gd/gd/<ID>/pdf
                                case_id = title_elem['href'].split('/')[-1]  # e.g. 2025_SGHC_190
                                pdf_url = f"https://www.elitigation.sg/gd/gd/{case_id}/pdf"
                                
                                # Visit the case page to scrape the structured CatchWords
                                case_res = requests.get(link, timeout=15)
                                case_soup = BeautifulSoup(case_res.content, 'html.parser')
                                
                                # Catchwords are in <ul class='catchwords'> or plain text with [Companies — ...] pattern
                                raw_text = case_soup.get_text()
                                # Extract all [....] bracketed catchword entries
                                all_cw = re.findall(r'\[([^\[\]]+)\]', raw_text)
                                # Keep only those related to corporate / company law
                                corp_cw = [cw.strip() for cw in all_cw
                                           if re.match(r'Companies\s*[\u2014\-]', cw.strip())]
                                corporate_catchwords = " | ".join(corp_cw) if corp_cw else "N/A"
                                
                                # Download the PDF directly via the known URL pattern
                                safe_title = re.sub(r'[\\/*?:"<>|]', "", title)[:150]
                                pdf_path = os.path.join("Data", f"{safe_title}.pdf")
                                pdf_res = requests.get(pdf_url, stream=True, timeout=30)
                                if pdf_res.status_code == 200:
                                    with open(pdf_path, 'wb') as pf:
                                        for chunk in pdf_res.iter_content(4096):
                                            pf.write(chunk)
                                    print(f"Downloaded PDF: {title}")
                                else:
                                    print(f"PDF not available (HTTP {pdf_res.status_code}): {title}")
                                    pdf_path = "N/A"
                            except Exception as e:
                                print(f"Error processing {title}: {e}")
                        
                        scraped_data.append({
                            'Search_Year': year,
                            'Primary_Query': catchword,
                            'Case_Title': title,
                            'Corporate_CatchWords': corporate_catchwords,
                            'Document_URL': link,
                            'Local_PDF_Path': pdf_path
                        })
                    
                    # 6. Pagination
                    try:
                        next_button = driver.find_element(By.XPATH, "//a[contains(text(), 'Next')]")
                        if "disabled" in next_button.get_attribute("class") or not next_button.is_displayed():
                            break
                        next_button.click()
                        time.sleep(3)
                    except:
                        break # Next button not found, break pagination loop

    finally:
        driver.quit()
        
    df = pd.DataFrame(scraped_data)
    
    # Run the PDF Linguistic validation function 
    print("Validating downloaded PDFs for NLP richness...")
    if not df.empty:
        df['Is_Valid_Corporate'] = df['Local_PDF_Path'].apply(is_valid_corporate_pdf)
    else:
        df['Is_Valid_Corporate'] = False
        
    df.to_csv("refined_corporate_judgments.csv", index=False)
    
    valid_count = df['Is_Valid_Corporate'].sum() if not df.empty else 0
    print(f"Successfully scraped {len(df)} cases. High-quality NLP corporate matches: {valid_count}")
    return df

if __name__ == "__main__":
    # Define your parameters based on your requirements
    target_years = ["2024", "2025", "2026"]
    target_catchwords = [
        "Companies - Incorporation of companies - Lifting corporate veil",
        "Companies - Directors - Terms of appointment - Incorporation of company's constitution into director's contract of service",
        "Companies - Oppression",
        "Companies - Shares",
        "Companies - Shares - Allotment",
        "Companies - Accounts",
        "Companies - Winding up",
        "Companies - Directors - Liabilities",
        "Companies - Directors - Duties",
        "Companies - Directors - Breach of fiduciary duties",
        "Companies - Derivative action",
        "Companies - Minority shareholders",
        "Companies - Quasi-partnerships",
        "Companies - Schemes of arrangement",
        "Companies - Judicial management",
        "Companies - Separate legal personality",
        "Companies - Shares - Valuation",
        "Companies - Register of members",
        "Companies - Winding up - Just and equitable ground",
        "Companies - Charges",
        "Companies - Joint venture",
        "Companies - Cross-border insolvency"
    ]

    # Run the scraper
    df = scrape_by_catchwords(target_years, target_catchwords, target_count=100)