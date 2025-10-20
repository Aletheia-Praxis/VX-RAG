"""
Indexing PDFs from vx-underground.org using Selenium.
1. Opens the site in Chrome browser.
2. User manually passes the CAPTCHA.
3. After passing — automatically parses all PDF links and saves the index as JSON.
"""

import json
import os
from typing import List, Dict, Set, Optional
from urllib.parse import urljoin
from seleniumwire import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from dotenv import load_dotenv

PDF_LIMIT = 100


def get_all_pdf_links(driver, base_url: str, visited: Optional[Set[str]] = None) -> List[str]:
    """
    Recursively retrieves all PDF links from the site using Selenium.
    """
    if visited is None:
        visited = set()
    if base_url in visited or len(visited) > PDF_LIMIT:
        return []
    visited.add(base_url)

    try:
        driver.get(base_url)
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        html = driver.page_source
        safe_url = base_url.replace('https://', '').replace('/', '_').replace(':', '')
        with open(f"debug_{safe_url}.html", "w", encoding="utf-8") as f:
            f.write(html)
        soup = BeautifulSoup(html, "html.parser")
    except Exception as e:
        print(f"Error loading {base_url}: {e}")
        return []

    import time
    pdf_links = []
    # 1. Find all PDF files in the current folder
    pdf_spans = soup.find_all("span", class_="truncate")
    found_pdfs = 0
    for span in pdf_spans:
        name = span.get_text(strip=True)
        if name.lower().endswith(".pdf"):
            pdf_links.append({
                "name": name,
                "url": base_url,
                "path": base_url
            })
            found_pdfs += 1
            print(f"[FOUND] PDF: {name} @ {base_url}")
    print(f"[INFO] {found_pdfs} PDF(s) found in {base_url}")

    # 2. Find all subfolders for recursion
    folder_spans = soup.find_all("span", class_="truncate")
    found_folders = 0
    for span in folder_spans:
        folder_name = span.get_text(strip=True)
        if folder_name.endswith("/"):
            found_folders += 1
            print(f"[INFO] Entering folder: {folder_name} from {base_url}")
            try:
                folder_elem = driver.find_element(By.XPATH, f"//span[contains(@class, 'truncate') and text()='{folder_name}']")
                folder_elem.click()
                time.sleep(1.5)  # Add a pause for loading
                WebDriverWait(driver, 7).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
                new_url = driver.current_url
                # If there is a subfolder named Paper — go into it
                sub_soup = BeautifulSoup(driver.page_source, "html.parser")
                sub_folder_spans = sub_soup.find_all("span", class_="truncate")
                for sub_span in sub_folder_spans:
                    sub_folder_name = sub_span.get_text(strip=True)
                    if sub_folder_name.lower() == "paper":
                        print(f"[INFO] Entering Paper folder in {new_url}")
                        try:
                            paper_elem = driver.find_element(By.XPATH, f"//span[contains(@class, 'truncate') and text()='Paper']")
                            paper_elem.click()
                            time.sleep(1.5)
                            WebDriverWait(driver, 7).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
                            paper_url = driver.current_url
                            pdf_links.extend(get_all_pdf_links(driver, paper_url, visited))
                            driver.back()
                        except Exception as e:
                            print(f"[WARN] Could not click Paper folder: {e}")
                # Recursion into subfolder
                if new_url not in visited and len(visited) < PDF_LIMIT:
                    pdf_links.extend(get_all_pdf_links(driver, new_url, visited))
                driver.back()
            except Exception as e:
                print(f"Could not click folder {folder_name}: {e}")
    print(f"[INFO] {found_folders} folder(s) found in {base_url}")

    return pdf_links


def main():
    load_dotenv()
    start_urls = [
        "https://vx-underground.org/Archive",
        "https://vx-underground.org/Malware%20Analysis",
        "https://vx-underground.org/Papers",
        "https://vx-underground.org/tmp"
    ]
    output_file = "pdf_index.json"
    chrome_options = Options()
    chrome_options.add_argument("--window-size=1200,800")
    user_data_dir = os.getenv("CHROME_USER_DATA_DIR")
    profile = os.getenv("CHROME_PROFILE")
    if user_data_dir:
        chrome_options.add_argument(f"--user-data-dir={user_data_dir}")
    if profile:
        chrome_options.add_argument(f"--profile-directory={profile}")
    chrome_path = os.getenv("CHROMIUM_PATH")
    if chrome_path:
        chrome_options.binary_location = chrome_path
    chromedriver_path = os.getenv("CHROMEDRIVER_PATH")
    from selenium.webdriver.chrome.service import Service
    if chromedriver_path:
        service = Service(executable_path=chromedriver_path)
        driver = webdriver.Chrome(service=service, options=chrome_options)
    else:
        driver = webdriver.Chrome(options=chrome_options)
    print("Open the browser, pass the CAPTCHA, then press Enter in the console...")
    driver.get(start_urls[0])
    input("After passing the CAPTCHA, press Enter...")
    all_pdfs = []
    for url in start_urls:
        all_pdfs.extend(get_all_pdf_links(driver, url))
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_pdfs, f, ensure_ascii=False, indent=4)
    print(f"Found {len(all_pdfs)} PDF files. Index saved to {output_file}")


if __name__ == "__main__":
    main()
