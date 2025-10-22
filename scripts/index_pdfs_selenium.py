"""
Indexing PDFs from vx-underground.org using Selenium.
1. Opens the site in Chrome browser.
2. User manually passes the CAPTCHA.
3. After passing — automatically parses all PDF links and saves the index as JSON.
"""

import json
import os
import logging
import re
from typing import List, Set, Optional, Dict
from urllib.parse import urljoin, quote
from selenium.webdriver.remote.webdriver import WebDriver
from seleniumwire import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import requests

PDF_LIMIT = int(os.getenv('PDF_LIMIT', 100))


def get_all_pdf_links(driver: WebDriver, base_url: str, visited: Optional[Set[str]] = None) -> List[Dict[str, str]]:
    """
    Recursively retrieves all PDF links from the site using Selenium, considering patterns.
    """
    import time
    if visited is None:
        visited = set()
    if base_url in visited or len(visited) > PDF_LIMIT:
        return []
    visited.add(base_url)

    try:
        driver.get(base_url)
        WebDriverWait(driver, int(os.getenv('SELENIUM_TIMEOUT', 10))).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        html = driver.page_source
        safe_url = base_url.replace('https://', '').replace('/', '_').replace(':', '')
        debug_logs_dir = os.getenv('DEBUG_LOGS_DIR', 'scripts/debug_logs')
        with open(f"{debug_logs_dir}/debug_{safe_url}.html", "w", encoding="utf-8") as f:
            f.write(html)
        soup = BeautifulSoup(html, "html.parser")
    except Exception as e:
        logging.info(f"Error loading {base_url}: {e}")
        return []

    pdf_links: List[Dict[str, str]] = []
    found_pdfs = 0
    for span in soup.find_all("span", class_="truncate"):
        name = span.get_text(strip=True)
        if name.lower().endswith(".pdf"):
            s3_link = None
            parent = span.parent
            # Search for <a> with X-Amz-Algorithm near the PDF
            if parent is not None:
                for a in parent.find_all("a", href=True):
                    href = a.get("href")
                    if href and "X-Amz-Algorithm" in str(href):
                        s3_link = str(href)
                        break
            # If not found — search across the entire page
            if not s3_link:
                for a in soup.find_all("a", href=True):
                    href = a.get("href")
                    if href and name.replace(" ", "%20") in str(href) and "X-Amz-Algorithm" in str(href):
                        s3_link = str(href)
                        break
            url = base_url
            pdf_links.append({
                "name": str(name),
                "url": str(url),
                "path": str(s3_link) if s3_link else ""
            })
            found_pdfs += 1
            logging.info(f"[FOUND] PDF: {name} @ {base_url} S3: {s3_link}")
    logging.info(f"[INFO] {found_pdfs} PDF(s) found in {base_url}")

    # Pattern for Malware Analysis: /Malware Analysis/{year}/{subfolder}/Paper
    if "Malware%20Analysis" in base_url:
        # If this is a yearly folder, look for subfolders
        for span in soup.find_all("span", class_="truncate"):
            folder_name = span.get_text(strip=True)
            if folder_name.endswith("/") and folder_name[:4].isdigit():
                try:
                    folder_elem = driver.find_element(By.XPATH, f"//span[contains(@class, 'truncate') and text()='{folder_name}']")
                    folder_elem.click()
                    time.sleep(1.5)
                    WebDriverWait(driver, 7).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
                    year_url = driver.current_url
                    # In yearly folders, look for subfolders with Paper
                    year_soup = BeautifulSoup(driver.page_source, "html.parser")
                    year_spans = year_soup.find_all("span", class_="truncate")
                    for sub_span in year_spans:
                        subfolder_name = sub_span.get_text(strip=True)
                        if subfolder_name.endswith("/") and not subfolder_name[:4].isdigit():
                            try:
                                sub_elem = driver.find_element(By.XPATH, f"//span[contains(@class, 'truncate') and text()='{subfolder_name}']")
                                sub_elem.click()
                                time.sleep(1.5)
                                WebDriverWait(driver, 7).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
                                # Look for Paper folder
                                sub_soup = BeautifulSoup(driver.page_source, "html.parser")
                                sub_spans = sub_soup.find_all("span", class_="truncate")
                                for paper_span in sub_spans:
                                    if paper_span.get_text(strip=True) == "Paper":
                                        try:
                                            paper_elem = driver.find_element(By.XPATH, "//span[contains(@class, 'truncate') and text()='Paper']")
                                            paper_elem.click()
                                            time.sleep(1.5)
                                            WebDriverWait(driver, 7).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
                                            paper_url = driver.current_url
                                            pdf_links.extend(get_all_pdf_links(driver, paper_url, visited))
                                            driver.back()
                                        except Exception as e:
                                            logging.info(f"[WARN] Could not click Paper folder: {e}")
                                driver.back()
                            except Exception as e:
                                logging.info(f"Could not click subfolder {subfolder_name}: {e}")
                    driver.back()
                except Exception as e:
                    logging.info(f"Could not click year folder {folder_name}: {e}")
        return pdf_links

    # For Papers: recursively traverse all nested folders
    if "Papers" in base_url:
        for span in soup.find_all("span", class_="truncate"):
            folder_name = span.get_text(strip=True)
            if folder_name.endswith("/"):
                try:
                    folder_elem = driver.find_element(By.XPATH, f"//span[contains(@class, 'truncate') and text()='{folder_name}']")
                    folder_elem.click()
                    time.sleep(1.5)
                    WebDriverWait(driver, 7).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
                    new_url = driver.current_url
                    pdf_links.extend(get_all_pdf_links(driver, new_url, visited))
                    driver.back()
                except Exception as e:
                    logging.info(f"Could not click folder {folder_name}: {e}")
        return pdf_links

    # For Archive/The Old New Thing: PDFs are in yearly folders
    if "Archive/The%20Old%20New%20Thing" in base_url:
        for span in soup.find_all("span", class_="truncate"):
            folder_name = span.get_text(strip=True)
            if folder_name.endswith("/") and folder_name[:4].isdigit():
                try:
                    folder_elem = driver.find_element(By.XPATH, f"//span[contains(@class, 'truncate') and text()='{folder_name}']")
                    folder_elem.click()
                    time.sleep(1.5)
                    WebDriverWait(driver, 7).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
                    year_url = driver.current_url
                    pdf_links.extend(get_all_pdf_links(driver, year_url, visited))
                    driver.back()
                except Exception as e:
                    logging.info(f"Could not click year folder {folder_name}: {e}")
        return pdf_links

    # For tmp: PDFs are located directly
    if "tmp" in base_url:
        return pdf_links

    # For others — standard recursion
    for span in soup.find_all("span", class_="truncate"):
        folder_name = span.get_text(strip=True)
        if folder_name.endswith("/"):
            try:
                # Escape quotes
                safe_folder_name = folder_name.replace("'", "\\'").replace('"', '\"')
                folder_elem = driver.find_element(By.XPATH, f"//span[contains(@class, 'truncate') and text()='{safe_folder_name}']")
                folder_elem.click()
                time.sleep(1.5)
                WebDriverWait(driver, 7).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
                new_url = driver.current_url
                pdf_links.extend(get_all_pdf_links(driver, new_url, visited))
                driver.back()
            except Exception as e:
                logging.info(f"Could not click folder {folder_name}: {e}")
    return pdf_links


def download_pdfs(pdf_links: List[Dict[str, str]], driver: Optional[WebDriver] = None) -> None:
    """
    Downloads PDF files from the provided links to the 'downloaded_pdfs' directory.
    """
    download_dir = 'downloaded_pdfs'
    os.makedirs(download_dir, exist_ok=True)
    # Prepare requests session with retries
    session = requests.Session()
    # Use cookies from Selenium driver if available (helps after CAPTCHA)
    try:
        if driver is not None:
            selenium_cookies = driver.get_cookies() or []  # type: ignore
            for c in selenium_cookies:
                name = c.get('name')
                value = c.get('value')
                domain = c.get('domain')
                if name is None or value is None:
                    continue
                # requests' cookiejar expects string names/values
                try:
                    session.cookies.set(str(name), str(value), domain=str(domain) if domain is not None else None)
                except Exception:
                    # fallback: set without domain
                    try:
                        session.cookies.set(str(name), str(value))
                    except Exception:
                        logging.info(f"Could not set cookie {name}")
            # Try to mirror browser User-Agent
            try:
                ua = driver.execute_script('return navigator.userAgent')  # type: ignore
                if ua:
                    session.headers.update({'User-Agent': ua})
            except Exception as e:
                logging.info(f"Could not set User-Agent from Selenium driver: {e}")
    except Exception as e:
        logging.info(f"Could not copy cookies from Selenium driver: {e}")

    # Retry strategy
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    retries = Retry(total=3, backoff_factor=1, status_forcelist=(429, 500, 502, 503, 504))
    adapter = HTTPAdapter(max_retries=retries)
    session.mount('https://', adapter)
    session.mount('http://', adapter)

    # Parallel download with configurable workers
    import time
    from concurrent.futures import ThreadPoolExecutor

    workers = int(os.getenv('DOWNLOAD_WORKERS', '4'))

    def download_single(link: Dict[str, str]) -> None:
        url = link.get('url')
        path = link.get('path', '')
        name = link.get('name') or os.path.basename(str(url))
        # Basic sanitization: allow alnum, space, dash, underscore and dot
        safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_', '.')).strip()
        if not safe_name.lower().endswith('.pdf'):
            safe_name += '.pdf'
        # Ensure filename safe for filesystem
        safe_name = safe_name.replace('/', '_').replace('\\', '_')

        # Create subfolder based on path (extract relative path after domain)
        subfolder = ''
        if path:
            from urllib.parse import urlparse
            parsed = urlparse(path)
            subfolder = parsed.path.strip('/').replace('/', os.sep)
        full_dir = os.path.join(download_dir, subfolder)
        os.makedirs(full_dir, exist_ok=True)

        # Avoid overwriting: add suffix if exists
        filepath = os.path.join(full_dir, safe_name)
        base, ext = os.path.splitext(filepath)
        counter = 1
        while os.path.exists(filepath):
            filepath = f"{base}({counter}){ext}"
            counter += 1

        # Use path (S3 link) for download, skip if empty
        download_url = path
        if not download_url:
            logging.warning(f"Skipping entry without S3 path: {link}")
            return

        start_time = time.time()
        try:
            with session.get(str(download_url), stream=True, timeout=60) as resp:
                resp.raise_for_status()
                # Write in chunks
                with open(filepath, 'wb') as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            download_time = time.time() - start_time
            file_size = os.path.getsize(filepath)
            logging.info(f"Downloaded: {safe_name} from {download_url} (size: {file_size} bytes, time: {download_time:.2f}s)")
        except Exception as e:
            error_msg = str(e).lower()
            if "expired" in error_msg or "timestamp" in error_msg:
                logging.warning(f"S3 link expired, skipping: {download_url} - {e}")
            else:
                logging.error(f"Failed to download {download_url}: {e}")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        executor.map(download_single, pdf_links)


def main() -> None:
    load_dotenv()
    debug_logs_dir = os.getenv('DEBUG_LOGS_DIR', 'scripts/debug_logs')
    debug_log_file = os.getenv('DEBUG_LOG_FILE', 'scripts/debug_logs/debug.log')
    os.makedirs(debug_logs_dir, exist_ok=True)
    logging.basicConfig(filename=debug_log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    # Dynamically retrieve start URLs from the main page
    def get_start_urls_from_main_page(driver: WebDriver, main_url: str = "https://vx-underground.org/") -> list:
        driver.get(main_url)
        WebDriverWait(driver, int(os.getenv('SELENIUM_TIMEOUT', 10))).until(EC.presence_of_element_located((By.ID, "file-display")))
        soup = BeautifulSoup(driver.page_source, "html.parser")
        start_urls = []
        for div in soup.select("#file-display > div"):
            phx_click = div.get("phx-click")
            if phx_click and "change-directory" in str(phx_click):
                match = re.search(r'"value":"([^"]+)"', str(phx_click))
                if match:
                    folder_path = match.group(1)
                    # Form absolute URL
                    start_urls.append(urljoin(main_url, quote(folder_path)))
        return start_urls

    main_url = "https://vx-underground.org/"

    chrome_options = Options()
    chrome_options.add_argument("--window-size=1200,800")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--remote-debugging-port=9222")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-background-timer-throttling")
    chrome_options.add_argument("--disable-backgrounding-occluded-windows")
    chrome_options.add_argument("--disable-renderer-backgrounding")
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

    start_urls = get_start_urls_from_main_page(driver, main_url)
    output_file = os.getenv('OUTPUT_FILE', 'pdf_index.json')
    logging.info("Open the browser, pass the CAPTCHA, then press Enter in the console...")
    # Open the first root directory for CAPTCHA passing
    driver.get(main_url)
    input("After passing the CAPTCHA, press Enter...")
    all_pdfs = []
    for url in start_urls:
        all_pdfs.extend(get_all_pdf_links(driver, url))
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_pdfs, f, ensure_ascii=False, indent=4)
    message = f"Found {len(all_pdfs)} PDF files. Index saved to {output_file}"
    print(message)
    logging.info(message)
    # Download the PDFs (pass driver so cookies/User-Agent can be reused)
    download_pdfs(all_pdfs, driver)
    completion_message = "Script completed successfully."
    print(completion_message)
    logging.info(completion_message)


if __name__ == "__main__":
    main()
