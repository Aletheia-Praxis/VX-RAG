"""
Indexing PDFs from vx-underground.org using Selenium.
1. Opens the site in Chrome browser.
2. User manually passes the CAPTCHA.
3. After passing — automatically parses all PDF links and saves the index as JSON.
"""

import json
import os
import logging
import time
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
import concurrent.futures

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
    pdf_spans = soup.find_all("span", class_="truncate")
    found_pdfs = 0
    for span in pdf_spans:
        name = span.get_text(strip=True)
        if name.lower().endswith(".pdf"):
            pdf_links.append({
                "name": name,
                "url": urljoin(base_url, name),
                "path": base_url
            })
            found_pdfs += 1
            logging.info(f"[FOUND] PDF: {name} @ {base_url}")
    logging.info(f"[INFO] {found_pdfs} PDF(s) found in {base_url}")

    # Pattern for Malware Analysis: /Malware Analysis/{year}/{subfolder}/Paper
    if "Malware%20Analysis" in base_url:
        # If this is a yearly folder, look for subfolders
        for span in pdf_spans:
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
        folder_spans = soup.find_all("span", class_="truncate")
        for span in folder_spans:
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
        folder_spans = soup.find_all("span", class_="truncate")
        for span in folder_spans:
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
    folder_spans = soup.find_all("span", class_="truncate")
    for span in folder_spans:
        folder_name = span.get_text(strip=True)
        if folder_name.endswith("/"):
            try:
                # Escape quotes
                safe_folder_name = folder_name.replace("'", "\\'").replace('"', '\\"')
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
            selenium_cookies = driver.get_cookies() or []
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
                ua = driver.execute_script('return navigator.userAgent')
                if ua:
                    session.headers.update({'User-Agent': ua})
            except Exception:
                pass
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

    def download_single(link):
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

        if not url:
            logging.warning(f"Skipping entry without URL: {link}")
            return

        start_time = time.time()
        try:
            with session.get(str(url), stream=True, timeout=60) as resp:
                resp.raise_for_status()
                # Write in chunks
                with open(filepath, 'wb') as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            download_time = time.time() - start_time
            file_size = os.path.getsize(filepath)
            logging.info(f"Downloaded: {safe_name} from {url} (size: {file_size} bytes, time: {download_time:.2f}s)")
        except Exception as e:
            logging.error(f"Failed to download {url}: {e}")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        executor.map(download_single, pdf_links)


def main() -> None:
    load_dotenv()
    debug_logs_dir = os.getenv('DEBUG_LOGS_DIR', 'scripts/debug_logs')
    debug_log_file = os.getenv('DEBUG_LOG_FILE', 'scripts/debug_logs/debug.log')
    os.makedirs(debug_logs_dir, exist_ok=True)
    logging.basicConfig(filename=debug_log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    # Add specific paths to folders with PDFs
    start_urls = []
    # Archive/The Old New Thing: years 2003–2025
    for year in range(2003, 2026):
        start_urls.append(f"https://vx-underground.org/Archive/The%20Old%20New%20Thing/{year}")
    # Other specific folders
    start_urls.extend([
        "https://vx-underground.org/Archive/Collections/CocoMelon",
        "https://vx-underground.org/Malware%20Analysis/2006/2006-01-15%20-%20Win32-Neshta/Paper",
        "https://vx-underground.org/Malware%20Analysis/2006/2006-06-26%20-%20Blackmailer%20-%20The%20story%20of%20Gpcode/Paper",
        "https://vx-underground.org/Malware%20Analysis/2007/2007-01-09%20-%20A%20Rustock-ing%20Stuffer/Paper",
        "https://vx-underground.org/Malware%20Analysis/2007/2007-04-03%20-%20A%20Case%20Study%20of%20the%20Rustock%20Rootkit%20and%20Spam%20Bot/Paper",
        "https://vx-underground.org/Malware%20Analysis/2007/2007-10-31%20-%20Trojan.Bayrob%20Strikes%20Again!/Paper",
        "https://vx-underground.org/Malware%20Analysis/2007/2007-11-01%20-%20Spam%20from%20the%20kernel/Paper",
        "https://vx-underground.org/Malware%20Analysis/2007/2007-12-04%20-%20Inside%20the%20Ron%20Paul%20Spam%20Botnet/Paper",
        "https://vx-underground.org/Malware%20Analysis/2007/2007-12-16%20-%20Pushdo%20-%20Analysis%20of%20a%20Modern%20Malware%20Distribution%20System/Paper",
        "https://vx-underground.org/Malware%20Analysis/2008/2008-05-18%20-%20Rustock.C%20%E2%80%93%20Unpacking%20a%20Nested%20Doll/Paper",
        "https://vx-underground.org/Malware%20Analysis/2008/2008-06-08%20-%20%D0%9F%D0%BE%D1%82%D0%BE%D0%BC%D0%BE%D0%BA%20%C2%AB%D0%BD%D0%B5%D1%86%D0%B5%D0%BD%D0%B7%D1%83%D1%80%D0%BD%D0%BE%D0%B3%D0%BE%C2%BB%20%D1%82%D1%80%D0%BE%D1%8F%D0%BD%D0%B0%20%D0%B8%D0%BB%D0%B8%20%D0%BA%D0%B0%D0%BA%20%D0%B2%D0%BE%D1%80%D1%83%D1%8E%D1%82%20%D0%BF%D0%B0%D1%80%D0%BE%D0%BB%D0%B8%20%D0%BD%D0%B0%20FTP/Paper",
        "https://vx-underground.org/Malware%20Analysis/2008/2008-06-10%20-%20Who's%20behind%20the%20GPcode%20ransomware/Paper",
        "https://vx-underground.org/Malware%20Analysis/2008/2008-11-30%20-%20Agent.btz%20-%20A%20Threat%20That%20Hit%20Pentagon/Paper",
        "https://vx-underground.org/Malware%20Analysis/2009/2009-03-08%20-%20Conficker%20C%20Analysis/Paper",
        "https://vx-underground.org/Malware%20Analysis/2009/2009-03-17%20-%20Gheg%20spambot/Paper",
        "https://vx-underground.org/Malware%20Analysis/2009/2009-03-29%20-%20GhostNet/Paper",
        "https://vx-underground.org/Malware%20Analysis/2009/2009-05-07%20-%20W32.Qakbot/Paper",
        "https://vx-underground.org/Malware%20Analysis/2009/2009-05-31%20-%20Conficker.A%20binaries/Paper",
        "https://vx-underground.org/Malware%20Analysis/2009/2009-06-23%20-%20Virut%20Encryption%20Analysis/Paper",
        "https://vx-underground.org/Malware%20Analysis/2009/2009-07-08%20-%20Cyber%20attackers%20target%20South%20Korea%20and%20US/Paper",
        "https://vx-underground.org/Malware%20Analysis/2009/2009-07-11%20-%20Special!!!%20ZeuS%20Botnet%20for%20Dummies/Paper",
        "https://vx-underground.org/Malware%20Analysis/2009/2009-08-05%20-%20PC%20Users%20Threatened%20by%20Conficker%20Worm%20and%20new%20Internet-browser%20Modifier/Paper",
        "https://vx-underground.org/Malware%20Analysis/2009/2009-10-01%20-%20Detecting%20ZeuS/Paper",
        "https://vx-underground.org/Malware%20Analysis/2009/2009-10-29%20-%20Two-Headed%20Trojan%20Targets%20Online%20Banks/Paper",
        "https://vx-underground.org/Malware%20Analysis/2009/2009-11-02%20-%20New%20banking%20trojan%20W32.Silon%20-%20msjet51.dll/Paper",
        "https://vx-underground.org/Malware%20Analysis/2009/2009-11-02%20-%20Win32-Opachki.A%20-%20Trojan%20that%20removes%20Zeus%20(but%20it%20is%20not%20benign)/Paper",
        "https://vx-underground.org/Malware%20Analysis/2009/2009-11-11%20-%20Trojan-Win32-Opachki%20-%20redirections%20Google/Paper",
        "https://vx-underground.org/Malware%20Analysis/2009/2009-11-03%20-%20Opachki,%20from%20(and%20to)%20Russia%20with%20love/Paper",
        "https://vx-underground.org/Malware%20Analysis/2009/2009-11-11%20-%20Trojan-Win32-Opachki%20-%20redirections%20Google/Paper",
        "https://vx-underground.org/Malware%20Analysis/2010/2010-01-07%20-%20Lethic/Paper",
        "https://vx-underground.org/Malware%20Analysis/2010/2010-01-17%20-%20Jan%2017%20Trojan%20Darkmoon.B%20EXE%20Haiti%20relief/Paper",
        "https://vx-underground.org/Malware%20Analysis/2010/2010-01-25%20-%20Leveraging%20ZeuS%20to%20send%20spam%20through%20social%20networks/Paper",
        "https://vx-underground.org/Malware%20Analysis/2010/2010-02-02%20-%20ZeuS%20spreading%20via%20Facebook/Paper",
        "https://vx-underground.org/Malware%20Analysis/2010/2010-02-04%20-%20SpyEye%20Bot%20versus%20Zeus%20Bot/Paper",
        "https://vx-underground.org/Malware%20Analysis/2010/2010-02-08%20-%20List%20of%20Aurora%20-%20Hydraq%20-%20Roarur%20files/Paper",
        "https://vx-underground.org/Malware%20Analysis/2010/2010-02-19%20-%20SpyEye%20Bot%20(Part%20two)/Paper",
        "https://vx-underground.org/Malware%20Analysis/2010/2010-02-20%20-%20Facebook%20&%20VISA%20phishing%20campaign%20proposed%20by%20ZeuS/Paper",
        "https://vx-underground.org/Malware%20Analysis/2010/2010-03-03%20-%20Black%20Energy%20Crypto/Paper",
        "https://vx-underground.org/Malware%20Analysis/2010/2010-03-03%20-%20BlackEnergy%20Version%202%20Threat%20Analysis/Paper",
        "https://vx-underground.org/Malware%20Analysis/2010/2010-03-07%20-%20March%202010%20Opachki%20Trojan%20update%20and%20sample/Paper",
        "https://vx-underground.org/Malware%20Analysis/2010/2010-03-10%20-%20ZeuS%20Banking%20Trojan%20Report/Paper",
        "https://vx-underground.org/Malware%20Analysis/2010/2010-03-15%20-%20New%20phishing%20campaign%20against%20Facebook%20led%20by%20Zeus/Paper",
        "https://vx-underground.org/Malware%20Analysis/2010/2010-03-31%20-%20ICS%20Advisory%20(ICSA-10-090-01)-%20Mariposa%20Botnet/Paper",
        "https://vx-underground.org/Malware%20Analysis/2010/2010-04-01%20-%20SpyEye%20vs.%20ZeuS%20Rivalry/Paper",
        "https://vx-underground.org/Malware%20Analysis/2010/2010-04-19%20-%20ZeuS%20on%20IRS%20Scam%20remains%20actively%20exploited/Paper",
        "https://vx-underground.org/Malware%20Analysis/2010/2010-04-26%20-%20SpyEye%E2%80%99s%20-Kill%20Zeus-%20Bark%20is%20Worse%20Than%20its%20Bite/Paper",
        "https://vx-underground.org/Malware%20Analysis/2010/2010-05-03%20-%20A%20Brief%20Look%20at%20Zeus-Zbot%202.0/Paper",
        "https://vx-underground.org/Malware%20Analysis/2010/2010-05-03%20-%20Heloag%20has%20rather%20no%20friends,%20just%20a%20master/Paper",
        "https://vx-underground.org/Malware%20Analysis/2010/2010-05-11%20-%20Qakbot,%20Data%20Thief%20Unmasked-%20Part%20I/Paper",
        "https://vx-underground.org/Malware%20Analysis/2010/2010-05-27%20-%20Sasfis%20Propagation/Paper",
        "https://vx-underground.org/Malware%20Analysis/2010/2010-05-28%20-%20CVE-2009-3129%20XLS%20for%20office%202002-2007%20with%20fud%20keylogger%20EIDHR/Paper",
        "https://vx-underground.org/Malware%20Analysis/2010/2010-06-10%20-%20Review%20of%20the%20Virus.Win32.Virut.ce%20Malware%20Sample/Paper",
        "https://vx-underground.org/Malware%20Analysis/2010/2010-07-14%20-%20Who%20Was%20the%2012th%20Russian%20Spy%20at%20Microsoft/Paper",
        "https://vx-underground.org/Malware%20Analysis/2010/2010-07-14%20-%20ZeuS%20Version%20scheme%20by%20the%20trojan%20author/Paper",
        "https://vx-underground.org/Malware%20Analysis/2010/2010-07-15%20-%20Black%20DDoS/Paper",
        "https://vx-underground.org/Malware%20Analysis/2010/2010-07-24%20-%20Why%20won%E2%80%99t%20my%20sample%20run/Paper",
        "https://vx-underground.org/Malware%20Analysis/2010/2010-07-30%20-%20CVE-2010-2568%20keylogger%20Win32-Chymine.A/Paper",
        "https://vx-underground.org/Malware%20Analysis/2010/2010-08-25%20-%20Military%20Computer%20Attack%20Confirmed/Paper",
        "https://vx-underground.org/Malware%20Analysis/2010/2010-09-01%20-%20Injection%20as%20a%20way%20of%20life/Paper",
        "https://vx-underground.org/Malware%20Analysis/2010/2010-09-17%20-%20SpyEye%20Botnet%E2%80%99s%20Bogus%20Billing%20Feature/Paper",
        "https://vx-underground.org/Malware%20Analysis/2010/2010-10-25%20-%20Businesses%20Beware%20-%20Qakbot%20Spreads%20like%20a%20Worm,%20Stings%20like%20a%20Trojan/Paper",
        "https://vx-underground.org/Malware%20Analysis/2010/2010-11-10%20-%20Lethic%20Botnet%20Returns,%20Uses%20Realtek%20Identifier/Paper",
        "https://vx-underground.org/Malware%20Analysis/2010/2010-11-12%20-%20De-obfuscating%20and%20reversing%20the%20user-mode%20agent.pdf/Paper",
        "https://vx-underground.org/Malware%20Analysis/2010/2010-11-15%20-%20Tracing%20the%20Crimeware%20Origins%20by%20Reversing%20Injected%20Code/Paper",
        "https://vx-underground.org/Malware%20Analysis/2010/2010-11-16%20-%20The%20Device%20Driver%20Process%20Injection%20Rootkit/Paper",
        "https://vx-underground.org/Malware%20Analysis/2010/2010-11-20%20-%20The%20Kernel-Mode%20Device%20Driver%20Stealth%20Rootkit/Paper",
        "https://vx-underground.org/Malware%20Analysis/2010/2010-12-20%20-%20End%20of%20the%20Line%20for%20the%20Bredolab%20Botnet/Paper",
        "https://vx-underground.org/Malware%20Analysis/2011/2011-01-09%20-%20CVE-2010-3333%20DOC%20with%20info%20theft%20trojan%20from%20the%20American%20Chamber%20of%20Commerce/Paper",
        "https://vx-underground.org/Malware%20Analysis/2011/2011-01-20%20-%20Beschreibung%20des%20Virus%20Backdoor.Win32.%20Buterat.afj/Paper",
        "https://vx-underground.org/Malware%20Analysis/2011/2011-01-30%20-%20GpCode%20Ransomware%202010%20Simple%20Analysis/Paper",
        "https://vx-underground.org/Malware%20Analysis/2011/2011-03-02%20-%20TDL4%20and%20Glupteba%20-%20Piggyback%20PiggyBugs/Paper",
        "https://vx-underground.org/Malware%20Analysis/2011/2011-03-08%20-%20Worm-Win32-Yimfoca.A/Paper",
        "https://vx-underground.org/Malware%20Analysis/2011/2011-03-11%20-%20Trojan.Koredos%20Comes%20with%20an%20Unwelcomed%20Surprise/Paper",
        "https://vx-underground.org/Malware%20Analysis/2011/2011-03-28%20-%20Microsoft%20Hunting%20Rustock%20Controllers/Paper",
        "https://vx-underground.org/Malware%20Analysis/2011/2011-04-16%20-%20Troj-Sasfis-O/Paper",
        "https://vx-underground.org/Malware%20Analysis/2011/2011-04-19%20-%20TDSS%20part%201-%20The%20x64%20Dollar%20Question/Paper",
        "https://vx-underground.org/Malware%20Analysis/2011/2011-04-26%20-%20SpyEye%20Targets%20Opera,%20Google%20Chrome%20Users/Paper",
        "https://vx-underground.org/Malware%20Analysis/2011/2011-04-28%20-%20Un%20observateur%20d%E2%80%99%C3%A9v%C3%A9nements%20aveugle%E2%80%A6/Paper",
        "https://vx-underground.org/Malware%20Analysis/2011/2011-04-30%20-%20BKA-Trojaner%20(Ransomware)/Paper",
        "https://vx-underground.org/Malware%20Analysis/2011/2011-05-19%20-%20Win32-Expiro/Paper",
        "https://vx-underground.org/Malware%20Analysis/2011/2011-06-22%20-%20Criminals%20gain%20control%20over%20Mac%20with%20BackDoor.Olyx/Paper",
        "https://vx-underground.org/Malware%20Analysis/2011/2011-06-29%20-%20Inside%20a%20Back%20Door%20Attack/Paper",
        "https://vx-underground.org/Malware%20Analysis/2011/2011-07-06%20-%20Cybercriminals%20switch%20from%20MBR%20to%20NTFS/Paper",
        "https://vx-underground.org/Malware%20Analysis/2011/2011-07-08%20-%20Trojan.Mayachok.2-%20%D0%B0%D0%BD%D0%B0%D0%BB%D0%B8%D0%B7%20%D0%BF%D0%B5%D1%80%D0%B2%D0%BE%D0%B3%D0%BE%20%D0%B8%D0%B7%D0%B2%D0%B5%D1%81%D1%82%D0%BD%D0%BE%D0%B3%D0%BE%20VBR-%D0%B1%D1%83%D1%82%D0%BA%D0%B8%D1%82%D0%B0/Paper",
        "https://vx-underground.org/Malware%20Analysis/2011/2011-07-14%20-%20Cycbot%20-%20Ready%20to%20Ride/Paper",
        "https://vx-underground.org/Malware%20Analysis/2011/2011-07-26%20-%20SpyEye%20Trojan%20defeating%20online%20banking%20defenses/Paper",
        "https://vx-underground.org/Malware%20Analysis/2011/2011-07-28%20-%20Trojan%20Tricks%20Victims%20Into%20Transferring%20Funds/Paper",
        "https://vx-underground.org/Papers/AV%20Tech",
        "https://vx-underground.org/Papers/ICS%20SCADA/Duqu",
        "https://vx-underground.org/Papers/ICS%20SCADA/GreyEnergy%20and%20BlackEnergy",
        "https://vx-underground.org/Papers/ICS%20SCADA/Havex",
        "https://vx-underground.org/Papers/ICS%20SCADA/Industroyer",
        "https://vx-underground.org/Papers/ICS%20SCADA/Other",
        "https://vx-underground.org/Papers/ICS%20SCADA/Pipedream",
        "https://vx-underground.org/Papers/ICS%20SCADA/Stuxnet",
        "https://vx-underground.org/Papers/ICS%20SCADA/Triton",
        "https://vx-underground.org/Papers/LLVMs%20and%20Mutating%20Code",
        "https://vx-underground.org/Papers/Linux/Evasion",
        "https://vx-underground.org/Papers/Linux/Hooking",
        "https://vx-underground.org/Papers/Linux/Infection",
        "https://vx-underground.org/Papers/Linux/Internals",
        "https://vx-underground.org/Papers/Linux/Kernel%20Mode",
        "https://vx-underground.org/Papers/Linux/Persistence",
        "https://vx-underground.org/Papers/Linux/Process%20Injection",
        "https://vx-underground.org/Papers/Linux/System%20Components%20and%20Abuse",
        "https://vx-underground.org/Papers/MacOS%20Malware",
        "https://vx-underground.org/Papers/Malware%20Defense/AV%20Tech",
        "https://vx-underground.org/Papers/Russian/XSS-%D0%BA%D0%BE%D0%BB%D0%BB%D0%B5%D0%BA%D1%86%D0%B8%D1%8F",
        "https://vx-underground.org/Papers/VXUG%20Zines",
        "https://vx-underground.org/Papers/Web%20Malware",
        "https://vx-underground.org/Papers/Windows/AMSI",
        "https://vx-underground.org/Papers/Windows/Evasion",
        "https://vx-underground.org/Papers/Windows/GPU%20Abuse",
        "https://vx-underground.org/Papers/Windows/Hooking",
        "https://vx-underground.org/Papers/Windows/Infection",
        "https://vx-underground.org/Papers/Windows/Initial%20Access",
        "https://vx-underground.org/Papers/Windows/Internals%20and%20Analysis",
        "https://vx-underground.org/Papers/Windows/Kernel%20Mode",
        "https://vx-underground.org/Papers/Windows/Keylogging",
        "https://vx-underground.org/Papers/Windows/LSASS",
        "https://vx-underground.org/Papers/Windows/Networking",
        "https://vx-underground.org/Papers/Windows/Persistence",
        "https://vx-underground.org/Papers/Windows/Process%20Injection",
        "https://vx-underground.org/Papers/Windows/Shellcode%20Execution",
        "https://vx-underground.org/Papers/Windows/Syscalls",
        "https://vx-underground.org/Papers/Windows/System%20Components%20and%20Abuse",
        "https://vx-underground.org/Papers/Windows/Windows%20COM"
    ] + [
        urljoin("https://vx-underground.org/Malware%20Analysis/2025/", quote(f"{item.split(' - ')[0].strip()} - {item.split(' - ')[1].strip()}/Paper"))
        for item in [
            "2025-01-02 - NonEuclid RAT",
            "2025-01-03 - RATs on the island (Remote Access Trojans in Sri Lanka's Cybersecurity Landscape)",
            "2025-01-03 - SwaetRAT Delivery Through Python",
            "2025-01-04 - Solara - Roblox Executor Malware",
            "2025-01-06 - EAGERBEE, with updated and novel components, targets the Middle East",
            "2025-01-06 - Hangro - Investigating North Korean VPN Infrastructure Part 1",
            "2025-01-07 - PacketCrypt Classic Cryptocurrency Miner on PHP Servers",
            "2025-01-07 - Turla Cyber Campaign Targeting Pakistan’s Critical Infrastructure",
            "2025-01-07 - Unveiling Russian Surveillance Tech Expansion in Central Asia and Latin America",
            "2025-01-08 - Akira Ransomware Group & Malware Analysis Report",
            "2025-01-08 - TMPN (Skuld) Stealer - The dark side of open source",
            "2025-01-09 - Hackers claim to breach Russian state agency managing property, land records",
            "2025-01-09 - HexaLocker V2 - Skuld Stealer Paving the Way prior to Encryption",
            "2025-01-10 - FunkSec – Alleged Top Ransomware Group Powered by AI",
            "2025-01-13 - Abusing AWS Native Services- Ransomware Encrypting S3 Buckets with SSE-C",
            "2025-01-13 - Double-Tap Campaign - Russia-nexus APT possibly related to APT28 conducts cyber espionage on Central Asia and Kazakhstan diplomatic relations",
            "2025-01-14 - From Royal to BlackSuit",
            "2025-01-14 - Justice Department and FBI Conduct International Operation to Delete Malware Used by China-Backed Hackers",
            "2025-01-14 - More Than Malware Families- Retooling Our Approach to Tracking Software",
            "2025-01-14 - One Mikro Typo - How a simple DNS misconfiguration enables malware delivery by a Russian botnet",
            "2025-01-14 - Russia's largest platform for state procurement hit by cyberattack from pro-Ukraine group",
            "2025-01-15 - Article 113- One of the Russian-Ukrainian cyberwars, a review of the first major blackout in Ukraine caused by the Sandworm APT organization",
            "2025-01-15 - BabbleLoader - A Deep Dive into EDR and Machine Learning-Based Endpoint Protection Evasion",
            "2025-01-15 - F.A.C.C.T. found new attacks of pro-Ukrainian cyber spies Sticky Werewolf",
            "2025-01-15 - Zombies Never Die - Analysis of the Current Situation of Large Botnet AIRASHI",
            "2025-01-16 - Analysis of Threat Actor Data Posting",
            "2025-01-16 - FortiGate Firewall Configs Dumped- Revisiting CVE-2022-40684 Exploitation",
            "2025-01-16 - Lazarus APT - Techniques for Hunting Contagious Interview",
            "2025-01-16 - MintsLoader - StealC and BOINC Delivery",
            "2025-01-16 - New Star Blizzard spear-phishing campaign targets WhatsApp accounts",
            "2025-01-16 - Will the Real Volt Typhoon Please Stand Up",
            "2025-01-20 - APT actor classification \"addiction\" - Practical issues of attribution seen in Lazarus subgroup classification",
            "2025-01-20 - Qbot is Back.Connect",
            "2025-01-21 - Love and hate under war - The GamaCopy organization, which imitates the Russian Gamaredon, uses military — related bait to launch attacks on Russia",
            "2025-01-21 - Silent Lynx APT Targets Various Entities Across Kyrgyzstan & Neighbouring Nations",
            "2025-01-22 - Categorizing Software with Code Families",
            "2025-01-22 - PlushDaemon compromises supply chain of Korean VPN service",
            "2025-01-23 - Cluster of Infrastructure likely used by Affiliate of Dark Scorpius (Black Basta)",
            "2025-01-23 - Helldown Ransomware Malware Analysis Report",
            "2025-01-23 - Lumma Stealer - Fake CAPTCHAs & New Techniques to Evade Detection",
            "2025-01-23 - RID Hijacking Technique Utilized by Andariel Attack Group",
            "2025-01-23 - The J-Magic Show - Magic Packets and Where to find them",
            "2025-01-25 - Sophos MDR tracks two ransomware campaigns using “email bombing,” Microsoft Teams “vishing”",
            "2025-01-27 - Cobalt Strike and a Pair of SOCKS Lead to LockBit Ransomware",
            "2025-01-27 - Technical Analysis of Xloader Versions 6 and 7 - Part 1",
            "2025-01-29 - North Korean APT Lazarus Targets Developers with Malicious npm Package",
            "2025-01-29 - Operation Phantom Circuit - North Korea’s Global Data Exfiltration Campaign",
            "2025-01-30 - Backdoor found in two healthcare patient monitors, linked to IP in China",
            "2025-01-30 - Coyote Banking Trojan - A Stealthy Attack via LNK Files",
            "2025-01-30 - Cybercrime websites selling hacking tools to transnational organized crime groups seized",
            "2025-01-30 - One ClickFix and LummaStealer reCAPTCHA’s Our Attention - Part 1",
            "2025-01-30 - Ongoing Email Bombing Campaigns leading to Remote Access and Post-Exploitation",
            "2025-01-30 - TAG-124’s Multi-Layered TDS Infrastructure and Extensive User Base",
            "2025-01-30 - UAC-0063 - Cyber Espionage Operation Expanding from Central Asia",
            "2025-01-31 - Attackers Leveraging Microsoft Teams Defaults and Quick Assist for Social Engineering Attacks",
            "2025-02-02 - Do the CONTEC CMS8000 Patient Monitors Contain a Chinese Backdoor - The Reality is More Complicated…",
            "2025-02-03 - LegionLoader exposed",
            "2025-02-03 - macOS FlexibleFerret - Further Variants of DPRK Malware Family Unearthed",
            "2025-02-04 - Analyzing ELF-Sshdinjector.A!tr with a Human and Artificial Analyst",
            "2025-02-04 - CVE-2025-0411 - Ukrainian Organizations Targeted in Zero-Day Campaign and Homoglyph Attacks",
            "2025-02-04 - Unpacking the BADBOX Botnet with Censys",
            "2025-02-05 - Lazarus Group Targets Organizations with Sophisticated LinkedIn Recruiting Scam",
            "2025-02-05 - Stealthy Attack - Dual Injection Undermines Chrome’s App-Bound Encryption",
            "2025-02-06 - Code injection attacks using publicly disclosed ASP.NET machine keys",
            "2025-02-06 - Google Tag Manager Skimmer Steals Credit Card Info From Magento Site",
            "2025-02-07 - SI-CERT TZ016 - BeaverTail & InvisibleFerret",
            "2025-02-09 - Analysis of malicious mobile applications impersonating popular Polish apps — OLX, Allegro, IKO",
            "2025-02-10 - Further insights into Ivanti CSA 4.6 vulnerabilities exploitation",
            "2025-02-10 - Tracking Ransomware - January 2025",
            "2025-02-11 - RATatouille - Cooking Up Chaos in the I2P Kitchen",
            "2025-02-11 - Sandworm APT Exploits Trojanized KMS Tools to Target Ukrainian Users in Cyber Espionage Campaign",
            "2025-02-11 - Sandworm APT Targets Ukrainian Users with Trojanized Microsoft KMS Activation Tools in Cyber Espionage Campaigns",
            "2025-02-12 - BTMOB RAT - Newly Discovered Android Malware Spreading via Phishing Sites",
            "2025-02-12 - Defying tunneling - A Wicked approach to detecting malicious network traffic",
            "2025-02-12 - North Korean Hackers Exploit PowerShell Trick to Hijack Devices in New Cyberattack",
            "2025-02-12 - Surge in attacks exploiting old ThinkPHP and ownCloud flaws",
            "2025-02-12 - Suspected North Korean hacker hacks a large number of data from a government document system developer",
            "2025-02-12 - The BadPilot campaign - Seashell Blizzard subgroup conducts multiyear global access operation",
            "2025-02-12 - Two tales and one Antidot(e) — a new mobile malware campaign in Poland",
            "2025-02-12 - Unpacking Pyarmor v8+ scripts",
            "2025-02-13 - Analyzing DEEP#DRIVE- North Korean Threat Actors Observed Exploiting Trusted Platforms for Targeted Attacks",
            "2025-02-13 - China-linked Espionage Tools Used in Ransomware Attacks",
            "2025-02-13 - Cybercrooks Are Using Fake Job Listings to Steal Crypto",
            "2025-02-13 - From South America to Southeast Asia - The Fragile Web of REF7707",
            "2025-02-13 - Inside the Scam - North Korea’s IT Worker Threat",
            "2025-02-13 - Multiple Russian Threat Actors Targeting Microsoft Device Code Authentication",
            "2025-02-13 - RedMike (Salt Typhoon) Exploits Vulnerable Cisco Devices of Global Telecommunications Providers",
            "2025-02-13 - Storm-2372 conducts device code phishing campaign",
            "2025-02-13 - Technical Analysis of Xloader Versions 6 and 7 - Part 2",
            "2025-02-13 - Threat hunting case study - SocGholish",
            "2025-02-13 - You've Got Malware - FINALDRAFT Hides in Your Drafts",
            "2025-02-15 - Dissecting a fresh BlankGrabber sample",
            "2025-02-18 - An Update on Fake Updates - Two New Actors, and New Mac Malware",
            "2025-02-18 - An inside look at NSA (Equation Group) TTPs from China’s lense",
            "2025-02-18 - Exposing the Deceit - Phishing Sites Impersonating Government Entities",
            "2025-02-18 - IOCs Green Nailao campaign (NailaoLocker, ShadowPad)",
            "2025-02-19 - #StopRansomware - Ghost (Cring) Ransomware",
            "2025-02-19 - Technical Analysis of Lockbit4.0 Evasion Tales",
            "2025-02-19 - The Pangu Team—iOS Jailbreak and Vulnerability Research Giant- A Member of i-SOON’s Exploit-Sharing Network",
            "2025-02-20 - 48 Minutes - How Fast Phishing Attacks Exploit Weaknesses",
            "2025-02-20 - APT-C-28 Group Launched New Cyber Attack With Fileless RokRat Malware",
            "2025-02-20 - DeceptiveDevelopment targets freelance developers",
            "2025-02-20 - GhostSocks - Lumma's Partner In Proxy",
            "2025-02-20 - Linkc Ransomware - The New Cybercriminal Group Targeting Artificial Intelligence Data",
            "2025-02-20 - Meet NailaoLocker - a ransomware distributed in Europe by ShadowPad and PlugX backdoors",
            "2025-02-20 - Updated Shadowpad Malware Leads to Ransomware Deployment",
            "2025-02-20 - Weathering the storm - In the midst of a Typhoon",
            "2025-02-21 - Angry Likho - Old beasts in a new forest",
            "2025-02-21 - How’s that for a malicious Linkc, new group launches DLS",
            "2025-02-21 - TRM Links North Korea to Record $1.5 Billion Record Hack",
            "2025-02-24 - Android trojan TgToxic updates its capabilities",
            "2025-02-24 - Auto-Color - An Emerging and Evasive Linux Backdoor",
            "2025-02-24 - Cryptocurrency APT Intelligence - Unveiling Lazarus Group’s Intrusion Techniques",
            "2025-02-24 - LCRYX Ransomware - How a VB Ransomware Locks Your System",
            "2025-02-24 - Six Months Undetected - Analysis of archive.org hosted .NET PE Injector",
            "2025-02-24 - The GitVenom campaign - Cryptocurrency theft using GitHub",
            "2025-02-25 - Ghostwriter - New Campaign Targets Ukrainian Government and Belarusian Opposition",
            "2025-02-25 - PolarEdge - Unveiling an uncovered ORB network",
            "2025-02-26 - Alert Number - I-022625-PSA - North Korea Responsible for $1.5 Billion Bybit Hack",
            "2025-02-26 - Inside BlackBasta - What Leaked Conversations Reveal About Their Ransomware Operations",
            "2025-02-27 - BlackBasta Leaks - Lessons from the Ascension Health attack",
            "2025-02-27 - Disrupting a global cybercrime network abusing generative AI",
            "2025-02-27 - Long Live The Vo1d Botnet - New Variant Hits 1.6 Million TV Globally",
            "2025-02-27 - Modern Approach to Attributing Hacktivist Groups",
            "2025-02-27 - NailaoLoader - Hiding Execution Flow via Patching",
            "2025-02-27 - NanoCore Malware Analysis",
            "2025-02-27 - Phishing Email Attacks by the Larva-24005 Group Targeting Japan",
            "2025-02-27 - Russian campaign targeting Romanian WhatsApp numbers",
            "2025-02-27 - Squidoor - Suspected Chinese Threat Actor’s Backdoor Targets Global Organizations",
            "2025-02-27 - The Rise of the Fake Tech Workforce - State-Sponsored Infiltration of U.S. Technical Supply Chains",
            "2025-02-27 - Winos 4.0 Spreads via Impersonation of Official Email to Target Users in Taiwan",
            "2025-02-28 - Agent AI, Basta Parser Extraordinaire",
            "2025-02-28 - Black Basta exposed - A look at a cybercrime data leak",
            "2025-02-28 - JavaGhost’s Persistent Phishing Attacks From the Cloud",
            "2025-02-28 - New DDoS Botnet Discovered - Over 30,000 Hacked Devices, Majority of Observed Activity Traced to Iran",
            "2025-02-28 - Notorious Malware, Spam Host “Prospero” Moves to Kaspersky Lab",
            "2025-03-01 - An in-depth analysis of APT37’s latest campaign",
            "2025-03-01 - Ransomware - de REvil à Black Basta, que sait-on de Tramp",
            "2025-03-02 - Pivoting on Black Basta's (leaked) Infrastructure",
            "2025-03-03 - Black Basta and Cactus Ransomware Groups Add BackConnect Malware to Their Arsenal",
            "2025-03-03 - PureLogs Deep Analysis- Evasion, Data Theft, and Encryption Mechanism",
            "2025-03-04 - Analysis of Kimsuky Group association with emergency martial arts-themed APT attack",
            "2025-03-04 - Black Basta Leak Analysis",
            "2025-03-04 - Likely DPRK Network Backstops on GitHub, Targets Companies Globally",
            "2025-03-04 - Ragnar Loader Indicators of Compromise (IOC)",
            "2025-03-04 - Thousands of websites hit by four backdoors in 3rd party JavaScript attack",
            "2025-03-04 - Tracking Emmenhtal",
            "2025-03-05 - Initial Takeaways from the Black Basta Chat Leaks",
            "2025-03-05 - Satori Threat Intelligence Disruption - BADBOX 2.0 Targets Consumer Devices with Multiple Fraud Schemes",
            "2025-03-05 - Silk Typhoon targeting IT supply chain",
            "2025-03-05 - Water Ouroboros",
            "2025-03-06 - Deciphering Black Basta’s Infrastructure from the Chat Leak",
            "2025-03-06 - The Next Level - Typo DGAs Used in Malicious Redirection Chains",
            "2025-03-06 - Unveiling EncryptHub - Analysis of a multi-stage malware campaign",
            "2025-03-07 - Akira Ransomware Expands to Linux - The attacking abilities and strategies",
            "2025-03-07 - Remote Monitoring and Management (RMM) Tooling Increasingly an Attacker’s First Choice",
            "2025-03-10 - Blind Eagle- …And Justice for All",
            "2025-03-10 - DieNet and #Shiite_Harvest claimed responsibility for disabling ten significant Iraqi websites",
            "2025-03-10 - Lazarus Strikes npm Again with New Wave of Malicious Packages",
            "2025-03-10 - Trump Cryptocurrency Delivers ConnectWise RAT",
            "2025-03-11 - AI-Assisted Fake GitHub Repositories Fuel SmartLoader and LummaStealer Distribution",
            "2025-03-11 - Blind Eagle Hacks Colombian Institutions Using NTLM Flaw, RATs and GitHub-Based Attacks",
            "2025-03-11 - Cato CTRL Threat Research - Ballista – New IoT Botnet Targeting Thousands of TP-Link Archer Routers",
            "2025-03-11 - DCRat backdoor returns",
            "2025-03-11 - DragonForce Ransomware - Unveiling Its Tactics and Impact",
            "2025-03-11 - IOCs for Anubis Backdoor",
            "2025-03-12 - Ghost in the Router - China-Nexus Espionage Actor UNC3886 Targets Juniper Routers",
            "2025-03-12 - Golang backdoor with a side of ChromeUpdateAlert App",
            "2025-03-12 - Lookout Discovers New Spyware by North Korean APT37",
            "2025-03-12 - Medusa Ransomware",
            "2025-03-13 - Analyzing OBSCURE#BAT Threat Actors Lure Victims into Executing Malicious Batch Scripts to Deploy Stealthy Rootkits",
            "2025-03-13 - ArechClient; Decoding IOCs and finding the onboard browser extension",
            "2025-03-13 - Botnets never die",
            "2025-03-13 - Decrypting Encrypted files from Akira Ransomware (Linux-ESXI variant 2024) using a bunch of GPUs",
            "2025-03-13 - Inside BRUTED - Black Basta (RaaS) Members Used Automated Brute Forcing Framework to Target Edge Network Devices",
            "2025-03-13 - New Ransomware Operator Exploits Fortinet Vulnerability Duo",
            "2025-03-13 - Tracking Ransomware - February 2025",
            "2025-03-13 - Work Hard, Pay Harder!",
            "2025-03-14 - Android Banking Trojan – OctoV2, masquerading as Deepseek AI",
            "2025-03-14 - Lumma Stealer – A tale that starts with a fake Captcha",
            "2025-03-14 - SocGholish’s Intrusion Techniques Facilitate Distribution of RansomHub Ransomware",
            "2025-03-15 - Understanding SalatStealer - Features and Impact",
            "2025-03-16 - Analyzing the RedTiger Malware Stealer",
            "2025-03-16 - Bybit – What We Know So Far",
            "2025-03-17 - Black Basta’s blunder - exploiting the gang’s leaked chats",
            "2025-03-17 - DollyWay World Domination - Eight Years of Evolving Website Malware Campaigns",
            "2025-03-18 - Code-signing certificate abuse in the Black Basta chat leaks (and how to fight back)",
            "2025-03-18 - Operation AkaiRyū - MirrorFace invites Europe to Expo 2025 and revives ANEL backdoor",
            "2025-03-20 - Operation FishMedley",
            "2025-03-20 - Reversing FUD AMOS Stealer",
            "2025-03-20 - UAT-5918 targets critical infrastructure entities in Taiwan",
            "2025-03-22 - Back to Business - Lumma Stealer Returns with Stealthier Methods",
            "2025-03-23 - Analyzing Vidar Stealer",
            "2025-03-24 - Weaver Ant, the Web Shell Whisperer - Tracking a Live China-nexus Operation",
            "2025-03-25 - IBM X-Force discovers new Sheriff Backdoor used to target Ukraine",
            "2025-03-25 - Inside DollyWay’s C2 Infrastructure - Traffic Direction Systems and the LosPollos Connection",
            "2025-03-25 - Inside Kimsuky’s Latest Cyberattack - Analyzing Malicious Scripts and Payloads",
            "2025-03-25 - On the Hunt for Ghost(Socks)",
            "2025-03-25 - Operation ForumTroll - APT attack with Google Chrome zero-day exploit chain",
            "2025-03-25 - Phishing Campaign Targets Defense and Aerospace Firms Linked to Ukraine Conflict",
            "2025-03-25 - Tempted to Classifying APT Actors- Practical Challenges of Attribution in the Case of Lazarus’s Subgroup",
            "2025-03-26 - CoffeeLoader - A Brew of Stealthy Techniques",
            "2025-03-26 - Lynx Ransomware - Learn details about the operation and how to mitigate this threat",
            "2025-03-26 - The Long and Short(cut) of It- KoiLoader Analysis",
            "2025-03-27 - A Phishing Tale of DoH and DNS MX Abuse",
            "2025-03-28 - A Deep Dive into Water Gamayun’s Arsenal and Infrastructure",
            "2025-03-28 - Exposing Crocodilus - New Device Takeover Malware Targeting Android Devices",
            "2025-03-28 - Hidden Malware Strikes Again - Mu-Plugins Under Attack",
            "2025-03-28 - TsarBot - A New Android Banking Trojan Targeting Over 750 Banking, Finance, and Cryptocurrency Applications",
            "2025-03-31 - Analyzing New HijackLoader Evasion Tactics",
            "2025-03-31 - CPU_HU - Fileless cryptominer targeting exposed PostgreSQL with over 1.5K victims",
            "2025-03-31 - DarkCloud Stealer",
            "2025-03-31 - From Contagious to ClickFake Interview - Lazarus leveraging the ClickFix tactic",
            "2025-03-31 - Gootloader Returns - Malware Hidden in Google Ads for Legal Documents",
            "2025-03-31 - Malware hiding in plain sight - Spying on North Korean Hackers",
            "2025-03-31 - Operation HollowQuill - Malware delivered into Russian R&D Networks via Research Decoy PDFs",
            "2025-03-31 - The Espionage Toolkit of Earth Alux - A Closer Look at its Advanced Techniques",
            "2025-04-01 - Auto-color - Linux backdoor",
            "2025-04-01 - Salvador Stealer - New Android Malware That Phishes Banking Details & OTPs",
            "2025-04-01 - Same Russian-Speaking Threat Actor, New Tactics Abuse of Cloudflare Services for Phishing and Telegram to Filter Victim IPs",
            "2025-04-02 - An in-depth look at Black Basta's TTPs",
            "2025-04-02 - BeaverTail and Tropidoor Malware Distributed via Recruitment Emails",
            "2025-04-02 - Tracking Adversaries - EvilCorp, the RansomHub affiliate",
            "2025-04-03 - Threat actors leverage tax season to deploy tax-themed phishing campaigns",
            "2025-04-03 - UAC-0219 Attack Detection - A New Cyber-Espionage Campaign Using a PowerShell Stealer WRECKSTEEL",
            "2025-04-04 - Lazarus Expands Malicious npm Campaign - 11 New Packages Add Malware Loaders and Bitbucket Payloads",
            "2025-04-04 - OPSEC Failure Exposes Coquettte's Malware Campaigns on Bulletproof Hosting Servers",
            "2025-04-07 - UAC-0226 Attack Detection - New Cyber-Espionage Campaign Targeting Ukrainian Innovation Hubs and Government Entities with GIFTEDCROOK Stealer",
            "2025-04-08 - Exploitation of CLFS zero-day leads to ransomware activity",
            "2025-04-08 - Goodbye HTA, Hello MSI- New TTPs and Clusters of an APT driven by Multi-Platform Attacks",
            "2025-04-08 - Inside DanaBot’s Infrastructure - In Support of Operation Endgame II",
            "2025-04-08 - State-Sponsored Tactics - How Gamaredon and ShadowPad Operate and Rotate Their Infrastructure",
            "2025-04-10 - GOFFEE continues to attack organizations in Russia",
            "2025-04-10 - Newly Registered Domains Distributing SpyNote Malware",
            "2025-04-11 - Flesh Stealer - A Report on Multivector Data Theft",
            "2025-04-11 - Interview with the Chollima",
            "2025-04-11 - Threat Spotlight - Hijacked and Hidden - New Backdoor and Persistence Technique",
            "2025-04-14 - BPFDoor’s Hidden Controller Used Against Asia, Middle East Targets",
            "2025-04-14 - New Malware Variant Identified - ResolverRAT Enters the Maze",
            "2025-04-14 - Proton66 Part 1 - Mass Scanning and Exploit Campaigns",
            "2025-04-14 - Slow Pisces Targets Developers With Coding Challenges and Introduces New Customized Python Malware",
            "2025-04-15 - CyberSOC Insights - Analysis of a Black Basta Attack Campaign",
            "2025-04-15 - Hunting Mice In Tunnels II - Fake CAPTCHAs and Ransomware",
            "2025-04-15 - Renewed APT29 Phishing Campaign Against European Diplomats",
            "2025-04-15 - UNC5174’s evolution in China’s ongoing cyber warfare- From SNOWLIGHT to VShell",
            "2025-04-16 - Inside Gamaredon’s PteroLNK - Dead Drop Resolvers and evasive Infrastructure",
            "2025-04-16 - Interlock ransomware evolving under the radar",
            "2025-04-17 - Around the World in 90 Days - State-Sponsored Actors Try ClickFix",
            "2025-04-17 - Breaking the B0 ransomware - Investigation & Decryption",
            "2025-04-17 - IronHusky updates the forgotten MysterySnail RAT to target Russia and Mongolia",
            "2025-04-17 - Mitigating ELUSIVE COMET Zoom remote control attacks",
            "2025-04-17 - Proton66 Part 2 - Compromised WordPress Pages and Malware Campaigns",
            "2025-04-17 - Unmasking the new XorDDoS controller and infrastructure",
            "2025-04-21 - Unmasking the Evolving Threat - A Deep Dive into the Latest Version of Lumma InfoStealer with Code Flow Obfuscation",
            "2025-04-22 - Distribution of PebbleDash Malware in March 2025",
            "2025-04-22 - Infostealer Malware FormBook Spread via Phishing Campaign – Part I",
            "2025-04-22 - Phishing for Codes - Russian Threat Actors Target Microsoft 365 OAuth Workflows",
            "2025-04-22 - Russian organizations targeted by backdoor masquerading as secure networking software updates",
            "2025-04-23 - AsyncRAT Malware Analysis",
            "2025-04-23 - Introducing ToyMaker, an initial access broker working in cahoots with double extortion gangs",
            "2025-04-23 - Russian Infrastructure Plays Crucial Role in North Korean Cybercrime Operations",
            "2025-04-23 - Understanding the threat landscape for Kubernetes and containerized assets",
            "2025-04-24 - Contagious Interview (DPRK) Launches a New Campaign Creating Three Front Companies to Deliver a Trio of Malware - BeaverTail, InvisibleFerret, and OtterCookie",
            "2025-04-24 - Crypters And Tools. Part 2- Different Paws — Same Tangle",
            "2025-04-24 - Understanding Alcatraz ~ Obfuscator Analysis [EN]",
            "2025-04-25 - Earth Kurma APT Campaign Targets Southeast Asian Government, Telecom Sectors",
            "2025-04-25 - Rolling in the Deep(Web) - Lazarus Tsunami",
            "2025-04-25 - The Persistent Threat of Salt Typhoon - Tracking Exposures of Potentially Targeted Devices",
            "2025-04-28 - Top Tier Target - What It Takes to Defend a Cybersecurity Company from Today’s Adversaries",
            "2025-04-28 - Uncovering Actor TTP Patterns and the Role of DNS in Investment Scams",
            "2025-04-29 - Gremlin Stealer - New Stealer on Sale in Underground Forum",
            "2025-04-29 - Nitrogen Dropping Cobalt Strike – A Combination of “Chemical Elements”",
            "2025-04-29 - Russia – Assignment of cyber attacks against France to the Russian military intelligence service (APT28) (29 April 2025)",
            "2025-04-29 - Uncovering MintsLoader With Recorded Future Malware Intelligence Hunting",
            "2025-04-29 - Yet Another NodeJS Backdoor (YaNB)- A Modern Challenge",
            "2025-04-30 - Advisory - Pahalgam Attack themed decoys used by APT36 to target the Indian Government",
            "2025-04-30 - Finding Malware - Unveiling LUMMAC.V2 with Google Security Operations",
            "2025-05-01 - Deep Dive Fog ransomware",
            "2025-05-01 - FortiGuard Incident Response Team Detects Intrusion into Middle East Critical National Infrastructure",
            "2025-05-01 - I StealC You - Tracking the Rapid Changes To StealC",
            "2025-05-01 - TerraStealerV2 and TerraLogger - Golden Chickens' New Malware Families Discovered",
            "2025-05-02 - Prelude - Crypto Heist Causes HAVOC",
            "2025-05-02 - Venom Spider Uses Server-Side Polymorphism to Weave a Web Around Victims",
            "2025-05-05 - Negotiations with the Akira ransomware group - an ill-advised approach",
            "2025-05-06 - Defending Against UNC3944 - Cybercrime Hardening Guidance from the Frontlines",
            "2025-05-06 - Here Comes Mirai - IoT Devices RSVP to Active Exploitation",
            "2025-05-06 - Rise of Oriental Gudgeon",
            "2025-05-06 - Telegram Tango - Dancing with a Scammer",
            "2025-05-07 - Additional Features of OtterCookie Malware Used by WaterPlum",
            "2025-05-07 - COLDRIVER Using New Malware To Steal Documents From Western Targets and NGOs",
            "2025-05-07 - Iranian Cyber Actors Impersonate Model Agency in Suspected Espionage Operation",
            "2025-05-08 - Multilayered Email Attack - How a PDF Invoice and Geo-Fencing Led to RAT Malware",
            "2025-05-08 - Negotiations with the Akira ransomware group - an ill-advised approach",
            "2025-05-08 - Threat Analysis - SAP Vulnerability Exploited in the Wild by Chinese Threat Actor",
            "2025-05-09 - Classic Rock - Hunting a Botnet that preys on the Old",
            "2025-05-09 - Lumma Stealer, coming and going",
            "2025-05-12 - Analysis of APT37 Attack Case Disguised as a Think Tank for National Security Strategy in South Korea (Operation. ToyBox Story)",
            "2025-05-12 - Open-source toolset of an Ivanti CSA attacker",
            "2025-05-12 - Unveiling Swan Vector APT Targeting Taiwan and Japan with varied DLL Implants",
            "2025-05-13 - China-Nexus Nation State Actors Exploit SAP NetWeaver (CVE-2025-31324) to Target Critical Infrastructures",
            "2025-05-13 - Defining a new methodology for modeling and tracking compartmentalized threats",
            "2025-05-13 - Earth Ammit Disrupts Drone Supply Chains Through Coordinated Multi-Wave Attacks in Taiwan",
            "2025-05-13 - Sit, Fetch, Steal - Chihuahua Stealer - A new Breed of Infostealer",
            "2025-05-13 - TA406 Pivots to the Front",
            "2025-05-14 - Continued EAGERBEE (Thumtais) malware activity",
            "2025-05-14 - Technical Analysis of TransferLoader",
            "2025-05-15 - Ave Maria Malware Analysis",
            "2025-05-15 - Operation RoundPress",
            "2025-05-16 - DBatLoader (ModiLoader) Being Distributed to Turkish Users",
            "2025-05-16 - Printer company provided infected software downloads for half a year",
            "2025-05-16 - Ransomware Roundup – VanHelsing",
            "2025-05-17 - More_Eggs - A Venom Spider Backdoor Targeting HR",
            "2025-05-19 - A Sting on Bing - Bumblebee delivered through Bing SEO poisoning campaign",
            "2025-05-19 - Another Confluence Bites the Dust - Falling to ELPACO-team Ransomware",
            "2025-05-19 - Reversing a Microsoft-Signed Rootkit - The Netfilter Driver",
            "2025-05-20 - From banks to battalions - SideWinder’s attacks on South Asia’s public sector",
            "2025-05-21 - Disrupting Lumma Stealer - Microsoft leads global action against favored cybercrime tool",
            "2025-05-21 - The obfuscation game - MUT-9332 targets Solidity developers via malicious VS Code extensions",
            "2025-05-21 - TikTok Videos Promise Pirated Apps, Deliver Vidar and StealC Infostealers Instead",
            "2025-05-22 - Danabot- Analyzing a fallen empire",
            "2025-05-22 - De-obfuscating ALCATRAZ",
            "2025-05-22 - Russia-Aligned TAG-110 Targets Tajikistan with Macro-Enabled Word Documents",
            "2025-05-22 - UAT-6382 exploits Cityworks zero-day vulnerability to deliver malware",
            "2025-05-22 - ViciousTrap – Infiltrate, Control, Lure- Turning edge devices into honeypots en masse",
            "2025-05-23 - Mysterious hacking group Careto was run by the Spanish government, sources say",
            "2025-05-27 - Earth Lamia Develops Custom Arsenal to Target Multiple Industries",
            "2025-05-27 - Infostealer Malware FormBook Spread via Phishing Campaign – Part II",
            "2025-05-27 - Inside a VenomRAT Malware Campaign",
            "2025-05-27 - New Russia-affiliated actor Void Blizzard targets critical sectors for espionage",
            "2025-05-27 - SafePay - The new kid on the block",
            "2025-05-28 - Bombardino Crocodilo in Poland — analysis of IKO Lokaty mobile malware campaign",
            "2025-05-28 - GreyNoise Discovers Stealthy Backdoor Campaign Affecting Thousands of ASUS Routers",
            "2025-05-28 - Mark Your Calendar- APT41 Innovative Tactics",
            "2025-05-28 - NSIS Abuse and sRDI Shellcode - Anatomy of the Winos 4.0 Campaign",
            "2025-05-28 - Pakistan Telecommunication Company (PTCL) Targeted by Bitter APT During Heightened Regional Conflict",
            "2025-05-28 - PhaaS the Secrets - The Hidden Ties Between Tycoon2FA and Dadsec's Operations",
            "2025-05-28 - PumaBot - Novel Botnet Targeting IoT Surveillance Devices",
            "2025-05-29 - Chasing Eddies - New Rust-based InfoStealer used in CAPTCHA campaigns",
            "2025-05-29 - Deep Dive into a Dumped Malware without a PE Header",
            "2025-08-25 - Phishing Campaign Targeting Companies via UpCrypter",
            "2025-08-27 - Malicious Screen Connect Campaign Abuses AI-Themed Lures for Xworm Delivery",
            "2025-09-02 - Obscura an Obscure New Ransomware Variant",
            "2025-09-03 - Analyzing NotDoor Inside APT28’s Expanding Arsenal",
            "2025-09-03 - DragonForce Ransomware",
            "2025-09-03 - FANCY BEAR GONEPOSTAL – Espionage Tool Provides Backdoor Access to Microsoft Outlook",
            "2025-09-04 - Bells Ringing in Dar es Salaam",
            "2025-09-04 - New Botnet Emerges from the Shadows NightshadeC2",
            "2025-09-04 - North Korean Threat Actors Reveal Plans and Ops by Abusing Cyber Intel Platforms",
            "2025-09-05 - Unmasked Salat Stealer – A Deep Dive into Its Advanced Persistence Mechanisms and C2 Infrastructure",
            "2025-09-06 - Unknown Malware Using Azure Functions as C2",
            "2025-09-07 - APT37 Targets Windows with Rust Backdoor and Python",
            "2025-09-07 - ValleyRAT Exploiting BYOVD to Kill Endpoint Security",
            "2025-09-08 - Blurring the Lines Intrusion Shows Connection With Three Major Ransomware Gangs",
            "2025-09-08 - CyberVolk Ransomware Analysis of Double Encryption Structure and Disguised Decryption Logic",
            "2025-09-08 - MostereRAT Deployed AnyDeskTightVNC for Covert Full Access",
            "2025-09-08 - Off Your Docker Exposed APIs Are Targeted in New Malware Strain",
            "2025-09-09 - Agonizing Serpens (Aka Agrius) Targeting the Israeli Higher Education and Tech Sectors",
            "2025-09-09 - Analysis of Backdoor.WIN32.Buterat",
            "2025-09-09 - LunoBotnet A Self-Healing Linux Botnet with Modular DDoS and Cryptojacking Capabilities",
            "2025-09-09 - The Price of Free How Nulled Plugins Are Used to Weaken Your Defense",
            "2025-09-09 - Unmasking The Gentlemen Ransomware Tactics, Techniques, and Procedures Revealed",
            "2025-09-09 - ZynorRAT technical analysis Reverse engineering a novel, Turkish Go-based RAT",
            "2025-09-10 - AdaptixC2 A New Open-Source Framework Leveraged in Real-World Attacks",
            "2025-09-10 - EggStreme Malware Unpacking a New APT Framework Targeting a Philippine Military Company",
            "2025-09-21 - Block Blasters - Forensic Report",
            "2025-09-29 - Cybercrime Observations from the Frontlines UNC6040 Proactive Hardening Recommendations",
            "2025-10-06 - Massive Malicious NPM Package Attack Threatens Software Supply Chains",
            "2025-10-07 - 0-day vulnerability exploited by Cl0p patched by Oracle",
            "2025-10-07 - Phishing from Home The Hidden Danger in Remote Jobs Lurking in Tesla Google Ferrari and Glassdoor",
            "2025-10-08 - Exploring Invoice Fraud Email Attempts with Validin",
            "2025-10-08 - Oracle E-Business Suite Zero-Day Exploited in Widespread Extortion Campaign",
            "2025-10-09 - AdaptixC2 Uncovered Capabilities Tactics Hunting Strategies",
            "2025-10-09 - Inside Akira’s SonicWall Campaign Darktrace’s Detection and Response",
            "2025-10-09 - Inside a Crypto Scam Nexus"
        ]
    ])
    output_file = os.getenv('OUTPUT_FILE', 'pdf_index.json')
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
    logging.info("Open the browser, pass the CAPTCHA, then press Enter in the console...")
    driver.get(start_urls[0])
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
