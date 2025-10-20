"""
Indexing PDFs from vx-underground.org using Selenium.
1. Opens the site in Chrome browser.
2. User manually passes the CAPTCHA.
3. After passing — automatically parses all PDF links and saves the index as JSON.
"""

import json
import os
import logging
from typing import List, Dict, Set, Optional
from urllib.parse import urljoin, quote
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
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        html = driver.page_source
        safe_url = base_url.replace('https://', '').replace('/', '_').replace(':', '')
        with open(f"scripts/debug_logs/debug_{safe_url}.html", "w", encoding="utf-8") as f:
            f.write(html)
        soup = BeautifulSoup(html, "html.parser")
    except Exception as e:
        logging.info(f"Error loading {base_url}: {e}")
        return []

    pdf_links = []
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
                                sub_url = driver.current_url
                                # Look for Paper folder
                                sub_soup = BeautifulSoup(driver.page_source, "html.parser")
                                sub_spans = sub_soup.find_all("span", class_="truncate")
                                for paper_span in sub_spans:
                                    if paper_span.get_text(strip=True) == "Paper":
                                        try:
                                            paper_elem = driver.find_element(By.XPATH, f"//span[contains(@class, 'truncate') and text()='Paper']")
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


def main():
    load_dotenv()
    os.makedirs('scripts/debug_logs', exist_ok=True)
    logging.basicConfig(filename='scripts/debug_logs/debug.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
        f"https://vx-underground.org/Malware%20Analysis/2025/{quote(item.split(' - ')[0].strip())}%20-%20{quote(item.split(' - ')[1].strip())}/Paper"
        for item in [
            "2025-01-02 - NonEuclid RAT",
            "2025-01-03 - RATs on the island (Remote Access Trojans in Sri Lanka's Cybersecurity Landscape)",
            "2025-01-03 - SwaetRAT Delivery Through Python",
            "2025-01-04 - Solara - Roblox Executor Malware"
        ]
    ])
    output_file = "pdf_index.json"
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
    completion_message = "Script completed successfully."
    print(completion_message)
    logging.info(completion_message)


if __name__ == "__main__":
    main()
