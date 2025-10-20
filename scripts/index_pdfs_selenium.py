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
        with open(f"debug_{safe_url}.html", "w", encoding="utf-8") as f:
            f.write(html)
        soup = BeautifulSoup(html, "html.parser")
    except Exception as e:
        print(f"Error loading {base_url}: {e}")
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
            print(f"[FOUND] PDF: {name} @ {base_url}")
    print(f"[INFO] {found_pdfs} PDF(s) found in {base_url}")

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
                                            print(f"[WARN] Could not click Paper folder: {e}")
                                driver.back()
                            except Exception as e:
                                print(f"Could not click subfolder {subfolder_name}: {e}")
                    driver.back()
                except Exception as e:
                    print(f"Could not click year folder {folder_name}: {e}")
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
                    print(f"Could not click folder {folder_name}: {e}")
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
                    print(f"Could not click year folder {folder_name}: {e}")
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
                folder_elem = driver.find_element(By.XPATH, f"//span[contains(@class, 'truncate') and text()='{folder_name}']")
                folder_elem.click()
                time.sleep(1.5)
                WebDriverWait(driver, 7).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
                new_url = driver.current_url
                pdf_links.extend(get_all_pdf_links(driver, new_url, visited))
                driver.back()
            except Exception as e:
                print(f"Could not click folder {folder_name}: {e}")
    return pdf_links


def main():
    load_dotenv()
    # Add specific paths to folders with PDFs
    start_urls = []
    # Archive/The Old New Thing: years 2003–2025
    for year in range(2003, 2025):
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
        "https://vx-underground.org/Malware%20Analysis/2010/2010-12-20%20-%20End%20of%20the%20Line%20for%20the%20Bredolab%20Botnet/Paper"
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
        f"https://vx-underground.org/Malware%20Analysis/2025/{name.strip().replace(' ', '%20').replace('–', '-').replace('—', '-').replace('’', "'").replace('’', "'")}/Paper"
        for name in [
            "NonEuclid RAT",
            "RATs on the island (Remote Access Trojans in Sri Lanka's Cybersecurity Landscape)",
            "SwaetRAT Delivery Through Python",
            "Solara - Roblox Executor Malware",
            "EAGERBEE, with updated and novel components, targets the Middle East",
            "Hangro - Investigating North Korean VPN Infrastructure Part 1",
            "PacketCrypt Classic Cryptocurrency Miner on PHP Servers",
            "Turla Cyber Campaign Targeting Pakistan’s Critical Infrastructure",
            "Unveiling Russian Surveillance Tech Expansion in Central Asia and Latin America",
            "Akira Ransomware Group & Malware Analysis Report",
            "TMPN (Skuld) Stealer - The dark side of open source",
            "Hackers claim to breach Russian state agency managing property, land records",
            "HexaLocker V2 - Skuld Stealer Paving the Way prior to Encryption",
            "FunkSec – Alleged Top Ransomware Group Powered by AI",
            "Abusing AWS Native Services- Ransomware Encrypting S3 Buckets with SSE-C",
            "Double-Tap Campaign - Russia-nexus APT possibly related to APT28 conducts cyber espionage on Central Asia and Kazakhstan diplomatic relations",
            "From Royal to BlackSuit",
            "Justice Department and FBI Conduct International Operation to Delete Malware Used by China-Backed Hackers",
            "More Than Malware Families- Retooling Our Approach to Tracking Software",
            "One Mikro Typo - How a simple DNS misconfiguration enables malware delivery by a Russian botnet",
            "Russia's largest platform for state procurement hit by cyberattack from pro-Ukraine group",
            "Article 113- One of the Russian-Ukrainian cyberwars, a review of the first major blackout in Ukraine caused by the Sandworm APT organization",
            "BabbleLoader - A Deep Dive into EDR and Machine Learning-Based Endpoint Protection Evasion",
            "F.A.C.C.T. found new attacks of pro-Ukrainian cyber spies Sticky Werewolf",
            "Zombies Never Die - Analysis of the Current Situation of Large Botnet AIRASHI",
            "Analysis of Threat Actor Data Posting",
            "FortiGate Firewall Configs Dumped- Revisiting CVE-2022-40684 Exploitation",
            "Lazarus APT - Techniques for Hunting Contagious Interview",
            "MintsLoader - StealC and BOINC Delivery",
            "New Star Blizzard spear-phishing campaign targets WhatsApp accounts",
            "Will the Real Volt Typhoon Please Stand Up",
            "APT actor classification “addiction” - Practical issues of attribution seen in Lazarus subgroup classification",
            "Qbot is Back.Connect",
            "Love and hate under war - The GamaCopy organization, which imitates the Russian Gamaredon, uses military — related bait to launch attacks on Russia",
            "Silent Lynx APT Targets Various Entities Across Kyrgyzstan & Neighbouring Nations",
            "Categorizing Software with Code Families",
            "PlushDaemon compromises supply chain of Korean VPN service",
            "Cluster of Infrastructure likely used by Affiliate of Dark Scorpius (Black Basta)",
            "Helldown Ransomware Malware Analysis Report",
            "Lumma Stealer - Fake CAPTCHAs & New Techniques to Evade Detection",
            "RID Hijacking Technique Utilized by Andariel Attack Group",
            "The J-Magic Show - Magic Packets and Where to find them",
            "Sophos MDR tracks two ransomware campaigns using “email bombing,” Microsoft Teams “vishing”",
            "Cobalt Strike and a Pair of SOCKS Lead to LockBit Ransomware",
            "Technical Analysis of Xloader Versions 6 and 7 - Part 1",
            "North Korean APT Lazarus Targets Developers with Malicious npm Package",
            "Operation Phantom Circuit - North Korea’s Global Data Exfiltration Campaign",
            "Backdoor found in two healthcare patient monitors, linked to IP in China",
            "Coyote Banking Trojan - A Stealthy Attack via LNK Files",
            "Cybercrime websites selling hacking tools to transnational organized crime groups seized",
            "One ClickFix and LummaStealer reCAPTCHA’s Our Attention - Part 1",
            "Ongoing Email Bombing Campaigns leading to Remote Access and Post-Exploitation",
            "TAG-124’s Multi-Layered TDS Infrastructure and Extensive User Base",
            "UAC-0063 - Cyber Espionage Operation Expanding from Central Asia",
            "Attackers Leveraging Microsoft Teams Defaults and Quick Assist for Social Engineering Attacks",
            "Do the CONTEC CMS8000 Patient Monitors Contain a Chinese Backdoor - The Reality is More Complicated…",
            "LegionLoader exposed",
            "macOS FlexibleFerret - Further Variants of DPRK Malware Family Unearthed",
            "Analyzing ELF-Sshdinjector.A!tr with a Human and Artificial Analyst",
            "CVE-2025-0411 - Ukrainian Organizations Targeted in Zero-Day Campaign and Homoglyph Attacks",
            "Unpacking the BADBOX Botnet with Censys",
            "Lazarus Group Targets Organizations with Sophisticated LinkedIn Recruiting Scam",
            "Stealthy Attack - Dual Injection Undermines Chrome’s App-Bound Encryption",
            "Code injection attacks using publicly disclosed ASP.NET machine keys",
            "Google Tag Manager Skimmer Steals Credit Card Info From Magento Site",
            "SI-CERT TZ016 - BeaverTail & InvisibleFerret",
            "Analysis of malicious mobile applications impersonating popular Polish apps — OLX, Allegro, IKO",
            "Further insights into Ivanti CSA 4.6 vulnerabilities exploitation",
            "Tracking Ransomware - January 2025",
            "RATatouille - Cooking Up Chaos in the I2P Kitchen",
            "Sandworm APT Exploits Trojanized KMS Tools to Target Ukrainian Users in Cyber Espionage Campaign",
            "Sandworm APT Targets Ukrainian Users with Trojanized Microsoft KMS Activation Tools in Cyber Espionage Campaigns",
            "BTMOB RAT - Newly Discovered Android Malware Spreading via Phishing Sites",
            "Defying tunneling - A Wicked approach to detecting malicious network traffic",
            "North Korean Hackers Exploit PowerShell Trick to Hijack Devices in New Cyberattack",
            "Surge in attacks exploiting old ThinkPHP and ownCloud flaws",
            "Suspected North Korean hacker hacks a large number of data from a government document system developer",
            "The BadPilot campaign - Seashell Blizzard subgroup conducts multiyear global access operation",
            "Two tales and one Antidot(e) — a new mobile malware campaign in Poland",
            "Unpacking Pyarmor v8+ scripts",
            "Analyzing DEEP#DRIVE- North Korean Threat Actors Observed Exploiting Trusted Platforms for Targeted Attacks",
            "China-linked Espionage Tools Used in Ransomware Attacks",
            "Cybercrooks Are Using Fake Job Listings to Steal Crypto",
            "From South America to Southeast Asia - The Fragile Web of REF7707",
            "Inside the Scam - North Korea’s IT Worker Threat",
            "Multiple Russian Threat Actors Targeting Microsoft Device Code Authentication",
            "RedMike (Salt Typhoon) Exploits Vulnerable Cisco Devices of Global Telecommunications Providers",
            "Storm-2372 conducts device code phishing campaign",
            "Technical Analysis of Xloader Versions 6 and 7 - Part 2",
            "Threat hunting case study - SocGholish",
            "You've Got Malware - FINALDRAFT Hides in Your Drafts",
            "Dissecting a fresh BlankGrabber sample",
            "An Update on Fake Updates - Two New Actors, and New Mac Malware",
            "An inside look at NSA (Equation Group) TTPs from China’s lense",
            "Exposing the Deceit - Phishing Sites Impersonating Government Entities",
            "IOCs Green Nailao campaign (NailaoLocker, ShadowPad)",
            "#StopRansomware - Ghost (Cring) Ransomware",
            "Technical Analysis of Lockbit4.0 Evasion Tales",
            "The Pangu Team—iOS Jailbreak and Vulnerability Research Giant- A Member of i-SOON’s Exploit-Sharing Network",
            "48 Minutes - How Fast Phishing Attacks Exploit Weaknesses",
            "APT-C-28 Group Launched New Cyber Attack With Fileless RokRat Malware",
            "DeceptiveDevelopment targets freelance developers",
            "GhostSocks - Lumma's Partner In Proxy",
            "Linkc Ransomware - The New Cybercriminal Group Targeting Artificial Intelligence Data",
            "Meet NailaoLocker - a ransomware distributed in Europe by ShadowPad and PlugX backdoors",
            "Updated Shadowpad Malware Leads to Ransomware Deployment",
            "Weathering the storm - In the midst of a Typhoon",
            "Angry Likho - Old beasts in a new forest",
            "How’s that for a malicious Linkc, new group launches DLS",
            "TRM Links North Korea to Record $1.5 Billion Record Hack",
            "Android trojan TgToxic updates its capabilities",
            "Auto-Color - An Emerging and Evasive Linux Backdoor",
            "Cryptocurrency APT Intelligence - Unveiling Lazarus Group’s Intrusion Techniques",
            "LCRYX Ransomware - How a VB Ransomware Locks Your System",
            "Six Months Undetected - Analysis of archive.org hosted .NET PE Injector",
            "The GitVenom campaign - Cryptocurrency theft using GitHub",
            "Ghostwriter - New Campaign Targets Ukrainian Government and Belarusian Opposition",
            "PolarEdge - Unveiling an uncovered ORB network",
            "Alert Number - I-022625-PSA - North Korea Responsible for $1.5 Billion Bybit Hack",
            "Inside BlackBasta - What Leaked Conversations Reveal About Their Ransomware Operations",
            "BlackBasta Leaks - Lessons from the Ascension Health attack",
            "Disrupting a global cybercrime network abusing generative AI",
            "Long Live The Vo1d Botnet - New Variant Hits 1.6 Million TV Globally",
            "Modern Approach to Attributing Hacktivist Groups",
            "NailaoLoader - Hiding Execution Flow via Patching",
            "NanoCore Malware Analysis",
            "Phishing Email Attacks by the Larva-24005 Group Targeting Japan",
            "Russian campaign targeting Romanian WhatsApp numbers",
            "Squidoor - Suspected Chinese Threat Actor’s Backdoor Targets Global Organizations",
            "The Rise of the Fake Tech Workforce - State-Sponsored Infiltration of U.S. Technical Supply Chains",
            "Winos 4.0 Spreads via Impersonation of Official Email to Target Users in Taiwan",
            "Agent AI, Basta Parser Extraordinaire",
            "Black Basta exposed - A look at a cybercrime data leak",
            "JavaGhost’s Persistent Phishing Attacks From the Cloud",
            "New DDoS Botnet Discovered - Over 30,000 Hacked Devices, Majority of Observed Activity Traced to Iran",
            "Notorious Malware, Spam Host “Prospero” Moves to Kaspersky Lab",
            "An in-depth analysis of APT37’s latest campaign",
            "Ransomware - de REvil à Black Basta, que sait-on de Tramp",
            "Pivoting on Black Basta's (leaked) Infrastructure",
            "Black Basta and Cactus Ransomware Groups Add BackConnect Malware to Their Arsenal",
            "PureLogs Deep Analysis- Evasion, Data Theft, and Encryption Mechanism",
            "Analysis of Kimsuky Group association with emergency martial arts-themed APT attack",
            "Black Basta Leak Analysis",
            "Likely DPRK Network Backstops on GitHub, Targets Companies Globally",
            "Ragnar Loader Indicators of Compromise (IOC)",
            "Thousands of websites hit by four backdoors in 3rd party JavaScript attack",
            "Tracking Emmenhtal",
            "Initial Takeaways from the Black Basta Chat Leaks",
            "Satori Threat Intelligence Disruption - BADBOX 2.0 Targets Consumer Devices with Multiple Fraud Schemes",
            "Silk Typhoon targeting IT supply chain",
            "Water Ouroboros",
            "Deciphering Black Basta’s Infrastructure from the Chat Leak",
            "The Next Level - Typo DGAs Used in Malicious Redirection Chains",
            "Unveiling EncryptHub - Analysis of a multi-stage malware campaign",
            "Akira Ransomware Expands to Linux - The attacking abilities and strategies",
            "Remote Monitoring and Management (RMM) Tooling Increasingly an Attacker’s First Choice",
            "Blind Eagle- …And Justice for All",
            "DieNet and #Shiite_Harvest claimed responsibility for disabling ten significant Iraqi websites",
            "Lazarus Strikes npm Again with New Wave of Malicious Packages",
            "Trump Cryptocurrency Delivers ConnectWise RAT",
            "AI-Assisted Fake GitHub Repositories Fuel SmartLoader and LummaStealer Distribution",
            "Blind Eagle Hacks Colombian Institutions Using NTLM Flaw, RATs and GitHub-Based Attacks",
            "Cato CTRL Threat Research - Ballista – New IoT Botnet Targeting Thousands of TP-Link Archer Routers",
            "DCRat backdoor returns",
            "DragonForce Ransomware - Unveiling Its Tactics and Impact",
            "IOCs for Anubis Backdoor",
            "Ghost in the Router - China-Nexus Espionage Actor UNC3886 Targets Juniper Routers",
            "Golang backdoor with a side of ChromeUpdateAlert App",
            "Lookout Discovers New Spyware by North Korean APT37",
            "Medusa Ransomware",
            "Analyzing OBSCURE#BAT Threat Actors Lure Victims into Executing Malicious Batch Scripts to Deploy Stealthy Rootkits",
            "ArechClient; Decoding IOCs and finding the onboard browser extension",
            "Botnets never die",
            "Decrypting Encrypted files from Akira Ransomware (Linux-ESXI variant 2024) using a bunch of GPUs",
            "Inside BRUTED - Black Basta (RaaS) Members Used Automated Brute Forcing Framework to Target Edge Network Devices",
            "New Ransomware Operator Exploits Fortinet Vulnerability Duo",
            "Tracking Ransomware - February 2025",
            "Work Hard, Pay Harder!",
            "Android Banking Trojan – OctoV2, masquerading as Deepseek AI",
            "Lumma Stealer – A tale that starts with a fake Captcha",
            "SocGholish’s Intrusion Techniques Facilitate Distribution of RansomHub Ransomware",
            "Understanding SalatStealer - Features and Impact",
            "Analyzing the RedTiger Malware Stealer",
            "Bybit – What We Know So Far",
            "Black Basta’s blunder - exploiting the gang’s leaked chats",
            "DollyWay World Domination - Eight Years of Evolving Website Malware Campaigns",
            "Code-signing certificate abuse in the Black Basta chat leaks (and how to fight back)",
            "Operation AkaiRyū - MirrorFace invites Europe to Expo 2025 and revives ANEL backdoor",
            "Operation FishMedley",
            "Reversing FUD AMOS Stealer",
            "UAT-5918 targets critical infrastructure entities in Taiwan",
            "Back to Business - Lumma Stealer Returns with Stealthier Methods",
            "Analyzing Vidar Stealer",
            "Weaver Ant, the Web Shell Whisperer - Tracking a Live China-nexus Operation",
            "IBM X-Force discovers new Sheriff Backdoor used to target Ukraine",
            "Inside DollyWay’s C2 Infrastructure - Traffic Direction Systems and the LosPollos Connection",
            "Inside Kimsuky’s Latest Cyberattack - Analyzing Malicious Scripts and Payloads",
            "On the Hunt for Ghost(Socks)",
            "Operation ForumTroll - APT attack with Google Chrome zero-day exploit chain",
            "Phishing Campaign Targets Defense and Aerospace Firms Linked to Ukraine Conflict",
            "Tempted to Classifying APT Actors- Practical Challenges of Attribution in the Case of Lazarus’s Subgroup",
            "CoffeeLoader - A Brew of Stealthy Techniques",
            "Lynx Ransomware - Learn details about the operation and how to mitigate this threat",
            "The Long and Short(cut) of It- KoiLoader Analysis",
            "A Phishing Tale of DoH and DNS MX Abuse",
            "A Deep Dive into Water Gamayun’s Arsenal and Infrastructure",
            "Exposing Crocodilus - New Device Takeover Malware Targeting Android Devices",
            "Hidden Malware Strikes Again - Mu-Plugins Under Attack",
            "TsarBot - A New Android Banking Trojan Targeting Over 750 Banking, Finance, and Cryptocurrency Applications",
            "Analyzing New HijackLoader Evasion Tactics",
            "CPU_HU - Fileless cryptominer targeting exposed PostgreSQL with over 1.5K victims",
            "DarkCloud Stealer",
            "From Contagious to ClickFake Interview - Lazarus leveraging the ClickFix tactic",
            "Gootloader Returns - Malware Hidden in Google Ads for Legal Documents",
            "Malware hiding in plain sight - Spying on North Korean Hackers",
            "Operation HollowQuill - Malware delivered into Russian R&D Networks via Research Decoy PDFs",
            "The Espionage Toolkit of Earth Alux - A Closer Look at its Advanced Techniques",
            "Auto-color - Linux backdoor",
            "Salvador Stealer - New Android Malware That Phishes Banking Details & OTPs",
            "Same Russian-Speaking Threat Actor, New Tactics Abuse of Cloudflare Services for Phishing and Telegram to Filter Victim IPs",
            "An in-depth look at Black Basta's TTPs",
            "BeaverTail and Tropidoor Malware Distributed via Recruitment Emails",
            "Tracking Adversaries - EvilCorp, the RansomHub affiliate",
            "Threat actors leverage tax season to deploy tax-themed phishing campaigns",
            "UAC-0219 Attack Detection - A New Cyber-Espionage Campaign Using a PowerShell Stealer WRECKSTEEL",
            "Lazarus Expands Malicious npm Campaign - 11 New Packages Add Malware Loaders and Bitbucket Payloads",
            "OPSEC Failure Exposes Coquettte's Malware Campaigns on Bulletproof Hosting Servers",
            "UAC-0226 Attack Detection - New Cyber-Espionage Campaign Targeting Ukrainian Innovation Hubs and Government Entities with GIFTEDCROOK Stealer",
            "Exploitation of CLFS zero-day leads to ransomware activity",
            "Goodbye HTA, Hello MSI- New TTPs and Clusters of an APT driven by Multi-Platform Attacks",
            "Inside DanaBot’s Infrastructure - In Support of Operation Endgame II",
            "State-Sponsored Tactics - How Gamaredon and ShadowPad Operate and Rotate Their Infrastructure",
            "GOFFEE continues to attack organizations in Russia",
            "Newly Registered Domains Distributing SpyNote Malware",
            "Flesh Stealer - A Report on Multivector Data Theft",
            "Interview with the Chollima",
            "Threat Spotlight - Hijacked and Hidden - New Backdoor and Persistence Technique",
            "BPFDoor’s Hidden Controller Used Against Asia, Middle East Targets",
            "New Malware Variant Identified - ResolverRAT Enters the Maze",
            "Proton66 Part 1 - Mass Scanning and Exploit Campaigns",
            "Slow Pisces Targets Developers With Coding Challenges and Introduces New Customized Python Malware",
            "CyberSOC Insights - Analysis of a Black Basta Attack Campaign",
            "Hunting Mice In Tunnels II - Fake CAPTCHAs and Ransomware",
            "Renewed APT29 Phishing Campaign Against European Diplomats",
            "UNC5174’s evolution in China’s ongoing cyber warfare- From SNOWLIGHT to VShell",
            "Inside Gamaredon’s PteroLNK - Dead Drop Resolvers and evasive Infrastructure",
            "Interlock ransomware evolving under the radar",
            "Around the World in 90 Days - State-Sponsored Actors Try ClickFix",
            "Breaking the B0 ransomware - Investigation & Decryption",
            "IronHusky updates the forgotten MysterySnail RAT to target Russia and Mongolia",
            "Mitigating ELUSIVE COMET Zoom remote control attacks",
            "Proton66 Part 2 - Compromised WordPress Pages and Malware Campaigns",
            "Unmasking the new XorDDoS controller and infrastructure",
            "Unmasking the Evolving Threat - A Deep Dive into the Latest Version of Lumma InfoStealer with Code Flow Obfuscation",
            "Distribution of PebbleDash Malware in March 2025",
            "Infostealer Malware FormBook Spread via Phishing Campaign – Part I",
            "Phishing for Codes - Russian Threat Actors Target Microsoft 365 OAuth Workflows",
            "Russian organizations targeted by backdoor masquerading as secure networking software updates",
            "AsyncRAT Malware Analysis",
            "Introducing ToyMaker, an initial access broker working in cahoots with double extortion gangs",
            "Russian Infrastructure Plays Crucial Role in North Korean Cybercrime Operations",
            "Understanding the threat landscape for Kubernetes and containerized assets",
            "Contagious Interview (DPRK) Launches a New Campaign Creating Three Front Companies to Deliver a Trio of Malware - BeaverTail, InvisibleFerret, and OtterCookie",
            "Crypters And Tools. Part 2- Different Paws — Same Tangle",
            "Understanding Alcatraz ~ Obfuscator Analysis [EN]",
            "Earth Kurma APT Campaign Targets Southeast Asian Government, Telecom Sectors",
            "Rolling in the Deep(Web) - Lazarus Tsunami",
            "The Persistent Threat of Salt Typhoon - Tracking Exposures of Potentially Targeted Devices",
            "Top Tier Target - What It Takes to Defend a Cybersecurity Company from Today’s Adversaries",
            "Uncovering Actor TTP Patterns and the Role of DNS in Investment Scams",
            "Gremlin Stealer - New Stealer on Sale in Underground Forum",
            "Nitrogen Dropping Cobalt Strike – A Combination of “Chemical Elements”",
            "Russia – Assignment of cyber attacks against France to the Russian military intelligence service (APT28) (29 April 2025)",
            "Uncovering MintsLoader With Recorded Future Malware Intelligence Hunting",
            "Yet Another NodeJS Backdoor (YaNB)- A Modern Challenge",
            "Advisory - Pahalgam Attack themed decoys used by APT36 to target the Indian Government",
            "Finding Malware - Unveiling LUMMAC.V2 with Google Security Operations",
            "Deep Dive Fog ransomware",
            "FortiGuard Incident Response Team Detects Intrusion into Middle East Critical National Infrastructure",
            "I StealC You - Tracking the Rapid Changes To StealC",
            "TerraStealerV2 and TerraLogger - Golden Chickens' New Malware Families Discovered",
            "Prelude - Crypto Heist Causes HAVOC",
            "Venom Spider Uses Server-Side Polymorphism to Weave a Web Around Victims",
            "Negotiations with the Akira ransomware group - an ill-advised approach",
            "Defending Against UNC3944 - Cybercrime Hardening Guidance from the Frontlines",
            "Here Comes Mirai - IoT Devices RSVP to Active Exploitation",
            "Rise of Oriental Gudgeon",
            "Telegram Tango - Dancing with a Scammer",
            "Additional Features of OtterCookie Malware Used by WaterPlum",
            "COLDRIVER Using New Malware To Steal Documents From Western Targets and NGOs",
            "Iranian Cyber Actors Impersonate Model Agency in Suspected Espionage Operation",
            "Multilayered Email Attack - How a PDF Invoice and Geo-Fencing Led to RAT Malware",
            "Negotiations with the Akira ransomware group - an ill-advised approach",
            "Threat Analysis - SAP Vulnerability Exploited in the Wild by Chinese Threat Actor",
            "Classic Rock - Hunting a Botnet that preys on the Old",
            "Lumma Stealer, coming and going",
            "Analysis of APT37 Attack Case Disguised as a Think Tank for National Security Strategy in South Korea (Operation. ToyBox Story)",
            "Open-source toolset of an Ivanti CSA attacker",
            "Unveiling Swan Vector APT Targeting Taiwan and Japan with varied DLL Implants",
            "China-Nexus Nation State Actors Exploit SAP NetWeaver (CVE-2025-31324) to Target Critical Infrastructures",
            "Defining a new methodology for modeling and tracking compartmentalized threats",
            "Earth Ammit Disrupts Drone Supply Chains Through Coordinated Multi-Wave Attacks in Taiwan",
            "Sit, Fetch, Steal - Chihuahua Stealer - A new Breed of Infostealer",
            "TA406 Pivots to the Front",
            "Continued EAGERBEE (Thumtais) malware activity",
            "Technical Analysis of TransferLoader",
            "Ave Maria Malware Analysis",
            "Operation RoundPress",
            "DBatLoader (ModiLoader) Being Distributed to Turkish Users",
            "Printer company provided infected software downloads for half a year",
            "Ransomware Roundup – VanHelsing",
            "More_Eggs - A Venom Spider Backdoor Targeting HR",
            "A Sting on Bing - Bumblebee delivered through Bing SEO poisoning campaign",
            "Another Confluence Bites the Dust - Falling to ELPACO-team Ransomware",
            "Reversing a Microsoft-Signed Rootkit - The Netfilter Driver",
            "From banks to battalions - SideWinder’s attacks on South Asia’s public sector",
            "Disrupting Lumma Stealer - Microsoft leads global action against favored cybercrime tool",
            "The obfuscation game - MUT-9332 targets Solidity developers via malicious VS Code extensions",
            "TikTok Videos Promise Pirated Apps, Deliver Vidar and StealC Infostealers Instead",
            "Danabot- Analyzing a fallen empire",
            "De-obfuscating ALCATRAZ",
            "Russia-Aligned TAG-110 Targets Tajikistan with Macro-Enabled Word Documents",
            "UAT-6382 exploits Cityworks zero-day vulnerability to deliver malware",
            "ViciousTrap – Infiltrate, Control, Lure- Turning edge devices into honeypots en masse",
            "Mysterious hacking group Careto was run by the Spanish government, sources say",
            "Earth Lamia Develops Custom Arsenal to Target Multiple Industries",
            "Infostealer Malware FormBook Spread via Phishing Campaign – Part II",
            "Inside a VenomRAT Malware Campaign",
            "New Russia-affiliated actor Void Blizzard targets critical sectors for espionage",
            "SafePay - The new kid on the block",
            "Phishing Campaign Targeting Companies via UpCrypter",
            "Malicious Screen Connect Campaign Abuses AI-Themed Lures for Xworm Delivery",
            "Obscura an Obscure New Ransomware Variant",
            "Analyzing NotDoor Inside APT28’s Expanding Arsenal",
            "DragonForce Ransomware",
            "FANCY BEAR GONEPOSTAL – Espionage Tool Provides Backdoor Access to Microsoft Outlook",
            "Bells Ringing in Dar es Salaam",
            "New Botnet Emerges from the Shadows NightshadeC2",
            "North Korean Threat Actors Reveal Plans and Ops by Abusing Cyber Intel Platforms",
            "Unmasked Salat Stealer – A Deep Dive into Its Advanced Persistence Mechanisms and C2 Infrastructure",
            "Unknown Malware Using Azure Functions as C2",
            "APT37 Targets Windows with Rust Backdoor and Python",
            "ValleyRAT Exploiting BYOVD to Kill Endpoint Security",
            "Blurring the Lines Intrusion Shows Connection With Three Major Ransomware Gangs",
            "CyberVolk Ransomware Analysis of Double Encryption Structure and Disguised Decryption Logic",
            "MostereRAT Deployed AnyDeskTightVNC for Covert Full Access",
            "Off Your Docker Exposed APIs Are Targeted in New Malware Strain",
            "Agonizing Serpens (Aka Agrius) Targeting the Israeli Higher Education and Tech Sectors",
            "Analysis of Backdoor.WIN32.Buterat",
            "LunoBotnet A Self-Healing Linux Botnet with Modular DDoS and Cryptojacking Capabilities",
            "The Price of Free How Nulled Plugins Are Used to Weaken Your Defense",
            "Unmasking The Gentlemen Ransomware Tactics, Techniques, and Procedures Revealed",
            "ZynorRAT technical analysis Reverse engineering a novel, Turkish Go-based RAT",
            "AdaptixC2 A New Open-Source Framework Leveraged in Real-World Attacks",
            "EggStreme Malware Unpacking a New APT Framework Targeting a Philippine Military Company",
            "Block Blasters - Forensic Report",
            "Cybercrime Observations from the Frontlines UNC6040 Proactive Hardening Recommendations",
            "Massive Malicious NPM Package Attack Threatens Software Supply Chains",
            "0-day vulnerability exploited by Cl0p patched by Oracle",
            "Phishing from Home The Hidden Danger in Remote Jobs Lurking in Tesla Google Ferrari and Glassdoor",
            "Exploring Invoice Fraud Email Attempts with Validin",
            "Oracle E-Business Suite Zero-Day Exploited in Widespread Extortion Campaign",
            "AdaptixC2 Uncovered Capabilities Tactics Hunting Strategies",
            "Inside Akira’s SonicWall Campaign Darktrace’s Detection and Response",
            "Inside a Crypto Scam Nexus"
        ]
    ])
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
