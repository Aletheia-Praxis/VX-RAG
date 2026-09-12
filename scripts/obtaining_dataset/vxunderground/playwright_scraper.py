import os
import sqlite3
import hashlib
import requests
import time
import shutil
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Any
from playwright.sync_api import sync_playwright, Page
from playwright_stealth import Stealth
from dotenv import load_dotenv

# Load local environment variables from the script's directory
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(ENV_PATH)
BASE_URL = os.getenv("VXUG_BASE_URL", "https://vx-underground.org")
S3_HOST = os.getenv("VXUG_S3_HOST", "s3.us-east-005.backblazeb2.com")

# Chrome Profile for bypassing Cloudflare (must close Chrome before running!)
USER_DATA_DIR = os.getenv("VXUG_USER_DATA_DIR")
if not USER_DATA_DIR:
    raise ValueError("Critical Error: VXUG_USER_DATA_DIR must be set in the .env file. It should point to a valid Chrome profile.")

# Directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
default_data_dir = str(PROJECT_ROOT / "data" / "dataset" / "vxunderground")
DATA_DIR = Path(os.getenv("VXUG_DATA_DIR", default_data_dir))
DB_PATH = DATA_DIR / "vxug_index.db"
DOWNLOAD_DIR = DATA_DIR / "downloads"
TMP_DIR = DATA_DIR / "tmp"

DATA_DIR.mkdir(parents=True, exist_ok=True)
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)

# Concurrency globals
DB_LOCK = threading.Lock()
# Max 50 items in queue to prevent S3 link expiration
DOWNLOAD_QUEUE = queue.Queue(maxsize=50)
SHUTDOWN_FLAG = threading.Event()
MAX_WORKERS = int(os.getenv("VXUG_MAX_WORKERS", "10"))


def init_db() -> sqlite3.Connection:
    """
    Initialize the SQLite database for queue and file indexing.
    
    Creates `queue` and `files` tables if they don't exist. Inserts the root path
    into the queue if it's completely empty.
    
    Returns:
        sqlite3.Connection: The database connection object.
    """
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS queue (
            path TEXT PRIMARY KEY,
            status TEXT DEFAULT 'pending'
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS files (
            hash TEXT PRIMARY KEY,
            path TEXT,
            size INTEGER,
            download_date DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Insert root if queue is empty
    c.execute("SELECT count(*) FROM queue")
    if c.fetchone()[0] == 0:
        c.execute("INSERT INTO queue (path, status) VALUES (?, ?)", ("/", "pending"))
    conn.commit()
    return conn


def calculate_sha256(filepath: Path) -> str:
    """
    Calculate the SHA-256 hash of a file.
    
    Args:
        filepath (Path): The path to the file.
        
    Returns:
        str: The hex digest of the SHA-256 hash.
    """
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def download_file(url: str, temp_path: Path) -> bool:
    """
    Download a file from a URL to a temporary location using streaming.
    
    Args:
        url (str): The URL of the file to download.
        temp_path (Path): The temporary path where the file will be saved.
        
    Returns:
        bool: True if download was successful, False otherwise.
    """
    print(f"[Worker] Downloading {url.split('?')[0].split('/')[-1]}...")
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        with open(temp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"[Worker] Error downloading {url}: {e}")
        return False


def process_file(url: str, filename: str, folder_path: str, conn: sqlite3.Connection) -> None:
    """
    Download a file, calculate its hash, and deduplicate it before saving to the final directory.
    
    If the file hash already exists in the database, the temporary file is deleted.
    Otherwise, it is moved to the final structured directory.
    
    Args:
        url (str): The URL of the file.
        filename (str): The name of the file.
        folder_path (str): The logical path on the website to replicate locally.
        conn (sqlite3.Connection): Database connection.
    """
    # Ensure thread-safe unique temp file names
    thread_id = threading.get_ident()
    temp_file_path = TMP_DIR / f"{thread_id}_{filename}"
    
    if not download_file(url, temp_file_path):
        return

    file_hash = calculate_sha256(temp_file_path)
    file_size = temp_file_path.stat().st_size
    
    with DB_LOCK:
        c = conn.cursor()
        c.execute("SELECT hash FROM files WHERE hash = ?", (file_hash,))
        duplicate = c.fetchone() is not None

    if duplicate:
        print(f"[Worker] Duplicate found for hash {file_hash[:8]}..., skipping.")
        temp_file_path.unlink() # Delete temp file
    else:
        # Move to final destination maintaining folder structure
        final_dir = DOWNLOAD_DIR / folder_path.lstrip('/')
        final_dir.mkdir(parents=True, exist_ok=True)
        final_path = final_dir / filename
        
        # In case a file with same name exists, we append the hash
        if final_path.exists():
            final_path = final_dir / f"{temp_file_path.stem}_{file_hash[:8]}{temp_file_path.suffix}"
            
        shutil.move(str(temp_file_path), str(final_path))
        print(f"[Worker] Saved new file {filename} (Hash: {file_hash[:8]}...)")
        
        with DB_LOCK:
            c = conn.cursor()
            c.execute("INSERT INTO files (hash, path, size) VALUES (?, ?, ?)", 
                     (file_hash, str(final_path.relative_to(DOWNLOAD_DIR)), file_size))
            conn.commit()


def download_worker(conn: sqlite3.Connection) -> None:
    """Background worker thread that consumes download tasks from the queue."""
    while not SHUTDOWN_FLAG.is_set():
        try:
            # Block for up to 3 seconds waiting for a task
            task = DOWNLOAD_QUEUE.get(timeout=3)
            url, filename, folder_path = task
            process_file(url, filename, folder_path, conn)
            DOWNLOAD_QUEUE.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"[Worker] Unexpected error: {e}")


def crawl_path(page: Page, path: str, conn: sqlite3.Connection) -> None:
    """
    Crawl a specific path on the website using Playwright, enqueuing new directories and 
    adding files to the multithreaded download queue.
    
    Args:
        page (Page): The active Playwright page instance.
        path (str): The path to crawl (e.g., '/Papers').
        conn (sqlite3.Connection): Database connection.
    """
    url = f"{BASE_URL}{path}"
    print(f"[Crawler] Crawling {url}")
    
    try:
        # Use domcontentloaded instead of networkidle because Phoenix LiveView uses persistent WebSockets
        page.goto(url, wait_until="domcontentloaded", timeout=60000)
        
        # Check for Cloudflare challenge
        try:
            page.wait_for_selector("table tbody tr", timeout=15000)
        except Exception:
            if page.query_selector("iframe[src*='cloudflare']") or "verify you are human" in page.content().lower():
                print("[Crawler] Cloudflare challenge detected! Please solve it manually in the browser window.")
                page.wait_for_selector("table tbody tr", timeout=300000) # Wait 5 mins for manual solve
            else:
                page.screenshot(path=str(TMP_DIR / "error_screenshot.png"))
                print(f"[Crawler] No table found or empty directory. Saved screenshot to {TMP_DIR / 'error_screenshot.png'}")
                with DB_LOCK:
                    c = conn.cursor()
                    c.execute("UPDATE queue SET status = 'completed' WHERE path = ?", (path,))
                    conn.commit()
                return

        rows = page.query_selector_all("table tbody tr")
            
        for row in rows:
            size_cell = row.query_selector("td:nth-child(2)")
            size_text = size_cell.inner_text().strip() if size_cell else "–"
            
            name_span = row.query_selector("span.text-name")
            name_text = name_span.inner_text().strip() if name_span else ""

            # If the size cell has any digits, it's a file. Otherwise it's a folder (e.g. '-')
            if any(char.isdigit() for char in size_text):
                # Target only .pdf for now
                if name_text.lower().endswith('.pdf'):
                    download_link = row.query_selector(f"td:last-child a[href*='{S3_HOST}']")
                    if not download_link:
                        download_link = row.query_selector("a[aria-label^='Download']")
                        
                    if download_link:
                        dl_url = download_link.get_attribute("href")
                        if dl_url:
                            # Put in queue; blocks crawler if queue is full (maxsize=50)
                            print(f"[Crawler] Queuing {name_text} for download...")
                            DOWNLOAD_QUEUE.put((dl_url, name_text, path))
            else:
                # Add folder to queue
                if name_text and name_text != "..":
                    folder_path = path.rstrip('/') + '/' + name_text.strip('/')
                    if folder_path != path and folder_path != "/":
                        with DB_LOCK:
                            c = conn.cursor()
                            c.execute("INSERT OR IGNORE INTO queue (path, status) VALUES (?, ?)", (folder_path, "pending"))
        
        with DB_LOCK:
            c = conn.cursor()
            c.execute("UPDATE queue SET status = 'completed' WHERE path = ?", (path,))
            conn.commit()
        
    except Exception as e:
        print(f"[Crawler] Error crawling {path}: {e}")
        with DB_LOCK:
            c = conn.cursor()
            c.execute("UPDATE queue SET status = 'error' WHERE path = ?", (path,))
            conn.commit()


def main() -> None:
    """
    Main execution entry point. Initializes DB, starts background workers, 
    and begins crawling from the queue using Playwright.
    """
    conn = init_db()
    
    # Start background download workers
    print(f"Starting {MAX_WORKERS} background download workers...")
    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    for _ in range(MAX_WORKERS):
        executor.submit(download_worker, conn)
    
    try:
        with sync_playwright() as p:
            print(f"Connecting to personal Chrome profile at: {USER_DATA_DIR}")
            print("IMPORTANT: Make sure all Google Chrome windows are CLOSED before running this script!")
            
            # Using launch_persistent_context with channel="chrome" uses the real installed Google Chrome
            # This is the most reliable way to bypass Cloudflare.
            context = p.chromium.launch_persistent_context(
                user_data_dir=USER_DATA_DIR,
                channel="chrome",
                headless=False, # Must be False for Cloudflare to pass easily
                args=["--disable-blink-features=AutomationControlled"],
                viewport={"width": 1920, "height": 1080}
            )
            # Persistent context already has at least one page open
            page = context.pages[0] if context.pages else context.new_page()
            Stealth().apply_stealth_sync(page)
            
            while True:
                with DB_LOCK:
                    c = conn.cursor()
                    c.execute("SELECT path FROM queue WHERE status = 'pending' LIMIT 1")
                    row = c.fetchone()
                
                if not row:
                    print("Queue is empty. Waiting for final downloads to finish...")
                    DOWNLOAD_QUEUE.join()
                    print("Crawling finished.")
                    break
                    
                current_path = row[0]
                with DB_LOCK:
                    c = conn.cursor()
                    c.execute("UPDATE queue SET status = 'processing' WHERE path = ?", (current_path,))
                    conn.commit()
                
                try:
                    crawl_path(page, current_path, conn)
                except Exception as e:
                    print(f"Error crawling {current_path}: {e}")
                    with DB_LOCK:
                        c = conn.cursor()
                        c.execute("UPDATE queue SET status = 'failed' WHERE path = ?", (current_path,))
                        conn.commit()
                    
            context.close()
    except KeyboardInterrupt:
        print("\nCrawler stopped by user. Waiting for active downloads to finish (press Ctrl+C again to force quit)...")
    finally:
        SHUTDOWN_FLAG.set()
        executor.shutdown(wait=True)
        print("Shutdown complete.")


if __name__ == "__main__":
    main()
