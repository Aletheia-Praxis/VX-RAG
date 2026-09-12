# Vx-Underground Playwright Scraper

This is a high-performance, multithreaded scraper designed specifically for obtaining a massive `.pdf` dataset from [vx-underground.org](https://vx-underground.org).

## Features
- **Cloudflare Bypass:** Uses a persistent Chrome profile (via `playwright-stealth`) and runs in headed mode (`headless=False`) to avoid bot detection and "Verify you are human" pages.
- **Multithreading:** While the single Playwright crawler continuously maps the directory structure, an asynchronous `ThreadPoolExecutor` instantly triggers background downloads for discovered files.
- **Backpressure Protection:** A bounded queue (`maxsize=50`) limits the crawler's pace, guaranteeing that generated AWS S3 download links do not expire (`X-Amz-Expires`) before the background workers can fetch them.
- **Deduplication:** A SQLite database calculates and stores `SHA-256` hashes of every downloaded file, skipping duplicates even if they have different names or are in different folders.
- **Safe Recovery:** Progress is tracked in the SQLite DB (pending, processing, completed, error). If the connection fails or Cloudflare triggers a timeout, simply restart the script, and it will resume crawling only the `pending` or `error` directories.

## Prerequisites

1. Create a Python virtual environment and install dependencies:
```bash
pip install -r requirements.txt
playwright install chromium
```

2. **Create a local `.env` file** in this directory to configure your settings (use `.env.example` as a template if one exists, or create one manually):
```env
VXUG_BASE_URL="https://vx-underground.org"
VXUG_S3_HOST="s3.us-east-005.backblazeb2.com"
VXUG_USER_DATA_DIR="C:\\path\\to\\your\\chrome\\profile" # Mandatory
VXUG_MAX_WORKERS="10"
# VXUG_DATA_DIR="D:\\path\\to\\custom\\data\\dir" # Optional, defaults to project root
```

*Note: `VXUG_USER_DATA_DIR` is highly sensitive and mandatory. It must point to a persistent, valid Chrome profile on your disk that has successfully passed Cloudflare challenges.*

## Usage

**CRITICAL:** Before running the script, ensure that ALL Google Chrome browser windows (especially the one tied to `VXUG_USER_DATA_DIR`) are completely closed! Otherwise, Playwright will crash with a lock error.

Run the scraper:
```bash
python playwright_scraper.py
```

### Dealing with Errors (Cloudflare Captcha)
If you see `[Crawler] Cloudflare challenge detected!`, the script will pause for 5 minutes. You must manually click the "Verify you are human" checkbox in the opened Chrome window. Once verified, the script will automatically continue.

If the script stops or is interrupted, simply run it again. To retry folders that previously resulted in errors, run the following SQL command against `vxug_index.db`:
```sql
UPDATE queue SET status = 'pending' WHERE status = 'error';
```
