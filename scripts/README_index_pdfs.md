
# PDF Indexing Script for vx-underground.org

Minimalist script for crawling, downloading, and indexing all PDF files from <https://vx-underground.org>. Handles Cloudflare protection and stores files locally.

## Requirements

- Python 3.8+
- Virtual environment recommended
- Dependencies:
  - selenium
  - cloudscraper
  - beautifulsoup4
  - requests

Install dependencies:

```bash
pip install selenium cloudscraper beautifulsoup4 requests
```

## Environment Configuration

Create a `.env` file in the project root. Example (you can also view `.env.example`):

```text
CHROMIUM_PATH=/path/to/chrome.exe
CHROMEDRIVER_PATH=/path/to/chromedriver.exe
CHROME_USER_DATA_DIR=/path/to/chrome/user/data
CHROME_PROFILE=Profile 1
PDF_LIMIT=100
DEBUG_LOGS_DIR=scripts/debug_logs
DEBUG_LOG_FILE=scripts/debug.log
SELENIUM_TIMEOUT=10
OUTPUT_FILE=pdf_index.json
```

Variables:

- `CHROMIUM_PATH`: Absolute path to your Chrome or Chromium browser executable
- `CHROMEDRIVER_PATH`: Absolute path to ChromeDriver executable
- `CHROME_USER_DATA_DIR`: Path to Chrome user data directory (for profile management)
- `CHROME_PROFILE`: Name of the Chrome profile to use (e.g., "Profile 1")
- `PDF_LIMIT`: Maximum number of PDFs to download
- `DEBUG_LOGS_DIR`: Directory for debug logs
- `DEBUG_LOG_FILE`: Path to debug log file
- `SELENIUM_TIMEOUT`: Timeout (in seconds) for Selenium operations
- `OUTPUT_FILE`: Name of the output index file (default: pdf_index.json)

## Usage

Before running the script:

1. Open Chrome with the profile configured for Selenium.
2. Visit <https://vx-underground.org> and solve the CAPTCHA if prompted.
3. Close all Chrome windows.

Then run the script:

```bash
python scripts/index_pdfs_selenium.py
```

## Script Overview: index_pdfs_selenium.py

- Recursively crawls vx-underground.org for PDF links
- Bypasses Cloudflare using Selenium and cloudscraper
- Downloads all found PDFs to `downloaded_pdfs`
- Builds an index file `pdf_index.json` with:
  - `name`: PDF filename
  - `local_path`: Local file path
  - `url`: Original source URL

## Output Structure

`pdf_index.json` example:

```json
[
  {
    "name": "filename.pdf",
    "local_path": "downloaded_pdfs/filename.pdf",
    "url": "https://vx-underground.org/path/filename.pdf"
  }
]
```

## Notes

- Cloudflare protection is handled automatically, but manual review may be required if bypass fails.
- Crawling is limited by `MAX_PAGES` to avoid excessive traffic.
- Download time depends on the number of files.
