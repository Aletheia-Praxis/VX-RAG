
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

Create a `.env` file in the project root. Example:

```text
BASE_URL=https://vx-underground.org
DOWNLOAD_DIR=downloaded_pdfs
SELENIUM_DRIVER_PATH=bin/chromedriver/chromedriver.exe
MAX_PAGES=100
```

Variables:

- `BASE_URL`: Target site URL
- `DOWNLOAD_DIR`: Local directory for PDFs
- `SELENIUM_DRIVER_PATH`: Path to ChromeDriver
- `MAX_PAGES`: Limit for recursive crawling

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
