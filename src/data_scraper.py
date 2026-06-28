"""Data fetching utilities.

This module provides a small helper to download a CSV from a provided URL and save it locally.
It also contains a scraper template for the official site — adapt selectors as needed.
"""
from pathlib import Path
import requests
from typing import Optional


def download_csv_url(url: str, save_path: str | Path) -> Path:
    save_path = Path(save_path)
    resp = requests.get(url)
    resp.raise_for_status()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_bytes(resp.content)
    return save_path


def scrape_official_site_example(save_path: str | Path, base_url: str = "https://www.cwl.gov.cn/kjxx/ssq/") -> Optional[Path]:
    """Template function: implement scraping logic depending on the official site layout.

    NOTE: Many official sites use JS or protections; this is a template for manual adaptation.
    """
    # This is a placeholder. Implement with requests + BeautifulSoup or Selenium if needed.
    return None
