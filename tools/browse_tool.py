import requests
import trafilatura


class BrowseTool:
    """Downloads and cleans web pages."""

    def fetch_clean(self, url: str) -> str:
        try:
            resp = requests.get(url, timeout=20)
            resp.raise_for_status()
            html = resp.text
            text = trafilatura.extract(
                html, include_comments=False, include_tables=False
            )
            return text or ""
        except Exception:
            return ""
