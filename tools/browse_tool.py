import requests
from bs4 import BeautifulSoup

# Try to import trafilatura, fallback to BeautifulSoup if not available
try:
    import trafilatura
    HAS_TRAFILATURA = True
except ImportError:
    HAS_TRAFILATURA = False


class BrowseTool:
    """Downloads and cleans web pages."""

    def fetch_clean(self, url: str) -> str:
        try:
            resp = requests.get(url, timeout=20, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            resp.raise_for_status()
            html = resp.text
            
            # Use trafilatura if available, otherwise fallback to BeautifulSoup
            if HAS_TRAFILATURA:
                text = trafilatura.extract(
                    html, include_comments=False, include_tables=False
                )
            else:
                # Fallback: use BeautifulSoup
                soup = BeautifulSoup(html, 'lxml')
                # Remove script and style elements
                for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                    element.decompose()
                text = soup.get_text(separator='\n', strip=True)
                # Clean up extra whitespace
                lines = [line.strip() for line in text.splitlines() if line.strip()]
                text = '\n'.join(lines[:100])  # Limit to first 100 lines
            
            return text or ""
        except Exception as e:
            print(f"Browse error: {e}")
            return ""
