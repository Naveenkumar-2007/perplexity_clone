import os
from typing import List, Dict
import requests
from config.config import Config


class SearchTool:
    """Tavily web search wrapper."""

    def __init__(self) -> None:
        self.api_key = os.getenv("TAVILY_API_KEY") or Config.TAVILY_API_KEY
        if not self.api_key:
            raise RuntimeError("TAVILY_API_KEY missing in .env")

    def search(self, query: str, num_results: int = 5) -> List[Dict]:
        url = "https://api.tavily.com/search"
        payload = {"query": query, "num_results": num_results}
        headers = {"Authorization": self.api_key}
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            return data.get("results", [])
        except (requests.exceptions.RequestException, ValueError) as e:
            print(f"Search error: {e}")
            return []
