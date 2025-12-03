# tools/knowledge_panel.py

import requests
from tavily import TavilyClient
from typing import Dict, List
import os


class KnowledgePanel:
    """
    Builds an entity knowledge panel similar to Perplexity:
    - Top image
    - Summary
    - Basic facts
    - Wikipedia link
    """

    def __init__(self):
        self.client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    def get_wikipedia_extract(self, query: str) -> Dict:
        """
        Returns summary + infobox data from Wikipedia.
        """
        try:
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '_')}"
            r = requests.get(url, timeout=10)
            data = r.json()

            return {
                "title": data.get("title", ""),
                "description": data.get("description", ""),
                "summary": data.get("extract", ""),
                "thumbnail": data.get("thumbnail", {}).get("source", ""),
                "url": data.get("content_urls", {}).get("desktop", {}).get("page", "")
            }
        except:
            return {}

    def get_fast_facts(self, query: str) -> List[str]:
        """
        Uses Tavily qna to extract AI-generated facts.
        """
        try:
            resp = self.client.qna(
                query=f"List 8 short bullet facts about {query}. No explanation, only facts.",
                n_tokens=150
            )
            answer = resp.get("answer", "")
            # Parse bullet points
            fact_lines = [line.strip("-• ").strip() for line in answer.split("\n") if line.strip()]
            return fact_lines[:8]  # Return max 8 facts
        except:
            return []

    def build_panel(self, query: str) -> Dict:
        """
        Builds the full knowledge panel.
        """
        wiki = self.get_wikipedia_extract(query)
        facts = self.get_fast_facts(query)

        return {
            "wiki": wiki,
            "facts": facts
        }
