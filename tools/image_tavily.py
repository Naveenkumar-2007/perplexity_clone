# tools/image_tavily.py

import os
from tavily import TavilyClient
from typing import List, Dict


class TavilyImageSearch:
    """
    Tavily image search API wrapper.
    """

    def __init__(self):
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise RuntimeError("Missing TAVILY_API_KEY in environment")
        self.client = TavilyClient(api_key=api_key)

    def search(self, query: str, count: int = 6) -> List[Dict]:
        """
        Fetch images for a query.
        """

        try:
            # Try the correct Tavily API method - it's get_search_context with search_depth="advanced"
            resp = self.client.search(
                query=query,
                max_results=count,
                include_images=True,
                include_answer=False
            )
        except Exception as e:
            print("Tavily image search error:", e)
            return []

        images = []
        raw_images = resp.get("images", [])
        
        for item in raw_images:
            # Handle both dict and direct response formats
            if isinstance(item, dict):
                images.append({
                    "title": item.get("title", item.get("description", "")),
                    "thumbnail_url": item.get("thumbnail", item.get("thumbnail_url", item.get("url", ""))),
                    "content_url": item.get("url", item.get("content_url", "")),
                })
            else:
                # Fallback for string URLs
                images.append({
                    "title": "",
                    "thumbnail_url": str(item),
                    "content_url": str(item),
                })
        
        return images
