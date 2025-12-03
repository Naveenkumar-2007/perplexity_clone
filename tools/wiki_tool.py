"""Wikipedia search tool."""

from langchain_community.utilities import WikipediaAPIWrapper


class WikiTool:
    """Wrapper for Wikipedia-based QA."""

    def __init__(self) -> None:
        self.api = WikipediaAPIWrapper(top_k_results=3, lang="en")

    def query(self, query: str) -> str:
        """Search Wikipedia and return a summarized answer."""
        return self.api.run(query)
