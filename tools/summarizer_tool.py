"""Summarization helper using the main LLM."""

from config.config import Config


class SummarizerTool:
    """Summarizes long texts using the LLM."""

    def __init__(self) -> None:
        self.llm = Config.get_llm()

    def summarize(self, text: str, max_words: int = 300) -> str:
        """
        Summarize the provided text.

        Args:
            text: Input text.
            max_words: Target summary length.

        Returns:
            Summary string.
        """
        prompt = (
            f"Summarize the following text in about {max_words} words:\n\n{text}"
        )
        resp = self.llm.invoke(prompt)
        return resp.content
