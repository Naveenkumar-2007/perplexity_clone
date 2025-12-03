from config.config import Config

class FollowUpGenerator:
    """
    Generate 3–5 follow-up suggestions like Perplexity.
    """

    def __init__(self):
        self.llm = Config.get_llm()

    def generate(self, answer: str, question: str):
        prompt = f"""
Given the user question and the assistant answer, generate 3 short follow-up questions the user might ask next.

Rules:
- Keep them brief (max 8–12 words)
- No numbered list
- No explanations
- Only return bullet points starting with "•"
- Must be relevant and helpful

User question: {question}
Assistant answer: {answer}

Generate follow-ups:
"""

        resp = self.llm.invoke(prompt).content
        lines = resp.strip().split("\n")

        # Only keep bullet lines
        suggestions = [l.replace("•", "").strip() for l in lines if "•" in l]
        return suggestions[:4]
