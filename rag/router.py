import re
from config.config import Config

class RouterAgent:
    """
    Production-grade router exactly like Perplexity:
    1. Rule-based fast routing
    2. NER-based entity detection
    3. Real-time classifier
    4. LLM semantic classifier (handles ANY query)
    """

    def __init__(self):
        self.llm = Config.get_llm()

    # ---------------- FAST RULES ----------------
    def contains(self, q, words):
        q = q.lower()
        return any(w in q for w in words)

    def is_greeting(self, q):
        q_low = q.lower().strip()
        return q_low in ["hi", "hello", "hey", "yo", "sup", "hi there", "hello there"]

    def is_image_query(self, q):
        image_words = ["image", "photo", "pic", "picture", "logo", "wallpaper", "screenshot"]
        return self.contains(q, image_words)

    def is_realtime(self, q):
        realtime = [
            "today", "now", "latest", "current",
            "price", "stock", "weather", "news",
            "update", "live", "score", "match", "schedule"
        ]
        return self.contains(q, realtime)

    def is_world_fact(self, q):
        patterns = [
            "prime minister", "president", "capital of",
            "ceo", "founder", "population", "richest",
            "oldest", "largest", "smallest", "currency",
            "country", "state", "city", "minister",
            "government", "party"
        ]
        return self.contains(q, patterns)

    def is_ai_model(self, q):
        ai_models = ["gpt", "gemini", "llama", "claude", "grok", "mistral", "phi"]
        return self.contains(q, ai_models)

    def is_definition(self, q):
        q = q.lower()
        return q.startswith(("what is", "define", "explain"))

    def is_deep(self, q):
        q = q.lower()
        return any(x in q for x in [
            "compare", "analysis", "impact", "advantages", "disadvantages",
            "evaluate", "future", "strategy", "risk"
        ])

    def is_entity(self, q):
        """Detects entities by uppercase words"""
        words = q.split()
        caps = [w for w in words if w[:1].isupper()]
        return len(caps) >= 1

    # ---------------- LLM CLASSIFIER ----------------
    def llm_decide(self, q):
        """
        FINAL DECISION MAKER.
        If rules fail or query is unusual → LLM decides mode.
        """
        system = {
            "role": "system",
            "content": """
Classify this query into exactly one mode:

- "web" → real-time facts, entities, news, people, companies, trending topics
- "rag" → definitions, conceptual explanations, structured factual info
- "llm" → normal chat, creative tasks, responses without external info
- "deep_research" → multi-step analysis, long reports, deep comparisons

Return ONLY one word: web, rag, llm, or deep_research.
"""
        }

        user = {"role": "user", "content": q}

        resp = self.llm.invoke([system, user]).content.strip().lower()
        if resp in ["web", "rag", "llm", "deep_research"]:
            return resp
        return "llm"

    # ---------------- FINAL ROUTER ----------------
    def route(self, q: str) -> str:
        q = q.strip()

        # LAYER 1 — FAST RULES
        if self.is_greeting(q): return "llm"
        if self.is_image_query(q): return "image"
        if self.is_realtime(q): return "web"
        if self.is_world_fact(q): return "web"
        if self.is_ai_model(q): return "web"
        
        # Short entity queries (1-2 words) → web
        if len(q.split()) <= 2 and self.is_entity(q): return "web"

        if self.is_deep(q): return "deep_research"
        if self.is_definition(q): return "rag"

        # LAYER 2 — LLM SEMANTIC CLASSIFICATION
        return self.llm_decide(q)
