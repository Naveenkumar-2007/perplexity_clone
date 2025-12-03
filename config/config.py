import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

    LLM_MODEL = "groq:openai/gpt-oss-120b"

    CHUNK_SIZE = 400
    CHUNK_OVERLAP = 80

    @classmethod
    def get_llm(cls):
        """Return chat LLM instance with tool calling disabled."""
        if not cls.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY missing in .env")
        os.environ["GROQ_API_KEY"] = cls.GROQ_API_KEY
        llm = init_chat_model(cls.LLM_MODEL)
        # Disable tool calling by binding empty tools list
        try:
            return llm.bind(tools=[])
        except:
            # Fallback if bind doesn't work
            return llm


