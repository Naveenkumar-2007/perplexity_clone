import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    
    # Use lighter model for free tier
    LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")

    CHUNK_SIZE = 400
    CHUNK_OVERLAP = 80
    
    # Disable heavy features on free tier (512MB RAM limit)
    LITE_MODE = os.getenv("LITE_MODE", "true").lower() == "true"

    @classmethod
    def get_llm(cls):
        """Return chat LLM instance with tool calling disabled."""
        if not cls.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY missing in .env")
        llm = ChatGroq(
            groq_api_key=cls.GROQ_API_KEY,
            model_name=cls.LLM_MODEL,
            temperature=0.7
        )
        # Disable tool calling by binding empty tools list
        try:
            return llm.bind(tools=[])
        except:
            # Fallback if bind doesn't work
            return llm


