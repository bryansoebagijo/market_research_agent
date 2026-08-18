"""
The reasoning model (Gemini) as a LangChain chat model.

Kept in one place so every node shares the same configured model, and so swapping
providers later is a one-file change. LangGraph nodes call this to think.
"""

from functools import lru_cache

from langchain_google_genai import ChatGoogleGenerativeAI

from config import settings


@lru_cache(maxsize=1)
def get_llm(temperature: float = 0.2) -> ChatGoogleGenerativeAI:
    """Return a shared Gemini chat model. Low temperature = steadier reasoning."""
    if not settings.gemini_api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to .env "
            "(free key at https://aistudio.google.com/apikey)."
        )
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.gemini_api_key,
        temperature=temperature,
    )
