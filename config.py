# config.py
import os
from dotenv import load_dotenv

load_dotenv()

def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except Exception:
        return default

# outputs / chunking
OUT_DIR = os.getenv("OUT_DIR", "out")
MAX_TOKENS = env_int("MAX_TOKENS", 8191)
CONCURRENCY = env_int("CONCURRENCY", 5)



# OpenAI
OPENAI_API_KEY        = os.getenv("OPENAI_API_KEY", "")
OPENAI_CONCEPTS_MODEL = os.getenv("OPENAI_CONCEPTS_MODEL", "gpt-5-mini-2025-08-07")

# common convenience alias used by some scripts
LLM_MODEL = os.getenv("LLM_MODEL", OPENAI_CONCEPTS_MODEL)
