"""Load Groq configuration and create a client for the chat API."""

import os

from dotenv import load_dotenv
from groq import Groq


load_dotenv()

MODEL_NAME = "openai/gpt-oss-120b"


def create_groq_client():
    """Create a Groq client using GROQ_API_KEY from the local environment."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add your key to the local .env file."
        )

    return Groq(api_key=api_key, max_retries=3, timeout=45.0)
