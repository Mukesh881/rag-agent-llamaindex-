"""
Configuration utilities: environment setup and logging.
"""

import os
import logging
from dotenv import load_dotenv


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with standard format."""
    if logging.getLogger().handlers:
        for h in list(logging.getLogger().handlers):
            logging.getLogger().removeHandler(h)

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def load_env(env_path: str | None = None) -> dict[str, str]:
    """Load API keys and environment variables from .env."""
    load_dotenv(dotenv_path=env_path)
    env = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
        "HUGGINGFACE_API_KEY": os.getenv("HUGGINGFACE_API_KEY", ""),
        "CHROMA_DB_PATH": os.getenv("CHROMA_DB_PATH", "./chroma_db"),
    }

    for key, value in env.items():
        if not value:
            logging.warning(f"{key} is not set in environment variables.")

    return env
