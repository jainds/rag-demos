from dotenv import load_dotenv
import os
from pathlib import Path

# Always load .env.test if it exists, else .env
DOTENV_PATH = Path(__file__).parent.parent / ".env.test"
if not DOTENV_PATH.exists():
    DOTENV_PATH = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=DOTENV_PATH, override=True)

# ... existing code ... 