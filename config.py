import os
from dotenv import load_dotenv

load_dotenv()

GEMINIAI_API_KEY = os.getenv("GEMINIAI_API_KEY")

if not GEMINIAI_API_KEY:
    raise ValueError("GEMINIAI_API_KEY is not set")