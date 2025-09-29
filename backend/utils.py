import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

def get_api_key():
    api_key = os.getenv("OPENAI_API_KEY")
    return OpenAI(api_key=api_key)