import os
import requests
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def get_response(query: str) -> Dict[str, Any]:
    """
    Call the Chroma API (p.ej. http://localhost:8002/chatbot)
    and return a dictionary with 'llm_response' and 'retrieved_docs'.
    """
    try:
        api_url = "http://localhost:8002/chatbot"
        payload = {"question": query}
        resp = requests.post(api_url, json=payload, timeout=10)

        resp.raise_for_status()
        data = resp.json()

        return data
    
    except Exception as e:
        logger.error(f"Error calling Chroma API: {e}", exc_info=True)
        return {
            "llm_response": f"An error ocurred consulting Chroma: {str(e)}",
            "retrieved_docs": []
        }