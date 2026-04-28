import os
import requests
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def get_response(query: str) -> Dict[str, Any]:
    """
    Llama a la API de Chroma (p.ej. http://localhost:8002/chatbot)
    y retorna un diccionario con 'llm_response' y 'retrieved_docs'.
    """
    try:
        api_url = "http://localhost:8002/chatbot"
        payload = {"question": query}
        resp = requests.post(api_url, json=payload, timeout=10)

        resp.raise_for_status()
        data = resp.json()

        return data
    
    except Exception as e:
        logger.error(f"Error llamando a la API de Chroma: {e}", exc_info=True)
        return {
            "llm_response": f"Ocurrió un error consultando a Chroma: {str(e)}",
            "retrieved_docs": []
        }