"""Embedding-provider factory.

Reader (`api`) and writer (`populate_chroma`) both call `get_embeddings()` so a
single env var keeps them in lockstep.

To switch providers, set EMBEDDING_PROVIDER in .env (default: chroma_default).
Supported values:
  - chroma_default : Chroma's built-in DefaultEmbeddingFunction
                     (ONNX-quantized all-MiniLM-L6-v2, 384 dims). Free, local.
  - openai         : OpenAIEmbeddings, model from EMBEDDING_MODEL
                     (default: text-embedding-3-small, 1536 dims). Requires
                     OPENAI_API_KEY and incurs API cost on every ingest.

IMPORTANT: switching provider also switches vector dimensionality. You must
use a fresh ChromaDB collection (or wipe the volume) when changing providers
— mixing 384-dim and 1536-dim vectors in the same collection will error or
silently degrade retrieval.
"""

import os

from langchain_core.embeddings import Embeddings


class ChromaDefaultEmbeddings(Embeddings):
    """LangChain-compatible adapter for Chroma's DefaultEmbeddingFunction
    (ONNX-quantized all-MiniLM-L6-v2)."""

    def __init__(self):
        from chromadb.utils import embedding_functions
        self._fn = embedding_functions.DefaultEmbeddingFunction()

    def embed_documents(self, texts):
        return self._fn(texts)

    def embed_query(self, text):
        return self._fn([text])[0]


def get_embeddings() -> Embeddings:
    provider = os.getenv("EMBEDDING_PROVIDER", "chroma_default").lower()

    if provider == "chroma_default":
        return ChromaDefaultEmbeddings()

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        return OpenAIEmbeddings(model=model)

    raise ValueError(
        f"Unknown EMBEDDING_PROVIDER={provider!r}. "
        f"Expected one of: chroma_default, openai."
    )
