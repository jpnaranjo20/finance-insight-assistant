"""Embedding-provider factory (writer side).

Mirrors api/app/embeddings.py so both ends use the same selection. See that
file for full provider docs and the dimensionality caveat when switching.
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
