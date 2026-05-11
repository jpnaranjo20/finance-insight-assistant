import asyncio
import hashlib
import os

from elasticsearch import AsyncElasticsearch
from langchain_core.documents import Document

_ES_HOST = os.getenv("ELASTICSEARCH_HOST", "elasticsearch")
_ES_PORT = int(os.getenv("ELASTICSEARCH_PORT", "9200"))
_ES_INDEX = "finance_chunks"
_RRF_K = 60


def _rrf_score(rank: int) -> float:
    return 1.0 / (_RRF_K + rank + 1)


def _chunk_id(doc: Document) -> str:
    return hashlib.sha256(doc.page_content.encode()).hexdigest()[:16]


async def hybrid_retrieve(vector_store, query: str, k: int = 10) -> list[Document]:
    """Query ChromaDB (dense) and Elasticsearch (BM25) concurrently, fuse with RRF."""
    dense_docs, sparse_docs = await asyncio.gather(
        _dense_search(vector_store, query, k),
        _sparse_search(query, k),
    )

    scores: dict[str, float] = {}
    id_to_doc: dict[str, Document] = {}

    for rank, doc in enumerate(dense_docs):
        cid = _chunk_id(doc)
        scores[cid] = scores.get(cid, 0.0) + _rrf_score(rank)
        id_to_doc[cid] = doc

    for rank, doc in enumerate(sparse_docs):
        cid = _chunk_id(doc)
        scores[cid] = scores.get(cid, 0.0) + _rrf_score(rank)
        id_to_doc.setdefault(cid, doc)

    ranked = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [id_to_doc[cid] for cid in ranked[:k]]


async def _dense_search(vector_store, query: str, k: int) -> list[Document]:
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": k}
    )
    return await retriever.ainvoke(query)


async def _sparse_search(query: str, k: int) -> list[Document]:
    es = AsyncElasticsearch(f"http://{_ES_HOST}:{_ES_PORT}")
    try:
        resp = await es.search(
            index=_ES_INDEX,
            body={"query": {"match": {"text": query}}, "size": k},
        )
        return [
            Document(
                page_content=hit["_source"]["text"],
                metadata={
                    "source":   hit["_source"].get("source", ""),
                    "Header 1": hit["_source"].get("header1", ""),
                    "Header 2": hit["_source"].get("header2", ""),
                },
            )
            for hit in resp["hits"]["hits"]
        ]
    finally:
        await es.close()
