import asyncio
import hashlib
import os

from elasticsearch import AsyncElasticsearch
from langchain_core.documents import Document

_ES_HOST = os.getenv("ELASTICSEARCH_HOST", "elasticsearch")
_ES_PORT = int(os.getenv("ELASTICSEARCH_PORT", "9200"))
_ES_INDEX = "finance_chunks"

# Module-level singleton — reuses the connection pool across requests.
_es_client = AsyncElasticsearch(f"http://{_ES_HOST}:{_ES_PORT}")
_RRF_K = 60
_DENSE_WEIGHT  = float(os.getenv("DENSE_WEIGHT",  "0.5"))
_SPARSE_WEIGHT = float(os.getenv("SPARSE_WEIGHT", "0.5"))


def _rrf_score(rank: int) -> float:
    return 1.0 / (_RRF_K + rank + 1)


def _chunk_id(doc: Document) -> str:
    return hashlib.sha256(doc.page_content.encode()).hexdigest()[:16]


def _inject_provenance(
    doc: Document,
    source: str,
    dense_rank: int | None,
    bm25_rank: int | None,
    rrf_score: float,
) -> Document:
    doc.metadata.update({
        "_retrieval_source": source,
        "_dense_rank": dense_rank,
        "_bm25_rank": bm25_rank,
        "_rrf_score": rrf_score,
    })
    return doc


async def hybrid_retrieve(vector_store, query: str, k: int = 10) -> list[Document]:
    """Query ChromaDB (dense) and Elasticsearch (BM25) concurrently, fuse with RRF.

    Returns fused_docs. Each doc's metadata already contains
    _retrieval_source, _dense_rank, _bm25_rank, _rrf_score so consumers need not
    join separately.
    """
    dense_docs, sparse_docs = await asyncio.gather(
        _dense_search(vector_store, query, k),
        _sparse_search(query, k),
    )

    dense_ranks: dict[str, int] = {_chunk_id(d): rank for rank, d in enumerate(dense_docs)}
    sparse_ranks: dict[str, int] = {_chunk_id(d): rank for rank, d in enumerate(sparse_docs)}

    scores: dict[str, float] = {}
    id_to_doc: dict[str, Document] = {}

    for rank, doc in enumerate(dense_docs):
        cid = _chunk_id(doc)
        scores[cid] = scores.get(cid, 0.0) + _DENSE_WEIGHT * _rrf_score(rank)
        id_to_doc[cid] = doc

    for rank, doc in enumerate(sparse_docs):
        cid = _chunk_id(doc)
        scores[cid] = scores.get(cid, 0.0) + _SPARSE_WEIGHT * _rrf_score(rank)
        id_to_doc.setdefault(cid, doc)

    ranked = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    top = ranked[:k]

    docs: list[Document] = []
    for cid in top:
        in_dense = cid in dense_ranks
        in_sparse = cid in sparse_ranks
        source = "both" if (in_dense and in_sparse) else ("dense" if in_dense else "bm25")
        rrf = scores[cid]
        docs.append(_inject_provenance(
            id_to_doc[cid],
            source=source,
            dense_rank=dense_ranks.get(cid),
            bm25_rank=sparse_ranks.get(cid),
            rrf_score=rrf,
        ))

    return docs


async def dense_retrieve(vector_store, query: str, k: int = 10) -> list[Document]:
    """Dense-only retrieval. bm25_rank is None for all returned docs."""
    raw_docs = await _dense_search(vector_store, query, k)
    docs: list[Document] = []
    for rank, doc in enumerate(raw_docs):
        cid = _chunk_id(doc)
        rrf = _rrf_score(rank)
        docs.append(_inject_provenance(doc, source="dense", dense_rank=rank, bm25_rank=None, rrf_score=rrf))
    return docs


async def sparse_retrieve(query: str, k: int = 10) -> list[Document]:
    """BM25-only retrieval. dense_rank is None for all returned docs."""
    raw_docs = await _sparse_search(query, k)
    docs: list[Document] = []
    for rank, doc in enumerate(raw_docs):
        cid = _chunk_id(doc)
        rrf = _rrf_score(rank)
        docs.append(_inject_provenance(doc, source="bm25", dense_rank=None, bm25_rank=rank, rrf_score=rrf))
    return docs


async def _dense_search(vector_store, query: str, k: int) -> list[Document]:
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": k}
    )
    return await retriever.ainvoke(query)


async def _sparse_search(query: str, k: int) -> list[Document]:
    resp = await _es_client.search(
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
