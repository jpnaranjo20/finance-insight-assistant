import os
import time
import openai
from chromadb import HttpClient
from chromadb.config import Settings
from chromadb.api.models.Collection import Collection
from elasticsearch import Elasticsearch
from elasticsearch import helpers as es_helpers
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import MarkdownHeaderTextSplitter

from embeddings import get_embeddings

# Load environment variables
CHROMA_HOST = str(os.getenv("CHROMADB_HOST"))
CHROMA_PORT = int(os.getenv("CHROMADB_PORT"))
COLLECTION_NAME = str(os.getenv("COLLECTION_NAME"))
DATASET_ROOT_PATH = os.getenv("DATASET_DIRECTORY")
DATASET_MD_PATH = os.getenv("DATASET_MD_DIRECTORY")

ES_HOST = str(os.getenv("ELASTICSEARCH_HOST", "elasticsearch"))
ES_PORT = int(os.getenv("ELASTICSEARCH_PORT", "9200"))
ES_INDEX = "finance_chunks"

POPULATE_TARGET = str(os.getenv("POPULATE_TARGET", "none")).lower()

os.makedirs(DATASET_ROOT_PATH, exist_ok=True)
os.makedirs(DATASET_MD_PATH, exist_ok=True)

# Embedding provider is selected via EMBEDDING_PROVIDER (default: chroma_default).
# We compute embeddings explicitly in Python so the reader (api/) and writer
# (this script) share one factory and the env var controls both sides.
embedder = get_embeddings()

BATCH_SIZE = 200


#  ChromaDB helpers 

def _chroma_client_and_collection():
    while True:
        try:
            client = HttpClient(
                host=CHROMA_HOST,
                port=CHROMA_PORT,
                settings=Settings(allow_reset=True, anonymized_telemetry=False),
            )
            # We do NOT attach an embedding_function to the collection —
            # embeddings are always supplied explicitly to upsert().
            col = client.get_or_create_collection(name=COLLECTION_NAME)
            return client, col
        except Exception as e:
            print(f"Error connecting to ChromaDB: {e}. Retrying in 5s...")
            time.sleep(5)


def get_existing_sources_from_chroma(collection: Collection, page_size: int = 1000):
    """Paginate through the collection to collect already-indexed source filenames.

    Paginates so SQLite's variable limit is never hit on large collections.
    """

    existing_sources = set()
    offset = 0

    # Walk the collection page by page. Each call to collection.get() with
    # limit + offset issues a single SQL query bounded by `page_size` rows,
    # which keeps us safely under SQLite's bind-parameter ceiling
    # (SQLITE_MAX_VARIABLE_NUMBER, default 999 on old builds / 32766 on newer).
    while True:
        # Ask Chroma for one page of records (only metadatas — we don't need
        # the raw documents or embeddings here, which keeps the response small).
        result = collection.get(
            include=["metadatas"],
            limit=page_size,
            offset=offset,
        )
        metadatas = result.get("metadatas") or []

        # Empty page → we've walked off the end of the collection. Stop.
        if not metadatas:
            break

        # Each metadata dict looks like {"source": "<original_pdf_name>.pdf",
        # ...}. Collect the unique sources we've seen so the caller can
        # filter out files that are already indexed.
        for meta in metadatas:
            source_file = meta.get("source")
            if source_file:
                existing_sources.add(source_file)

        # Short page → this was the last one (Chroma returned fewer rows than
        # we asked for). Bail out without making one more empty round-trip.
        if len(metadatas) < page_size:
            break

        # Otherwise advance the offset and fetch the next page.
        offset += page_size

    return existing_sources


def upsert_with_retry(
    collection: Collection,
    chunk_ids: list,
    chunk_docs: list,
    chunk_metadata: list,
    start: int,
    end: int,
    max_retries: int = 5,
):
    retries = 0
    while retries < max_retries:
        try:
            batch_docs = chunk_docs[start:end]
            batch_embeddings = embedder.embed_documents(batch_docs)
            collection.upsert(
                ids=chunk_ids[start:end],
                documents=batch_docs,
                metadatas=chunk_metadata[start:end],
                embeddings=batch_embeddings,
            )
            print(f"Batch {start // BATCH_SIZE + 1} added to ChromaDB.")
            break
        except openai.RateLimitError as e:
            retries += 1
            wait_time = 2 ** retries
            print(f"Rate limit error: {e}. Retrying in {wait_time}s...")
            time.sleep(wait_time)
        except Exception as e:
            print(f"Error upserting batch: {e}")
            break


#  Elasticsearch helpers 

def _es_client():
    while True:
        try:
            es = Elasticsearch(f"http://{ES_HOST}:{ES_PORT}")
            if es.ping():
                return es
            raise ConnectionError("Elasticsearch ping failed.")
        except Exception as e:
            print(f"Error connecting to Elasticsearch: {e}. Retrying in 5s...")
            time.sleep(5)


def _ensure_es_index(es: Elasticsearch):
    if not es.indices.exists(index=ES_INDEX):
        es.indices.create(
            index=ES_INDEX,
            mappings={
                "properties": {
                    "text":    {"type": "text",    "analyzer": "english"},
                    "source":  {"type": "keyword"},
                    "header1": {"type": "text"},
                    "header2": {"type": "text"},
                }
            },
        )
        print(f"Created Elasticsearch index '{ES_INDEX}'.")
    else:
        print(f"Elasticsearch index '{ES_INDEX}' already exists.")


def get_existing_sources_from_es(es: Elasticsearch) -> set:
    """Return source filenames already indexed via a terms aggregation."""
    try:
        resp = es.search(
            index=ES_INDEX,
            body={
                "size": 0,
                "aggs": {"sources": {"terms": {"field": "source", "size": 10000}}},
            },
        )
        return {b["key"] for b in resp["aggregations"]["sources"]["buckets"]}
    except Exception:
        return set()


def _index_batch_to_es(
    es: Elasticsearch,
    chunk_ids: list,
    chunk_docs: list,
    chunk_metadata: list,
    start: int,
    end: int,
):
    actions = [
        {
            "_index": ES_INDEX,
            "_id": chunk_ids[i],
            "_source": {
                "text":    chunk_docs[i],
                "source":  chunk_metadata[i].get("source", ""),
                "header1": chunk_metadata[i].get("Header 1", ""),
                "header2": chunk_metadata[i].get("Header 2", ""),
            },
        }
        for i in range(start, min(end, len(chunk_ids)))
    ]
    es_helpers.bulk(es, actions)
    print(f"Batch {start // BATCH_SIZE + 1} added to Elasticsearch.")


#  Shared chunking logic 

def _build_chunks(mds_directory: str, only_files: list | None = None):
    """Read MD files and return (chunk_docs, chunk_metadata, chunk_ids).

    If only_files is given, only those filenames are processed (incremental
    ingestion). Otherwise all files in the directory are processed.
    """
    docs = []
    file_names = []

    candidates = only_files if only_files is not None else os.listdir(mds_directory)
    s = time.time()
    for md_file in candidates:
        if not md_file.endswith(".md"):
            continue
        md_path = os.path.join(mds_directory, md_file)
        file_names.append(f"{md_file[:-3]}.pdf")
        with open(md_path, "r") as f:
            docs.append(f.read().lower())
    e = time.time()
    print(f"Text extraction complete in {round(e - s, 2) // 60}m {round(e - s, 2) % 60}s.")

    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2")]
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on, strip_headers=True)

    documents = []

    s = time.time()
    for i, doc in enumerate(docs):
        # Split the MD content from the headers. The split_text() method returns a list of Document objects.
        md_header_splits = md_splitter.split_text(text=doc)
        
        # Append source metadata and id info to each Document
        for j, docu in enumerate(md_header_splits):
            docu.metadata["source"] = file_names[i]
            docu.id = f"id{j}"
        documents.extend(md_header_splits)
    e = time.time()
    print(f"Markdown header split complete in {round(e - s, 2) // 60}m {round(e - s, 2) % 60}s.")

    s = time.time()
    chunks = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200).split_documents(documents)
    e = time.time()
    print(f"Documents split into {len(chunks)} chunks in {round(e - s, 2) // 60}m {round(e - s, 2) % 60}s.")

    chunk_docs, chunk_metadata, chunk_ids = [], [], []
    for i, chunk in enumerate(chunks):
        chunk_docs.append(chunk.page_content)
        chunk_ids.append(str(i))
        chunk_metadata.append(chunk.metadata)

    return chunk_docs, chunk_metadata, chunk_ids


def _filter_new_files(md_files: list, existing_sources: set) -> list:
    return [f for f in md_files if f"{f[:-3]}.pdf" not in existing_sources]


#  Population entry points 

def populate_chroma_store():
    print("Populating ChromaDB...")
    _, collection = _chroma_client_and_collection()
    mds_directory = os.getenv("DATASET_MD_DIRECTORY")
    md_files = os.listdir(mds_directory)
    existing = get_existing_sources_from_chroma(collection)
    new_files = _filter_new_files(md_files, existing)

    if not new_files:
        print("No new files found. Skipping ChromaDB population.")
        return
    print(f"Found {len(new_files)} new files to index into ChromaDB.")

    chunk_docs, chunk_metadata, chunk_ids = _build_chunks(mds_directory, only_files=new_files)

    s = time.time()
    for i in range(0, len(chunk_ids), BATCH_SIZE):
        upsert_with_retry(collection, chunk_ids, chunk_docs, chunk_metadata, i, i + BATCH_SIZE)
    e = time.time()
    print(f"ChromaDB vectors stored in {round(e - s, 2) // 60}m {round(e - s, 2) % 60}s.")


def populate_es_store():
    print("Populating Elasticsearch...")
    es = _es_client()
    _ensure_es_index(es)
    mds_directory = os.getenv("DATASET_MD_DIRECTORY")
    md_files = os.listdir(mds_directory)
    existing = get_existing_sources_from_es(es)
    new_files = _filter_new_files(md_files, existing)

    if not new_files:
        print("No new files found. Skipping Elasticsearch population.")
        return
    print(f"Found {len(new_files)} new files to index into Elasticsearch.")

    chunk_docs, chunk_metadata, chunk_ids = _build_chunks(mds_directory, only_files=new_files)

    s = time.time()
    for i in range(0, len(chunk_ids), BATCH_SIZE):
        _index_batch_to_es(es, chunk_ids, chunk_docs, chunk_metadata, i, i + BATCH_SIZE)
    e = time.time()
    print(f"{len(chunk_ids)} chunks indexed into Elasticsearch in {round(e - s, 2) // 60}m {round(e - s, 2) % 60}s.")


if __name__ == "__main__":
    if POPULATE_TARGET == "none":
        print("POPULATE_TARGET=none — skipping ingestion. Set to 'chroma', 'elasticsearch', or 'both' to ingest.")
    elif POPULATE_TARGET == "chroma":
        populate_chroma_store()
    elif POPULATE_TARGET == "elasticsearch":
        populate_es_store()
    elif POPULATE_TARGET == "both":
        populate_chroma_store()
        populate_es_store()
    else:
        print(f"Unknown POPULATE_TARGET='{POPULATE_TARGET}'. Valid values: none, chroma, elasticsearch, both.")
