import os
import time
import openai
from chromadb import HttpClient
from chromadb.config import Settings
from chromadb.api.models.Collection import Collection
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import MarkdownHeaderTextSplitter

from embeddings import get_embeddings

# Load environment variables
CHROMA_HOST = str(os.getenv("CHROMADB_HOST")) # This has to be the name of the service in the docker-compose file
CHROMA_PORT = int(os.getenv("CHROMADB_PORT"))
COLLECTION_NAME = str(os.getenv("COLLECTION_NAME"))
DATASET_ROOT_PATH = os.getenv("DATASET_DIRECTORY")
DATASET_MD_PATH = os.getenv("DATASET_MD_DIRECTORY")

os.makedirs(DATASET_ROOT_PATH, exist_ok=True)
os.makedirs(DATASET_MD_PATH, exist_ok=True)

# Embedding provider is selected via EMBEDDING_PROVIDER (default: chroma_default).
# We compute embeddings explicitly in Python so the reader (api/) and writer
# (this script) share one factory and the env var controls both sides.
embedder = get_embeddings()

# Define ChromaDB collection. We do NOT attach an embedding_function to the
# collection — embeddings are always supplied explicitly to upsert(), making
# `embedder` the single source of truth for vector generation.
while True:
    try:
        # Initialize the ChromaDB client
        chroma_client = HttpClient(host=CHROMA_HOST, port=CHROMA_PORT, settings=Settings(allow_reset=True, anonymized_telemetry=False))
        collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
        break
    except Exception as e:
        print("Error reading collection from ChromaDB: ", e)
        print("Retrying in 5 seconds...")
        time.sleep(5)

def get_existing_sources(collection: Collection):
    """
    Retrieve a set of existing source file names from a given collection.
    This function fetches all documents from the provided collection and extracts
    the source file names from their metadata. It returns a set of unique source
    file names.
    Args:
        collection (Collection): The collection from which to retrieve documents.
    Returns:
        set: A set of unique source file names extracted from the collection's metadata.
    """
    
    existing_sources = set()
    
    # collection.get() can be paginated in some Chroma versions, or you can use 'where'/'limit' arguments
    # For a small dataset, you can fetch them all at once:
    result = collection.get(include=["metadatas"])  # get all documents

    # 'metadatas' is a list of metadata dicts, one per document
    for meta in result["metadatas"]:
        # Each meta is like {"source": "file_name.pdf"}
        source_file = meta.get("source")
        if source_file:
            existing_sources.add(source_file)
    
    return existing_sources

# Function to populate ChromaDB with MDs
def populate_chroma():
    """
    Populates the ChromaDB with documents extracted from markdown files.
    This function performs the following steps:
    1. Retrieves the list of already-indexed file sources from the ChromaDB collection.
    2. Reads markdown files from the specified directory.
    3. Filters out files that have already been indexed.
    4. Extracts text content from the new markdown files.
    5. Splits the text content based on specified markdown headers.
    6. Further splits the header chunks into smaller text chunks.
    7. Creates lists to store the chunked documents, metadata, and IDs.
    8. Adds the documents and embeddings to the ChromaDB collection in batches, with retry logic for rate limit errors.
    Environment Variables:
    - DATASET_MD_DIRECTORY: The directory where markdown files are stored.
    Prints progress and timing information at each major step.
    """
    
    print("Populating ChromaDB...")
    
    # Get all already-indexed file soruces
    existing_files = get_existing_sources(collection)
    
    # Directory where the MDs are stored
    mds_directory = os.getenv("DATASET_MD_DIRECTORY")
    md_files = os.listdir(mds_directory)
    
    # Filter out already-indexed files
    new_md_files = []
    for md_file in md_files:
        pdf_name = f"{md_file[:-3]}.pdf"

        # Only process if pdf_name is NOT in the already indexed sources
        if pdf_name not in existing_files:
            new_md_files.append(md_file)
    
    # If there are no new files, skip population
    if not new_md_files:
        print("No new files found. Skipping ChromaDB population.")
        return
    else:
        print(f"Found {len(new_md_files)} new files to index.")
    
    docs = []
    file_names = []

    s = time.time()
    # Process each MD
    for md_file in os.listdir(mds_directory):
        
        # Path to MD file
        md_path = os.path.join(mds_directory, md_file)
        file_names.append(f'{md_file[:-3]}.pdf')
        
        # Get content from MD file as a string
        with open(md_path, "r") as file:
            content = file.read()
            docs.append(content.lower())
    e = time.time()
    print(f"Text extraction complete in {round(e-s, 2)//60} minutes and {round(e-s, 2) % 60} seconds.")
    
    # Define the headers to split on and create the MarkdownHeaderTextSplitter object
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2")
    ]
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on, strip_headers=True)
    
    documents = []

    s = time.time()
    for i, doc in enumerate(docs):
        # Split the MD content from the headers. The split_text() method returns a list of Document objects.
        md_header_splits = md_splitter.split_text(text=doc)
        
        # Append source metadata and id info to each Document
        for j, docu in enumerate(md_header_splits):
            docu.metadata["source"] = f'{file_names[i]}'
            docu.id = f'id{j}'
        
        # Append the Document objects to the documents list.
        # We use the extend method to append the list of Document objects.
        # That way, documents is a list of Document objects, not a list of lists of Document objects.
        documents.extend(md_header_splits)
    e = time.time()
    print(f"Split by markdown headers complete in {round(e-s, 2)//60} minutes and {round(e-s, 2) % 60} seconds.")
    
    # Now, we are going to split the header chunks into smaller chunks of text.
    s = time.time()
    chunk_size = 1500
    chunk_overlap = 200
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    # Chunks
    chunks = text_splitter.split_documents(documents)
    e = time.time()
    print(f"Documents split into {len(chunks)} chunks in {round(e-s, 2)//60} minutes and {round(e-s, 2) % 60} seconds.")
    
    # Create lists to store the chunked documents, metadata and IDs
    chunk_docs = []
    chunk_metadata = []
    chunk_ids = []

    s = time.time()
    i = 0
    for chunk in chunks:
        chunk_docs.append(chunk.page_content)
        chunk_ids.append(str(i))
        chunk_metadata.append(chunk.metadata)
        i += 1
    e = time.time()
    print(f"Chunk documents created in {round(e-s, 2)//60} minutes and {round(e-s, 2) % 60} seconds.")
    
    # Add the documents and embeddings to the ChromaDB collection
    BATCH_SIZE = 200
    s = time.time()
    
    def upsert_with_retry(collection: Collection, chunk_ids: list, chunk_docs: list, chunk_metadata: list, start: int, end: int, max_retries=5):
        """Upsert one batch with explicit embeddings (computed via the shared
        factory) and exponential backoff on OpenAI rate-limit errors.

        Args:
            collection (Collection): The collection to upsert documents into.
            chunk_ids (list): List of document IDs.
            chunk_docs (list): List of document contents.
            chunk_metadata (list): List of metadata dictionaries corresponding to the documents.
            start (int): The starting index of the batch.
            end (int): The ending index of the batch.
            max_retries (int, optional): Maximum number of retry attempts in case of rate limit errors. Defaults to 5.
        Raises:
            openai.error.RateLimitError: If the rate limit is exceeded and retries are exhausted.
            Exception: For any other errors encountered during the upsert process.
        """

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
                print(f"Batch {start//BATCH_SIZE + 1} added to ChromaDB.")
                break
            except openai.error.RateLimitError as e:
                retries += 1
                wait_time = 2 ** retries  # Exponential backoff
                print(f"Rate limit error: {e}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            except Exception as e:
                print(f"Error upserting batch: {e}")
                break

    for i in range(0, len(chunks), BATCH_SIZE):
        upsert_with_retry(collection, chunk_ids, chunk_docs, chunk_metadata, i, i + BATCH_SIZE)
        
    e = time.time()
    print(f"Vectors stored in {round(e-s, 2)//60} minutes and {round(e-s, 2) % 60} seconds.")
    
if __name__ == "__main__":
    populate = bool(int(os.getenv("POPULATE_CHROMA")))
    if populate:
        populate_chroma()
    else:
        print("Skipping ChromaDB population, if you want to populate ChromaDB, set the POPULATE_CHROMA environment variable to 1.")
