import sys
from pathlib import Path
import hashlib
import uuid

import chromadb
from sentence_transformers import SentenceTransformer

from rag.ingestion import (
    create_chunks,
    create_chunks_from_upload
)


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# EMBEDDING MODEL
# ============================================================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ============================================================
# CHROMADB
# ============================================================

chroma_client = chromadb.PersistentClient(
    path=str(PROJECT_ROOT / "vectorstore")
)


# ============================================================
# DOCUMENT COLLECTION
# ============================================================

def get_document_collection(document_name):

    document_hash = hashlib.md5(
        document_name.encode("utf-8")
    ).hexdigest()[:12]

    collection_name = f"doc_{document_hash}"

    return chroma_client.get_or_create_collection(
        name=collection_name
    )


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

def create_embeddings(chunks):

    if not chunks:
        return []

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = embedding_model.encode(
        texts,
        show_progress_bar=True
    )

    return embeddings.tolist()


# ============================================================
# STORE CHUNKS
# ============================================================

def store_chunks(chunks, collection):

    if not chunks:
        return 0

    embeddings = create_embeddings(
        chunks
    )

    ids = [
        f"chunk_{uuid.uuid4().hex}"
        for _ in chunks
    ]

    documents = [
        chunk["text"]
        for chunk in chunks
    ]

    metadatas = [
        {
            "source": chunk["source"],
            "page": chunk["page"]
        }
        for chunk in chunks
    ]

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    return len(chunks)


# ============================================================
# STORE UPLOADED PDF
# ============================================================

def store_uploaded_document(uploaded_file):

    print(
        f"\nProcessing: {uploaded_file.name}"
    )

    chunks = create_chunks_from_upload(
        uploaded_file
    )

    if not chunks:
        return 0

    print(
        f"Found {len(chunks)} chunks."
    )

    collection = get_document_collection(
        uploaded_file.name
    )

    count = store_chunks(
        chunks,
        collection
    )

    print(
        f"Stored {count} chunks."
    )

    return count


# ============================================================
# ORIGINAL DOCUMENT SUPPORT
# ============================================================

def store_documents():

    print(
        "Loading documents..."
    )

    chunks = create_chunks()

    if not chunks:

        print(
            "No documents found in data/documents/"
        )

        return

    print(
        f"Found {len(chunks)} chunks."
    )

    collection = chroma_client.get_or_create_collection(
        name="documents"
    )

    count = store_chunks(
        chunks,
        collection
    )

    print(
        "\n================================"
    )

    print(
        "Documents stored successfully!"
    )

    print(
        f"New chunks stored: {count}"
    )

    print(
        f"Total chunks: {collection.count()}"
    )

    print(
        "================================"
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    store_documents()