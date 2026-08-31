import sys
from pathlib import Path
import hashlib

import chromadb
from sentence_transformers import SentenceTransformer


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
# RETRIEVE DOCUMENTS
# ============================================================

def retrieve_documents(
    query,
    top_k=5,
    document_name=None
):

    # --------------------------------------------------------
    # Select collection
    # --------------------------------------------------------

    if document_name:

        collection = get_document_collection(
            document_name
        )

    else:

        collection = chroma_client.get_or_create_collection(
            name="documents"
        )


    # --------------------------------------------------------
    # Check collection
    # --------------------------------------------------------

    if collection.count() == 0:
        return []


    # --------------------------------------------------------
    # Create query embedding
    # --------------------------------------------------------

    query_embedding = embedding_model.encode(
        query
    ).tolist()


    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    n_results = min(
        top_k,
        collection.count()
    )

    results = collection.query(
        query_embeddings=[
            query_embedding
        ],
        n_results=n_results
    )


    retrieved_documents = []


    if not results.get("documents"):
        return retrieved_documents


    documents = results["documents"][0]

    metadatas = results["metadatas"][0]


    for i, document in enumerate(documents):

        metadata = metadatas[i] or {}

        retrieved_documents.append({

            "text": document,

            "source": metadata.get(
                "source",
                "Unknown"
            ),

            "page": metadata.get(
                "page",
                "Unknown"
            )

        })


    return retrieved_documents