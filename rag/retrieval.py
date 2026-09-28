import sys
from pathlib import Path

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
# RETRIEVE DOCUMENTS
# ============================================================

def retrieve_documents(
    query,
    top_k=5,
    document_name=None
):

    collection = chroma_client.get_or_create_collection(
        name="documents"
    )

    if collection.count() == 0:
        return []

    # --------------------------------------------------------
    # Query embedding
    # --------------------------------------------------------

    query_embedding = embedding_model.encode(
        query
    ).tolist()

    # --------------------------------------------------------
    # Build metadata filter
    # --------------------------------------------------------

    query_kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": min(top_k, collection.count())
    }

    # If a document is selected, ask ChromaDB directly
    # for chunks belonging to that document.
    if document_name:
        query_kwargs["where"] = {
            "source": document_name
        }

    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    results = collection.query(
        **query_kwargs
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