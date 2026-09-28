import os
import re
import uuid

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pypdf import PdfReader
from upstash_vector import Index

load_dotenv()

# --------------------------------------------------
# Environment
# --------------------------------------------------

GEMINIAI_API_KEY = os.getenv("GEMINIAI_API_KEY")
UPSTASH_VECTOR_REST_URL = os.getenv("UPSTASH_VECTOR_REST_URL")
UPSTASH_VECTOR_REST_TOKEN = os.getenv("UPSTASH_VECTOR_REST_TOKEN")

if not GEMINIAI_API_KEY:
    raise ValueError("GEMINIAI_API_KEY is not set")

if not UPSTASH_VECTOR_REST_URL:
    raise ValueError("UPSTASH_VECTOR_REST_URL is not set")

if not UPSTASH_VECTOR_REST_TOKEN:
    raise ValueError("UPSTASH_VECTOR_REST_TOKEN is not set")


# --------------------------------------------------
# Clients
# --------------------------------------------------

gemini = genai.Client(
    api_key=GEMINIAI_API_KEY
)

vector_index = Index(
    url=UPSTASH_VECTOR_REST_URL,
    token=UPSTASH_VECTOR_REST_TOKEN
)


# --------------------------------------------------
# FastAPI
# --------------------------------------------------

app = FastAPI(
    title="Agentic RAG + MCP API",
    description="Intelligent Research Assistant",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Request model
# --------------------------------------------------

class AskRequest(BaseModel):
    question: str
    document_name: str = ""


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def clean_text(text):
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def create_chunks(text, chunk_size=1200, overlap=200):
    text = clean_text(text)

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk.strip())

        if end >= len(text):
            break

        start = end - overlap

    return chunks


def get_embedding(text, task_type):
    response = gemini.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=768
        )
    )

    return response.embeddings[0].values


def get_embeddings(texts, task_type):
    response = gemini.models.embed_content(
        model="gemini-embedding-001",
        contents=texts,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=768
        )
    )

    return [embedding.values for embedding in response.embeddings]


# --------------------------------------------------
# Root
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "status": "online",
        "message": "Agentic RAG + MCP API is running"
    }


# --------------------------------------------------
# Health
# --------------------------------------------------

@app.get("/api/health")
def health():
    return {
        "status": "healthy"
    }


# --------------------------------------------------
# Documents
# --------------------------------------------------

@app.get("/api/documents")
def documents():
    try:
        # Query using a zero vector to retrieve document markers.
        results = vector_index.query(
            vector=[0.0] * 768,
            top_k=100,
            include_metadata=True
        )

        names = set()

        for result in results:
            metadata = result.metadata or {}

            if metadata.get("type") == "document":
                name = metadata.get("document")

                if name:
                    names.add(name)

        return {
            "documents": sorted(names)
        }

    except Exception as e:
        print("DOCUMENT ERROR:", str(e))
        return {
            "documents": []
        }


# --------------------------------------------------
# Upload PDF
# --------------------------------------------------

@app.post("/api/upload")
async def upload_pdf(
    file: UploadFile = File(...)
):

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed."
        )

    try:

        file_bytes = await file.read()

        if not file_bytes:
            raise HTTPException(
                status_code=400,
                detail="The uploaded PDF is empty."
            )

        # Read PDF directly from memory.
        import io

        reader = PdfReader(
            io.BytesIO(file_bytes)
        )

        all_chunks = []

        for page_number, page in enumerate(reader.pages, start=1):

            text = page.extract_text() or ""

            page_chunks = create_chunks(text)

            for chunk in page_chunks:
                all_chunks.append({
                    "text": chunk,
                    "page": page_number,
                    "document": file.filename
                })

        if not all_chunks:
            raise HTTPException(
                status_code=400,
                detail="Could not extract readable text from this PDF."
            )

        # Generate Gemini embeddings.
        texts = [
            item["text"]
            for item in all_chunks
        ]

        embeddings = get_embeddings(
            texts,
            "RETRIEVAL_DOCUMENT"
        )

        vectors = []

        for item, embedding in zip(
            all_chunks,
            embeddings
        ):

            vectors.append(
                {
                    "id": str(uuid.uuid4()),
                    "vector": embedding,
                    "metadata": {
                        "type": "chunk",
                        "document": item["document"],
                        "page": item["page"],
                        "text": item["text"]
                    }
                }
            )

        # Store vectors in Upstash.
        vector_index.upsert(
            vectors=vectors
        )

        # Store a document marker so the dropdown can list it.
        document_marker = {
            "id": "doc-" + str(uuid.uuid5(
                uuid.NAMESPACE_URL,
                file.filename
            )),
            "vector": [0.0] * 768,
            "metadata": {
                "type": "document",
                "document": file.filename
            }
        }

        vector_index.upsert(
            vectors=[document_marker]
        )

        return {
            "success": True,
            "message": "PDF uploaded and indexed successfully.",
            "document": file.filename,
            "chunks": len(all_chunks)
        }

    except HTTPException:
        raise

    except Exception as e:

        print(
            "UPLOAD ERROR:",
            str(e)
        )

        raise HTTPException(
            status_code=500,
            detail=f"Failed to process PDF: {str(e)}"
        )


# --------------------------------------------------
# Ask Question
# --------------------------------------------------

@app.post("/api/ask")
def ask(request: AskRequest):

    try:

        query_embedding = get_embedding(
            request.question,
            "RETRIEVAL_QUERY"
        )

        results = vector_index.query(
            vector=query_embedding,
            top_k=8,
            include_metadata=True
        )

        retrieved = []

        for result in results:

            metadata = result.metadata or {}

            if metadata.get("type") != "chunk":
                continue

            if (
                request.document_name
                and metadata.get("document")
                != request.document_name
            ):
                continue

            retrieved.append({
                "text": metadata.get("text", ""),
                "document": metadata.get(
                    "document",
                    "Unknown"
                ),
                "page": metadata.get(
                    "page",
                    "Unknown"
                ),
                "score": result.score
            })

        if not retrieved:

            return {
                "question": request.question,
                "document": request.document_name,
                "answer": (
                    "No relevant information was found "
                    "in the selected document."
                ),
                "sources": []
            }

        # Build context.
        context_parts = []

        for i, item in enumerate(retrieved):

            context_parts.append(
                f"""
SOURCE {i + 1}

Document:
{item['document']}

Page:
{item['page']}

Content:
{item['text']}
"""
            )

        context = "\n".join(context_parts)

        prompt = f"""
You are an intelligent research assistant.

Answer the user's question using ONLY the retrieved
document context below.

If the answer cannot be found in the context,
say that the information is not available in
the retrieved document context.

Do not invent information.

User Question:
{request.question}

Retrieved Document Context:
{context}

Give a clear and concise answer.
"""

        response = gemini.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt
        )

        sources = []

        for item in retrieved:

            sources.append({
                "document": item["document"],
                "page": item["page"]
            })

        return {
            "question": request.question,
            "document": request.document_name,
            "answer": response.text,
            "sources": sources
        }

    except Exception as e:

        print(
            "ASK ERROR:",
            str(e)
        )

        raise HTTPException(
            status_code=500,
            detail=f"Failed to answer question: {str(e)}"
        )