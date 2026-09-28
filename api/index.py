import os
import re
import uuid
import io

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from google import genai
from google.genai import types

from pypdf import PdfReader
from upstash_vector import Index


# ==================================================
# Environment
# ==================================================

load_dotenv()

GEMINIAI_API_KEY = os.getenv("GEMINIAI_API_KEY")
UPSTASH_VECTOR_REST_URL = os.getenv("UPSTASH_VECTOR_REST_URL")
UPSTASH_VECTOR_REST_TOKEN = os.getenv("UPSTASH_VECTOR_REST_TOKEN")


if not GEMINIAI_API_KEY:
    raise ValueError("GEMINIAI_API_KEY is not set")

if not UPSTASH_VECTOR_REST_URL:
    raise ValueError("UPSTASH_VECTOR_REST_URL is not set")

if not UPSTASH_VECTOR_REST_TOKEN:
    raise ValueError("UPSTASH_VECTOR_REST_TOKEN is not set")


# ==================================================
# Clients
# ==================================================

gemini = genai.Client(
    api_key=GEMINIAI_API_KEY
)

vector_index = Index(
    url=UPSTASH_VECTOR_REST_URL,
    token=UPSTASH_VECTOR_REST_TOKEN
)


# ==================================================
# FastAPI
# ==================================================

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


# ==================================================
# Request Model
# ==================================================

class AskRequest(BaseModel):
    question: str
    document_name: str = ""


# ==================================================
# Text Cleaning
# ==================================================

def clean_text(text):

    text = text.replace("\x00", " ")

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ==================================================
# Text Chunking
# ==================================================

def create_chunks(
    text,
    chunk_size=1200,
    overlap=200
):

    text = clean_text(text)

    if not text:
        return []

    chunks = []

    start = 0

    while start < len(text):

        end = min(
            start + chunk_size,
            len(text)
        )

        chunk = text[start:end]

        if chunk.strip():
            chunks.append(
                chunk.strip()
            )

        if end >= len(text):
            break

        start = end - overlap

    return chunks


# ==================================================
# Gemini Embedding - Single Text
# ==================================================

def get_embedding(
    text,
    task_type
):

    response = gemini.models.embed_content(

        model="gemini-embedding-001",

        contents=text,

        config=types.EmbedContentConfig(

            task_type=task_type,

            output_dimensionality=768
        )
    )

    return response.embeddings[0].values


# ==================================================
# Gemini Embeddings - Multiple Texts
# ==================================================

def get_embeddings(
    texts,
    task_type
):

    response = gemini.models.embed_content(

        model="gemini-embedding-001",

        contents=texts,

        config=types.EmbedContentConfig(

            task_type=task_type,

            output_dimensionality=768
        )
    )

    return [
        embedding.values
        for embedding in response.embeddings
    ]


# ==================================================
# Root
# ==================================================

@app.get("/")
def root():

    return {
        "status": "online",
        "message": "Agentic RAG + MCP API is running"
    }


# ==================================================
# Health
# ==================================================

@app.get("/api/health")
def health():

    return {
        "status": "healthy"
    }


# ==================================================
# Documents
# ==================================================

@app.get("/api/documents")
def documents():

    try:

        names = set()

        cursor = "0"

        while True:

            result = vector_index.range(

                cursor=cursor,

                limit=1000,

                include_vectors=False,

                include_metadata=True,

                include_data=False
            )

            for vector_info in result.vectors:

                metadata = (
                    vector_info.metadata
                    or {}
                )

                if metadata.get("type") == "chunk":

                    document_name = metadata.get(
                        "document"
                    )

                    if document_name:
                        names.add(
                            document_name
                        )

            cursor = result.next_cursor

            if cursor == "":
                break

        return {
            "documents": sorted(names)
        }

    except Exception as e:

        print(
            "DOCUMENT ERROR:",
            str(e)
        )

        raise HTTPException(
            status_code=500,
            detail=f"Failed to load documents: {str(e)}"
        )


# ==================================================
# Upload PDF
# ==================================================

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

        # ------------------------------------------
        # Read PDF
        # ------------------------------------------

        file_bytes = await file.read()

        if not file_bytes:

            raise HTTPException(
                status_code=400,
                detail="The uploaded PDF is empty."
            )


        # ------------------------------------------
        # Extract PDF pages
        # ------------------------------------------

        reader = PdfReader(
            io.BytesIO(file_bytes)
        )

        all_chunks = []


        for page_number, page in enumerate(
            reader.pages,
            start=1
        ):

            page_text = page.extract_text()

            if not page_text:
                continue

            page_chunks = create_chunks(
                page_text
            )


            for chunk in page_chunks:

                all_chunks.append({

                    "text": chunk,

                    "page": page_number,

                    "document": file.filename
                })


        if not all_chunks:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Could not extract readable "
                    "text from this PDF."
                )
            )


        # ------------------------------------------
        # Generate embeddings
        # ------------------------------------------

        texts = [
            item["text"]
            for item in all_chunks
        ]


        embeddings = get_embeddings(

            texts,

            "RETRIEVAL_DOCUMENT"
        )


        # ------------------------------------------
        # Prepare vectors
        # ------------------------------------------

        vectors = []


        for item, embedding in zip(
            all_chunks,
            embeddings
        ):

            vectors.append({

                "id": str(
                    uuid.uuid4()
                ),

                "vector": embedding,

                "metadata": {

                    "type": "chunk",

                    "document": item[
                        "document"
                    ],

                    "page": item[
                        "page"
                    ],

                    "text": item[
                        "text"
                    ]
                }
            })


        # ------------------------------------------
        # Store in Upstash
        # ------------------------------------------

        vector_index.upsert(
            vectors=vectors
        )


        return {

            "success": True,

            "message": (
                "PDF uploaded and indexed successfully."
            ),

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

            detail=(
                f"Failed to process PDF: {str(e)}"
            )
        )


# ==================================================
# Ask Question
# ==================================================

@app.post("/api/ask")
def ask(
    request: AskRequest
):

    try:

        # ------------------------------------------
        # Create query embedding
        # ------------------------------------------

        query_embedding = get_embedding(

            request.question,

            "RETRIEVAL_QUERY"
        )


        # ------------------------------------------
        # Build filter
        # ------------------------------------------

        query_filter = None

        if request.document_name:

            safe_document = (
                request.document_name
                .replace("'", "\\'")
            )

            query_filter = (
                f"type = 'chunk' "
                f"AND document = '{safe_document}'"
            )

        else:

            query_filter = (
                "type = 'chunk'"
            )


        # ------------------------------------------
        # Vector search
        # ------------------------------------------

        results = vector_index.query(

            vector=query_embedding,

            top_k=8,

            include_metadata=True,

            include_vectors=False,

            filter=query_filter
        )


        retrieved = []


        for result in results:

            metadata = (
                result.metadata
                or {}
            )


            if metadata.get("type") != "chunk":
                continue


            retrieved.append({

                "text": metadata.get(
                    "text",
                    ""
                ),

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


        # ------------------------------------------
        # No results
        # ------------------------------------------

        if not retrieved:

            return {

                "question": request.question,

                "document": request.document_name,

                "answer": (
                    "No relevant information was "
                    "found in the selected document."
                ),

                "sources": []
            }


        # ------------------------------------------
        # Build context
        # ------------------------------------------

        context_parts = []


        for i, item in enumerate(
            retrieved
        ):

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


        context = "\n".join(
            context_parts
        )


        # ------------------------------------------
        # Gemini prompt
        # ------------------------------------------

        prompt = f"""
You are an intelligent research assistant.

Answer the user's question using ONLY the
retrieved document context below.

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


        # ------------------------------------------
        # Generate answer
        # ------------------------------------------

        response = gemini.models.generate_content(

            model="gemini-3.5-flash-lite",

            contents=prompt
        )


        # ------------------------------------------
        # Sources
        # ------------------------------------------

        sources = []


        for item in retrieved:

            sources.append({

                "document": item[
                    "document"
                ],

                "page": item[
                    "page"
                ]
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

            detail=(
                f"Failed to answer question: {str(e)}"
            )
        )