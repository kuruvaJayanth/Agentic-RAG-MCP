import sys
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from google import genai

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.retrieval import retrieve_documents
from rag.ingestion import create_chunks_from_upload
from rag.embeddings import store_chunks, chroma_client
from config import GEMINIAI_API_KEY


# -----------------------------------------
# Gemini
# -----------------------------------------

client = genai.Client(
    api_key=GEMINIAI_API_KEY
)


# -----------------------------------------
# FastAPI
# -----------------------------------------

app = FastAPI(
    title="Agentic RAG + MCP API",
    description="Intelligent Research Assistant for Document Question Answering",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------
# Request model
# -----------------------------------------

class AskRequest(BaseModel):
    question: str
    document_name: str = ""


# -----------------------------------------
# Root
# -----------------------------------------

@app.get("/")
def root():
    return {
        "status": "online",
        "message": "Agentic RAG + MCP API is running"
    }


# -----------------------------------------
# Health
# -----------------------------------------

@app.get("/api/health")
def health():
    return {
        "status": "healthy"
    }


# -----------------------------------------
# Documents
# -----------------------------------------

@app.get("/api/documents")
def documents():

    documents_path = PROJECT_ROOT / "data" / "documents"

    if not documents_path.exists():
        return {
            "documents": []
        }

    files = [
        file.name
        for file in documents_path.glob("*.pdf")
    ]

    return {
        "documents": files
    }


# -----------------------------------------
# Upload PDF
# -----------------------------------------

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

        # Read uploaded PDF
        file_bytes = await file.read()

        if not file_bytes:
            raise HTTPException(
                status_code=400,
                detail="The uploaded PDF is empty."
            )

        # Save uploaded PDF locally
        documents_path = PROJECT_ROOT / "data" / "documents"

        documents_path.mkdir(
            parents=True,
            exist_ok=True
        )

        pdf_path = documents_path / file.filename

        with open(pdf_path, "wb") as f:
            f.write(file_bytes)

        # Object compatible with existing ingestion function
        class UploadedPDF:

            def __init__(self, name, data):
                self.name = name
                self.data = data

            def getvalue(self):
                return self.data

        uploaded_file = UploadedPDF(
            file.filename,
            file_bytes
        )

        # Extract and chunk
        chunks = create_chunks_from_upload(
            uploaded_file
        )

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="Could not extract readable text from this PDF."
            )

        # Use the SAME collection as existing documents
        collection = chroma_client.get_or_create_collection(
            name="documents"
        )

        # Store embeddings
        count = store_chunks(
            chunks,
            collection
        )

        return {
            "success": True,
            "message": "PDF uploaded and indexed successfully.",
            "document": file.filename,
            "chunks": count
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


# -----------------------------------------
# Ask Question
# -----------------------------------------

@app.post("/api/ask")
def ask(request: AskRequest):

    results = retrieve_documents(
        query=request.question,
        top_k=5,
        document_name=request.document_name
    )

    if not results:

        return {
            "question": request.question,
            "document": request.document_name,
            "answer": "No relevant information was found in the document.",
            "sources": []
        }

    # Build context
    context_parts = []

    for i, result in enumerate(results):

        context_parts.append(
            f"""
SOURCE {i + 1}

Document:
{result['source']}

Page:
{result['page']}

Content:
{result['text']}
"""
        )

    context = "\n".join(context_parts)

    # Grounded prompt
    prompt = f"""
You are an intelligent research assistant.

Answer the user's question using ONLY the retrieved
document context below.

If the answer cannot be found in the context,
say that the information is not available in the
retrieved document context.

Do not invent information.

User Question:
{request.question}

Retrieved Document Context:
{context}

Give a clear and concise answer.
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )

    answer = response.text

    sources = []

    for result in results:

        sources.append({
            "document": result["source"],
            "page": result["page"]
        })

    return {
        "question": request.question,
        "document": request.document_name,
        "answer": answer,
        "sources": sources
    }