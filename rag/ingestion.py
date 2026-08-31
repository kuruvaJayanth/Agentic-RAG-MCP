from pathlib import Path
from pypdf import PdfReader
import io


DOCUMENTS_DIR = Path("data/documents")


# ============================================================
# LOAD PDF DOCUMENTS FROM DATASET
# ============================================================

def load_documents():

    documents = []

    for pdf_path in DOCUMENTS_DIR.glob("*.pdf"):

        reader = PdfReader(pdf_path)

        for page_number, page in enumerate(reader.pages):

            text = page.extract_text()

            if text and text.strip():

                documents.append({

                    "text": text.strip(),

                    "source": pdf_path.name,

                    "page": page_number + 1

                })

    return documents


# ============================================================
# CHUNK TEXT
# ============================================================

def chunk_text(
    text,
    chunk_size=1000,
    overlap=200
):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():

            chunks.append(
                chunk.strip()
            )

        start += (
            chunk_size - overlap
        )

    return chunks


# ============================================================
# CREATE CHUNKS FROM EXISTING DOCUMENTS
# ============================================================

def create_chunks():

    documents = load_documents()

    chunks = []

    for document in documents:

        text_chunks = chunk_text(
            document["text"]
        )

        for chunk in text_chunks:

            chunks.append({

                "text": chunk,

                "source": document["source"],

                "page": document["page"]

            })

    return chunks


# ============================================================
# CREATE CHUNKS FROM UPLOADED PDF
# ============================================================

def create_chunks_from_upload(
    uploaded_file
):

    chunks = []

    # --------------------------------------------------------
    # Read uploaded PDF
    # --------------------------------------------------------

    pdf_bytes = uploaded_file.getvalue()

    pdf_stream = io.BytesIO(
        pdf_bytes
    )

    reader = PdfReader(
        pdf_stream
    )


    # --------------------------------------------------------
    # Process every page
    # --------------------------------------------------------

    for page_number, page in enumerate(
        reader.pages
    ):

        text = page.extract_text()


        if not text or not text.strip():

            continue


        text_chunks = chunk_text(
            text.strip()
        )


        # ----------------------------------------------------
        # Store chunks
        # ----------------------------------------------------

        for chunk in text_chunks:

            chunks.append({

                "text": chunk,

                "source": uploaded_file.name,

                "page": page_number + 1

            })


    return chunks


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    chunks = create_chunks()

    print(
        f"\nCreated {len(chunks)} chunks.\n"
    )


    for i, chunk in enumerate(
        chunks[:3]
    ):

        print(
            "=" * 60
        )

        print(
            f"Chunk: {i + 1}"
        )

        print(
            f"Source: {chunk['source']}"
        )

        print(
            f"Page: {chunk['page']}"
        )

        print(
            chunk["text"][:500]
        )