import sys
from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# MCP
# ============================================================

from mcp.server.fastmcp import FastMCP


# ============================================================
# RAG
# ============================================================

from rag.retrieval import retrieve_documents


# ============================================================
# AGENTIC TOOLS
# ============================================================

from mcp_server.agentic_tools import (
    refine_query,
    extract_entities,
    check_relevance
)


# ============================================================
# MCP SERVER
# ============================================================

mcp = FastMCP(
    "Agentic RAG Server"
)


# ============================================================
# RETRIEVAL TOOL
# ============================================================

@mcp.tool()
def retrieve_documents_tool(
    query: str,
    top_k: int = 5,
    document_name: str = ""
) -> str:

    """
    Retrieve relevant information from
    the selected PDF document.
    """

    results = retrieve_documents(
        query=query,
        top_k=top_k,
        document_name=document_name
    )

    if not results:

        return (
            "No relevant information was found "
            "in the selected document."
        )

    output = []

    for i, result in enumerate(results):

        output.append(
            f"""
--- Retrieved Chunk {i + 1} ---

Source:
{result['source']}

Page:
{result['page']}

Content:
{result['text']}
"""
        )

    return "\n".join(output)


# ============================================================
# QUERY REFINEMENT
# ============================================================

@mcp.tool()
def refine_query_tool(
    query: str
) -> str:

    """
    Refine the user's question.
    """

    return refine_query(
        query
    )


# ============================================================
# ENTITY EXTRACTION
# ============================================================

@mcp.tool()
def extract_entities_tool(
    query: str
) -> str:

    """
    Extract important entities and keywords.
    """

    result = extract_entities(
        query
    )

    return str(result)


# ============================================================
# RELEVANCE CHECK
# ============================================================

@mcp.tool()
def check_relevance_tool(
    query: str,
    retrieved_text: str
) -> str:

    """
    Check whether retrieved information
    is relevant.
    """

    result = check_relevance(
        query,
        retrieved_text
    )

    return str(result)


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    try:

        print(
            "MCP Server starting...",
            file=sys.stderr,
            flush=True
        )

        mcp.run()

    except Exception as e:

        print(
            "\nMCP SERVER ERROR:",
            file=sys.stderr,
            flush=True
        )

        import traceback

        traceback.print_exc(
            file=sys.stderr
        )

        raise