import asyncio
import sys
from pathlib import Path

from google import genai
from google.genai import types

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# CONFIG
# ============================================================

from config import GEMINIAI_API_KEY


# ============================================================
# GEMINI CLIENT
# ============================================================

gemini = genai.Client(
    api_key=GEMINIAI_API_KEY
)


# ============================================================
# AGENT
# ============================================================

async def ask_agent(
    question,
    document_name=None,
    chat_history=None
):

    # --------------------------------------------------------
    # PYTHON EXECUTABLE
    # --------------------------------------------------------
    # Use the Python interpreter that is currently running
    # the application.
    #
    # This works both locally and on Streamlit Cloud.

    python_executable = sys.executable


    # --------------------------------------------------------
    # MCP SERVER
    # --------------------------------------------------------

    server_params = StdioServerParameters(

        command=python_executable,

        args=[
            "-m",
            "mcp_server.server"
        ],

        cwd=str(
            PROJECT_ROOT
        )

    )


    # --------------------------------------------------------
    # CONNECT TO MCP
    # --------------------------------------------------------

    async with stdio_client(
        server_params
    ) as (read, write):

        async with ClientSession(
            read,
            write
        ) as session:

            await session.initialize()


            # =================================================
            # GET TOOLS
            # =================================================

            tools_response = (
                await session.list_tools()
            )


            function_declarations = []


            for tool in tools_response.tools:

                function_declarations.append(

                    types.FunctionDeclaration(

                        name=tool.name,

                        description=(
                            tool.description
                            or ""
                        ),

                        parameters_json_schema=(
                            tool.inputSchema
                        )

                    )

                )


            gemini_tools = [

                types.Tool(

                    function_declarations=(
                        function_declarations
                    )

                )

            ]


            # =================================================
            # DOCUMENT
            # =================================================

            if document_name:

                document_instruction = f"""

The user selected this PDF:

{document_name}

You MUST retrieve information ONLY
from this document.

"""

            else:

                document_instruction = """

Use the available document knowledge base.

"""


            # =================================================
            # CHAT HISTORY
            # =================================================

            history_text = ""


            if chat_history:

                history_text = (
                    "\nPrevious conversation:\n"
                )


                for message in chat_history[-6:]:

                    history_text += (

                        f"\n{message['role'].upper()}: "
                        f"{message['content']}\n"

                    )


            # =================================================
            # STEP 1
            # =================================================

            retrieval_prompt = f"""

You are an intelligent document research assistant.

{document_instruction}

{history_text}

Current user question:

{question}

Use the MCP tool:

retrieve_documents_tool

Call retrieve_documents_tool exactly ONE time.

Create a useful search query based on the question.

The tool arguments must contain:

query
top_k
document_name

Use this exact document name:

{document_name if document_name else ""}

Do not repeatedly retrieve documents.

"""


            print(
                "\n🧠 Agent is analyzing..."
            )


            response = gemini.models.generate_content(

                model="gemini-3.6-flash",

                contents=retrieval_prompt,

                config=types.GenerateContentConfig(

                    tools=gemini_tools

                )

            )


            # =================================================
            # CHECK TOOL CALL
            # =================================================

            function_calls = (
                response.function_calls
            )


            if not function_calls:

                return {

                    "answer": response.text,

                    "tools_used": [],

                    "steps": 1

                }


            # =================================================
            # TOOL CALL
            # =================================================

            function_call = function_calls[0]


            tool_name = (
                function_call.name
            )


            tool_args = dict(
                function_call.args
            )


            # Force selected document

            if document_name:

                tool_args[
                    "document_name"
                ] = document_name


            if "top_k" not in tool_args:

                tool_args[
                    "top_k"
                ] = 5


            print(
                f"\n🔧 Tool selected: {tool_name}"
            )


            print(
                f"Arguments: {tool_args}"
            )


            # =================================================
            # MCP CALL
            # =================================================

            result = await session.call_tool(

                tool_name,

                arguments=tool_args

            )


            # =================================================
            # EXTRACT CONTEXT
            # =================================================

            retrieved_context = ""


            for content in result.content:

                if hasattr(
                    content,
                    "text"
                ):

                    retrieved_context += (
                        content.text
                    )


            print(
                "\n✓ Documents retrieved"
            )


            # =================================================
            # FINAL ANSWER
            # =================================================

            final_prompt = f"""

You are an AI document research assistant.

Answer the user's question using ONLY
the retrieved document information.

Selected document:

{document_name}

Previous conversation:

{history_text}

Current question:

{question}

Retrieved information:

{retrieved_context}

Rules:

1. Answer directly.
2. Use ONLY retrieved information.
3. Do not invent facts.
4. Do not use outside knowledge.
5. If information is insufficient, say so.
6. Mention page numbers when available.
7. Mention the document name when useful.
8. Keep the answer clear and structured.

"""


            print(
                "\n🤖 Generating final answer..."
            )


            final_response = (
                gemini.models.generate_content(

                    model="gemini-3.6-flash",

                    contents=final_prompt

                )
            )


            return {

                "answer": final_response.text,

                "tools_used": [
                    tool_name
                ],

                "steps": 2

            }


# ============================================================
# TERMINAL TEST
# ============================================================

async def main():

    print(
        "\n======================================"
    )


    print(
        "       AGENTIC RAG + MCP"
    )


    print(
        "======================================"
    )


    document_name = input(

        "\nPDF name "
        "(Enter for all documents): "

    )


    if not document_name.strip():

        document_name = None


    question = input(

        "\nAsk your question: "

    )


    if not question.strip():

        return


    result = await ask_agent(

        question,

        document_name

    )


    print(
        "\n======================================"
    )


    print(
        "FINAL ANSWER"
    )


    print(
        "======================================"
    )


    print(
        result["answer"]
    )


    print(
        f"\nReasoning steps: "
        f"{result['steps']}"
    )


    print(
        "Tools used: "
        + ", ".join(
            result["tools_used"]
        )
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    asyncio.run(
        main()
    )