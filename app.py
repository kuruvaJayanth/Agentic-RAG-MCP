import asyncio
import streamlit as st

from agent.agent import ask_agent
from rag.embeddings import store_uploaded_document


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Agentic RAG + MCP",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# SESSION STATE
# ============================================================

if "document_processed" not in st.session_state:
    st.session_state.document_processed = False

if "document_name" not in st.session_state:
    st.session_state.document_name = None

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# HEADER
# ============================================================

st.title("🤖 Agentic RAG + MCP")

st.write(
    "Upload any PDF and ask questions about it."
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📄 Upload PDF")


    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"]
    )


    if uploaded_file:

        st.info(
            f"📄 {uploaded_file.name}"
        )


        if st.button(
            "⚡ Process Document",
            type="primary",
            use_container_width=True
        ):

            with st.spinner(
                "📚 Processing PDF..."
            ):

                try:

                    count = (
                        store_uploaded_document(
                            uploaded_file
                        )
                    )


                    if count > 0:

                        st.session_state.document_processed = True

                        st.session_state.document_name = (
                            uploaded_file.name
                        )

                        st.session_state.chunk_count = count

                        st.session_state.messages = []


                        st.success(
                            "✅ PDF processed!"
                        )


                        st.info(
                            f"{count} chunks indexed."
                        )


                    else:

                        st.error(
                            "No text found in PDF."
                        )


                except Exception as e:

                    st.error(
                        "PDF processing failed."
                    )

                    st.exception(e)


    # ========================================================
    # STATUS
    # ========================================================

    st.divider()

    st.subheader(
        "📊 Document Status"
    )


    if st.session_state.document_processed:

        st.success(
            "🟢 Ready"
        )


        st.write(
            f"**File:** "
            f"{st.session_state.document_name}"
        )


        st.write(
            f"**Chunks:** "
            f"{st.session_state.chunk_count}"
        )


    else:

        st.warning(
            "No document loaded"
        )


    # ========================================================
    # CLEAR CHAT
    # ========================================================

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# ACTIVE DOCUMENT
# ============================================================

if st.session_state.document_processed:

    st.success(
        f"📄 Active document: "
        f"**{st.session_state.document_name}**"
    )

else:

    st.info(
        "Upload and process a PDF from the sidebar."
    )


# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask something about your document..."
)


# ============================================================
# ASK AGENT
# ============================================================

if question:

    if not st.session_state.document_processed:

        st.warning(
            "Please upload and process a PDF first."
        )

        st.stop()


    # --------------------------------------------------------
    # USER MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append({

        "role": "user",

        "content": question

    })


    with st.chat_message(
        "user"
    ):

        st.markdown(
            question
        )


    # --------------------------------------------------------
    # AGENT
    # --------------------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "🤖 Agent is researching..."
        ):

            try:

                result = asyncio.run(

                    ask_agent(

                        question,

                        st.session_state.document_name,

                        st.session_state.messages[:-1]

                    )

                )


                answer = result["answer"]


                st.markdown(
                    answer
                )


                # ------------------------------------------------
                # SAVE ANSWER
                # ------------------------------------------------

                st.session_state.messages.append({

                    "role": "assistant",

                    "content": answer

                })


                # ------------------------------------------------
                # AGENT DETAILS
                # ------------------------------------------------

                with st.expander(
                    "🧠 Agent Details"
                ):

                    col1, col2 = st.columns(2)


                    with col1:

                        st.metric(
                            "Reasoning Steps",
                            result["steps"]
                        )


                    with col2:

                        st.metric(
                            "MCP Tools",
                            len(
                                result["tools_used"]
                            )
                        )


                    if result["tools_used"]:

                        st.write(
                            "**MCP Tools Used:**"
                        )


                        for tool in result[
                            "tools_used"
                        ]:

                            st.write(
                                f"🔧 `{tool}`"
                            )


            except Exception as e:

                st.error(
                    "❌ Agent failed."
                )

                st.exception(e)


# ============================================================
# SUGGESTIONS
# ============================================================

if (
    st.session_state.document_processed
    and not st.session_state.messages
):

    st.subheader(
        "💡 Try asking"
    )


    col1, col2 = st.columns(2)


    with col1:

        st.info(
            "What is the main purpose of this document?"
        )

        st.info(
            "What methodology is proposed?"
        )


    with col2:

        st.info(
            "What are the main findings?"
        )

        st.info(
            "What are the limitations?"
        )


# ============================================================
# TECHNOLOGY STACK
# ============================================================

st.divider()

st.subheader(
    "🛠️ Technology Stack"
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.info(
        "**Gemini**\n\n"
        "LLM + Agent reasoning"
    )


with col2:

    st.info(
        "**MCP**\n\n"
        "Tool communication"
    )


with col3:

    st.info(
        "**ChromaDB**\n\n"
        "Vector database"
    )


with col4:

    st.info(
        "**Sentence Transformers**\n\n"
        "Local embeddings"
    )