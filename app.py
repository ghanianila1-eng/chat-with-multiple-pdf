# app.py
import os
import tempfile
import time
from datetime import datetime

import streamlit as st

# LangChain / Chroma imports (versions pinned in requirements.txt)
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain.memory import ChatMessageHistory

# ----------------- Page config & CSS -----------------
st.set_page_config(page_title="Pro PDF Chatbot", page_icon="🤖", layout="wide")
st.markdown(
    """
    <style>
    .header {background: linear-gradient(90deg,#4f46e5,#8b5cf6); color: white; padding: 14px; border-radius: 10px;}
    .card { background: white; padding: 12px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.06);}
    .chat-user { background:#4f46e5; color:white; padding:10px; border-radius:12px; display:inline-block; max-width:80%;}
    .chat-bot { background:#f3f4f6; color:#111827; padding:10px; border-radius:12px; display:inline-block; max-width:80%;}
    .small { font-size:12px; color:#6b7280; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="header"><h2>📚 Professional PDF Chatbot — LangChain + Chroma + GPT-3.5</h2></div>', unsafe_allow_html=True)
st.write("Upload multiple PDFs, build a persistent knowledge base, and chat naturally — conversation history is on the sidebar.")

# ----------------- Session State init -----------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()
if "messages" not in st.session_state:
    # messages: list of dicts {role: "user"/"bot", "text": "...", "time": "..."}
    st.session_state.messages = []
if "db_initialized" not in st.session_state:
    st.session_state.db_initialized = False

DB_DIR = "chroma_db"  # persistent directory inside repository / Space

# ----------------- Helper functions -----------------
def _now_ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def save_uploaded_tmp(uploaded_file):
    """Save a Streamlit UploadedFile to a temporary path and return path."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name

def build_or_update_chroma(pdf_files, openai_api_key, persist=True):
    """
    Loads multiple PDFs, splits, creates embeddings, and builds a Chroma store.
    If persist=True, the DB is saved to DB_DIR.
    """
    os.environ["OPENAI_API_KEY"] = openai_api_key

    docs = []
    for f in pdf_files:
        tmp_path = save_uploaded_tmp(f)
        loader = PyPDFLoader(tmp_path)
        docs.extend(loader.load())
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vect = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=DB_DIR if persist else None)
    if persist:
        vect.persist()
    return vect

def load_existing_chroma(openai_api_key):
    os.environ["OPENAI_API_KEY"] = openai_api_key
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vect = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    return vect

def add_message(role, text):
    st.session_state.messages.append({"role": role, "text": text, "time": _now_ts()})

# ----------------- Sidebar (controls & history) -----------------
with st.sidebar:
    st.markdown("## ⚙️ Controls")
    openai_key = st.text_input("Enter OpenAI API Key (sk-...)", type="password")
    uploaded = st.file_uploader("Upload PDFs (multiple allowed)", accept_multiple_files=True, type=["pdf"])

    st.markdown("---")
    st.markdown("### 🔁 Database")
    col1, col2 = st.columns([2,2])
    with col1:
        if st.button("Build DB from Upload"):
            if not openai_key:
                st.warning("Add your OpenAI API key first.")
            elif not uploaded:
                st.warning("Upload 1+ PDF files to build DB.")
            else:
                with st.spinner("Processing PDFs and building Chroma DB..."):
                    try:
                        vect = build_or_update_chroma(uploaded, openai_key, persist=True)
                        st.session_state.vectorstore = vect
                        st.session_state.db_initialized = True
                        st.success("DB built and persisted to chroma_db/")
                    except Exception as e:
                        st.error(f"Failed to build DB: {e}")

    with col2:
        if st.button("Load existing DB"):
            if not openai_key:
                st.warning("Add your OpenAI API key first.")
            else:
                try:
                    with st.spinner("Loading existing DB..."):
                        vect = load_existing_chroma(openai_key)
                        st.session_state.vectorstore = vect
                        st.session_state.db_initialized = True
                        st.success("Existing DB loaded.")
                except Exception as e:
                    st.error(f"Failed to load DB: {e}")

    st.markdown("---")
    st.markdown("### 🕘 Conversation History")
    if len(st.session_state.messages) == 0:
        st.info("No messages yet. Start a conversation in the main area.")
    else:
        # show messages newest-first but keep original order preserved in main window
        for m in reversed(st.session_state.messages):
            who = "You" if m["role"] == "user" else "Bot"
            st.markdown(f"**{who}** — {m['time']}")
            st.markdown(m["text"])
            st.markdown("---")

    if st.button("🧹 Clear history"):
        st.session_state.chat_history = ChatMessageHistory()
        st.session_state.messages = []
        st.success("Cleared conversation history.")

    st.markdown("---")
    st.caption("Pro PDF Chatbot — multi-file, persistent Chroma DB, GPT-3.5")

# ----------------- Main chat area -----------------
col_main, col_right = st.columns([3,1])

with col_main:
    st.markdown("## 💬 Chat (continuous conversation)")
    st.markdown("Ask questions about all uploaded PDFs. The assistant keeps context across turns.")
    # Show a visual status
    if st.session_state.vectorstore is None:
        st.info("No knowledge base loaded. Use the sidebar to Build DB or Load existing DB.")
    else:
        st.success("Knowledge base loaded — ready to answer!")

    # Display main conversation (in order)
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"<div style='text-align:right'><div class='chat-user'>{msg['text']}</div><div class='small'>{msg['time']}</div></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='text-align:left'><div class='chat-bot'>{msg['text']}</div><div class='small'>{msg['time']}</div></div>", unsafe_allow_html=True)

    # Chat input — st.chat_input supports Enter to submit
    prompt = st.chat_input("Ask a question about the uploaded PDFs...")

    if prompt:
        # immediate user message
        add_message("user", prompt)

        # generate answer using retrieval + LLM
        if st.session_state.vectorstore is None:
            add_message("bot", "Please build or load the knowledge base first (sidebar).")
        else:
            try:
                with st.spinner("Retrieving context and generating answer..."):
                    # LLM and retrieval chain
                    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
                    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})

                    # history-aware retriever (improves followups)
                    context_prompt = ChatPromptTemplate.from_messages([
                        MessagesPlaceholder(variable_name="chat_history"),
                        ("human", "{input}"),
                        ("human", "Generate a precise search query for the retriever.")
                    ])
                    history_aware_retriever = create_history_aware_retriever(llm, retriever, context_prompt)

                    qa_prompt = ChatPromptTemplate.from_messages([
                        ("system", "You are an assistant. Use the retrieved document context to answer simply and accurately. If the answer is not in the documents, say you don't know."),
                        MessagesPlaceholder(variable_name="chat_history"),
                        ("human", "{input}")
                    ])
                    combine_docs_chain = create_stuff_documents_chain(llm, qa_prompt)
                    retrieval_chain = create_retrieval_chain(history_aware_retriever, combine_docs_chain)

                    # invoke with current chat history
                    result = retrieval_chain.invoke({
                        "input": prompt,
                        "chat_history": st.session_state.chat_history.messages
                    })

                    answer = result.get("answer") or result.get("result") or str(result)
                    # update LangChain chat memory + UI messages
                    st.session_state.chat_history.add_user_message(prompt)
                    st.session_state.chat_history.add_ai_message(answer)
                    add_message("bot", answer)
            except Exception as e:
                add_message("bot", f"Error generating answer: {e}")

with col_right:
    st.markdown("### ⚡ Quick controls")
    st.markdown("- Build DB from uploaded PDFs (persisted to `chroma_db/`)") 
    st.markdown("- Load previously built DB (if exists) to avoid reprocessing")
    st.markdown("- Sidebar shows full chat history and Clear button")
    st.markdown("---")
    st.markdown("### 📌 Tips")
    st.markdown("• Use concise questions for better retrieval\n• For complex queries, ask follow-ups")

# ----------------- Auto-scroll small hack (anchor) -----------------
st.markdown('<div id="end-of-chat" style="height:1px;"></div>', unsafe_allow_html=True)
st.markdown(
    """
    <script>
    (function() {
      var elem = document.getElementById('end-of-chat');
      if (elem) { elem.scrollIntoView({behavior: 'smooth', block: 'end'}); }
    })();
    </script>
    """,
    unsafe_allow_html=True,
)
