import streamlit as st
import os
import tempfile
import time
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain.memory import ChatMessageHistory

# ---------------- Streamlit config ----------------
st.set_page_config(page_title="Chat with PDFs — ChatGPT UI", layout="wide", page_icon="🤖")
st.markdown("<style>body { background-color:#0b0f14; color:#e6eef6; }</style>", unsafe_allow_html=True)

# ---------------- CSS for bubbles, avatars, typing ----------------
st.markdown(
    """
    <style>
    .chat-container { max-width: 900px; margin: 0 auto; padding: 1rem; }
    .chat-row { display:flex; gap:12px; margin-bottom:12px; align-items:flex-start; }
    .avatar { width:40px; height:40px; border-radius:50%; flex:0 0 40px; }
    .bubble { padding:14px; border-radius:14px; max-width:78%; line-height:1.4; }
    .user { background:#0b67ff; color:white; margin-left:auto; border-bottom-right-radius:4px; }
    .bot { background:#111418; color:#e6eef6; border:1px solid rgba(255,255,255,0.03); }
    .timestamp { font-size:11px; color:#8b98a8; margin-top:6px; }
    .typing { font-style:italic; color:#8b98a8; }
    /* Auto-scroll anchor spacing */
    #end-of-chat { height: 1px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- Session State ----------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()
if "messages" not in st.session_state:
    # messages: list of dicts {role: "user"/"bot", "text": "...", "time": "..."}
    st.session_state.messages = []

DB_DIR = "chroma_db"

# ---------------- Helpers ----------------
def add_message(role, text):
    t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.messages.append({"role": role, "text": text, "time": t})

def save_uploaded_temp(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name

# ---------------- Processing PDFs ----------------
def process_pdfs(pdf_files, api_key):
    if not api_key:
        return "Please enter OpenAI API key."
    os.environ["OPENAI_API_KEY"] = api_key

    docs = []
    for uploaded in pdf_files:
        tmp_path = save_uploaded_temp(uploaded)
        loader = PyPDFLoader(tmp_path)
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vs = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=DB_DIR)
    vs.persist()
    st.session_state.vectorstore = vs
    st.session_state.chat_history = ChatMessageHistory()
    st.session_state.messages = []
    return "✅ PDFs processed and saved."

# ---------------- Chat / Retrieval ----------------
def get_answer_and_update(query, api_key, simulate_typing=True):
    if st.session_state.vectorstore is None:
        return "Please upload and process PDFs first."

    os.environ["OPENAI_API_KEY"] = api_key
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k":3})

    # history-aware retriever
    context_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("human", "Generate a relevant search query from the conversation.")
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, context_prompt)

    # qa prompt
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant that answers from the provided document context. If unsure, say you don't know.\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    combine_docs_chain = create_stuff_documents_chain(llm, qa_prompt)
    retrieval_chain = create_retrieval_chain(history_aware_retriever, combine_docs_chain)

    # Simulate typing: add temporary "bot typing" message and render it, then replace
    if simulate_typing:
        add_message("bot", "⏳ Bot is typing...")
        # rerun UI to show typing (Streamlit re-renders automatically after function returns)
        time.sleep(0.8)  # brief pause for UI update

    result = retrieval_chain.invoke({"input": query, "chat_history": st.session_state.chat_history.messages})
    answer = result.get("answer", result.get("result", "No answer."))

    # remove typing placeholder (last bot message with typing text)
    if st.session_state.messages and st.session_state.messages[-1]["text"].startswith("⏳ Bot is typing"):
        st.session_state.messages.pop(-1)

    # update memory and messages
    st.session_state.chat_history.add_user_message(query)
    st.session_state.chat_history.add_ai_message(answer)
    add_message("user", query)
    add_message("bot", answer)

    return answer

# ---------------- Layout ----------------
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("OpenAI API Key", type="password")
    uploaded_files = st.file_uploader("Upload PDFs (multi)", type="pdf", accept_multiple_files=True)
    if st.button("Process PDFs"):
        if not api_key:
            st.warning("Enter API key")
        elif not uploaded_files:
            st.warning("Upload PDFs")
        else:
            with st.spinner("Processing PDFs..."):
                msg = process_pdfs(uploaded_files, api_key)
                st.success(msg)
    if st.button("Load existing DB"):
        if not api_key:
            st.warning("Enter API key")
        else:
            # try to load existing chroma
            try:
                embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
                st.session_state.vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
                st.success("Loaded existing vector DB")
            except Exception as e:
                st.error(f"Failed to load DB: {e}")

# Main area: header + chat box
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
st.markdown("## Chat with your PDFs — ChatGPT style")

# Chat area (scrollable container)
chat_placeholder = st.container()

with chat_placeholder:
    # Render messages
    for m in st.session_state.messages:
        if m["role"] == "user":
            st.markdown(
                f"""
                <div class="chat-row" style="justify-content:flex-end">
                    <div class="bubble user" style="display:inline-block">
                        {st.session_state.escape_html(m['text']) if False else m['text']}
                        <div class="timestamp">{m['time']}</div>
                    </div>
                    <img class="avatar" src="https://i.pravatar.cc/40?img=5" />
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="chat-row" style="justify-content:flex-start">
                    <img class="avatar" src="https://cdn-icons-png.flaticon.com/512/4140/4140049.png" />
                    <div class="bubble bot" style="display:inline-block">
                        {m['text']}
                        <div class="timestamp">{m['time']}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

# anchor for autoscroll
st.markdown('<div id="end-of-chat"></div>', unsafe_allow_html=True)

# Input row
col1, col2 = st.columns([8,1])
with col1:
    user_input = st.text_input("Type a message and press Enter", key="user_input")
with col2:
    if st.button("Send"):
        if not api_key:
            st.warning("Enter OpenAI API key in the sidebar.")
        elif not user_input:
            st.info("Type a question.")
        else:
            # add user message immediately and rerun to show it
            add_message("user", user_input)
            # run retrieval & get answer (this will append bot message)
            with st.spinner("Generating answer..."):
                get_answer_and_update(user_input, api_key, simulate_typing=True)
            # clear input
            st.session_state.user_input = ""

# ---------------- Auto-scroll to bottom ----------------
# This JS scrolls the page to the anchor element we added
st.markdown(
    """
    <script>
    (function() {
        var elem = document.getElementById("end-of-chat");
        if (elem) { elem.scrollIntoView({behavior: "smooth", block: "end"}); }
    })();
    </script>
    """,
    unsafe_allow_html=True,
)

st.markdown("</div>", unsafe_allow_html=True)

