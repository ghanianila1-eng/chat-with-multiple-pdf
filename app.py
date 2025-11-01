import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain.memory import ChatMessageHistory

# -------------------- 🎨 PAGE CONFIG --------------------
st.set_page_config(
    page_title="Chat with Your PDF",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS styling for professional look
st.markdown("""
    <style>
        .main {
            background: linear-gradient(180deg, #f9fafb 0%, #eef2ff 100%);
            padding: 2rem;
        }
        .chat-bubble-user {
            background-color: #6366f1;
            color: white;
            padding: 12px;
            border-radius: 12px;
            margin: 8px 0;
            width: fit-content;
            max-width: 80%;
        }
        .chat-bubble-bot {
            background-color: #f3f4f6;
            color: #111827;
            padding: 12px;
            border-radius: 12px;
            margin: 8px 0;
            width: fit-content;
            max-width: 80%;
        }
        .header {
            background: linear-gradient(to right, #6366f1, #8b5cf6);
            color: white;
            text-align: center;
            padding: 1rem;
            border-radius: 12px;
            margin-bottom: 20px;
        }
        .stButton>button {
            background-color: #4f46e5;
            color: white;
            border-radius: 10px;
            font-weight: 600;
            padding: 10px 24px;
        }
        .stButton>button:hover {
            background-color: #3730a3;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------- 🧠 INITIALIZATION --------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()

# -------------------- 🏷️ HEADER --------------------
st.markdown('<div class="header"><h2>💬 Chat with Your PDF</h2><p>Powered by LangChain + OpenAI + Streamlit</p></div>', unsafe_allow_html=True)

# -------------------- ⚙️ SIDEBAR --------------------
st.sidebar.image("https://huggingface.co/front/assets/huggingface_logo-noborder.svg", width=150)
st.sidebar.markdown("### ⚙️ App Settings")

api_key = st.sidebar.text_input("🔑 OpenAI API Key", type="password")
uploaded_files = st.sidebar.file_uploader("📎 Upload PDF(s)", type="pdf", accept_multiple_files=True)

if uploaded_files and api_key:
    os.environ["OPENAI_API_KEY"] = api_key

    st.sidebar.info("Processing your PDF(s)... Please wait ⏳")

    all_docs = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)
        pages = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        chunks = splitter.split_documents(pages)
        all_docs.extend(chunks)

        os.remove(tmp_path)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    st.session_state.vectorstore = Chroma.from_documents(all_docs, embeddings)
    st.sidebar.success("✅ PDF(s) processed successfully!")
    st.session_state.chat_history = ChatMessageHistory()

# -------------------- 💬 CHAT SECTION --------------------
st.subheader("🧠 Ask anything about your PDF")

def chat_with_pdf(query):
    if not api_key:
        return "Please enter your OpenAI API key."
    if not st.session_state.vectorstore:
        return "Please upload and process a PDF first."

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})

    # Create history-aware retriever
    context_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("human", "Generate a search query relevant to the user's latest question.")
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, context_prompt)

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use the given context to answer accurately.\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    combine_docs_chain = create_stuff_documents_chain(llm, qa_prompt)
    retrieval_chain = create_retrieval_chain(history_aware_retriever, combine_docs_chain)

    result = retrieval_chain.invoke({
        "input": query,
        "chat_history": st.session_state.chat_history.messages
    })

    st.session_state.chat_history.add_user_message(query)
    st.session_state.chat_history.add_ai_message(result["answer"])
    return result["answer"]

# Display chat messages with nice bubbles
for msg in st.session_state.chat_history.messages:
    if msg.type == "human":
        st.markdown(f"<div class='chat-bubble-user'>🧑‍💻 You: {msg.content}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-bubble-bot'>🤖 Bot: {msg.content}</div>", unsafe_allow_html=True)

# Chat input field
prompt = st.chat_input("Ask your question here...")

if prompt:
    with st.spinner("🤔 Thinking..."):
        answer = chat_with_pdf(prompt)
    st.markdown(f"<div class='chat-bubble-user'>🧑‍💻 You: {prompt}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='chat-bubble-bot'>🤖 Bot: {answer}</div>", unsafe_allow_html=True)

# -------------------- 🧹 CLEAR HISTORY BUTTON --------------------
if st.sidebar.button("🧹 Clear Chat History"):
    st.session_state.chat_history = ChatMessageHistory()
    st.sidebar.success("Chat history cleared!")
