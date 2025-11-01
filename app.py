import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ChatMessageHistory

# Directory for persistent embeddings
DB_DIR = "db"

# Initialize session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()


# ------------- Helper Functions -------------

def process_pdfs(pdf_files, api_key):
    """Loads multiple PDFs and saves embeddings persistently."""
    os.environ["OPENAI_API_KEY"] = api_key

    docs = []
    for uploaded_file in pdf_files:
        loader = PyPDFLoader(uploaded_file.name)
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectorstore = Chroma.from_documents(
        chunks, embedding=embeddings, persist_directory=DB_DIR
    )
    vectorstore.persist()
    st.session_state.vectorstore = vectorstore
    st.session_state.chat_history = ChatMessageHistory()
    st.success("✅ PDFs processed and saved successfully!")


def load_existing_db(api_key):
    """Loads existing Chroma DB if available."""
    os.environ["OPENAI_API_KEY"] = api_key
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    st.session_state.vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    st.success("✅ Loaded existing database successfully!")


def chat_with_docs(query):
    """Handles conversational question answering."""
    vectorstore = st.session_state.vectorstore
    chat_history = st.session_state.chat_history

    if vectorstore is None:
        st.warning("⚠️ Please upload PDFs or load an existing database first.")
        return

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Prompt template with memory
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant that answers questions using the provided context. "
         "If the answer isn't found, say you don’t know.\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    combine_docs_chain = create_stuff_documents_chain(llm, qa_prompt)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

    result = retrieval_chain.invoke({"input": query, "chat_history": chat_history.messages})

    # Save conversation
    chat_history.add_user_message(query)
    chat_history.add_ai_message(result["answer"])

    return result["answer"]

# ------------- Streamlit UI -------------

st.set_page_config(page_title="📚 PDF Chatbot", layout="wide")
st.title("🤖 Chat with Multiple PDFs (LangChain + Streamlit)")

st.sidebar.header("Configuration")
api_key = st.sidebar.text_input("🔑 Enter your OpenAI API Key", type="password")

# File uploader
uploaded_pdfs = st.sidebar.file_uploader(
    "📎 Upload one or more PDFs",
    type="pdf",
    accept_multiple_files=True
)

col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("Process PDFs"):
        if uploaded_pdfs and api_key:
            process_pdfs(uploaded_pdfs, api_key)
        else:
            st.warning("Please upload PDFs and enter your API key first.")
with col2:
    if st.button("Load Existing DB"):
        if api_key:
            load_existing_db(api_key)
        else:
            st.warning("Please enter your OpenAI API key.")

# Main Chat Interface
st.markdown("---")
st.subheader("💬 Chat")

user_input = st.text_input("Ask a question about your PDFs:", key="input")
if st.button("Ask"):
    if user_input:
        answer = chat_with_docs(user_input)
        st.markdown(f"**🧠 Answer:** {answer}")
    else:
        st.warning("Please enter a question first.")

# Display chat history
if len(st.session_state.chat_history.messages) > 0:
    st.markdown("### 🗂 Chat History")
    for msg in st.session_state.chat_history.messages:
        role = "🧑‍💻 You" if msg.type == "human" else "🤖 Bot"
        st.markdown(f"**{role}:** {msg.content}")
