import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Page setup
st.set_page_config(page_title="📚 Chat with Multiple PDFs", layout="wide")
st.title("📚 Chat with Multiple PDFs (LangChain + Streamlit)")

# Sidebar for setup
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Enter your OpenAI API key", type="password")
    uploaded_files = st.file_uploader(
        "Upload your PDF files", type=["pdf"], accept_multiple_files=True
    )
    process_btn = st.button("📄 Process PDFs")

# Session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# Process uploaded PDFs
if process_btn:
    if not api_key:
        st.warning("Please enter your OpenAI API key.")
    elif not uploaded_files:
        st.warning("Please upload at least one PDF.")
    else:
        os.environ["OPENAI_API_KEY"] = api_key
        all_docs = []

        with st.spinner("📚 Loading and processing PDFs..."):
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    loader = PyPDFLoader(tmp_file.name)
                    docs = loader.load()
                    all_docs.extend(docs)

            # Split and embed all documents
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            chunks = splitter.split_documents(all_docs)
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_documents(chunks, embeddings)

            # Store in session
            st.session_state.vectorstore = vectorstore

        st.success("✅ PDFs processed successfully! You can now start chatting.")

# Chat section
if st.session_state.vectorstore:
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.4)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=st.session_state.vectorstore.as_retriever(), memory=memory
    )
    st.session_state.qa_chain = qa_chain

    st.divider()
    st.subheader("💬 Chat with your PDFs")

    user_query = st.text_input("Ask something about your PDFs:")

    if user_query:
        response = qa_chain.run(user_query)
        st.session_state.chat_history.append(("You", user_query))
        st.session_state.chat_history.append(("Bot", response))

    for sender, message in st.session_state.chat_history:
        if sender == "You":
            st.markdown(f"**🧑‍💻 You:** {message}")
        else:
            st.markdown(f"**🤖 Bot:** {message}")
