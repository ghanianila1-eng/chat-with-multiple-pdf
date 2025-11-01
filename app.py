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

# --- Page setup ---
st.set_page_config(page_title="📘 Chat with your PDF", page_icon="🤖", layout="wide")
st.title("💬 Chat with Your PDF (Streamlit + LangChain)")

# --- Sidebar: API key and upload ---
st.sidebar.header("⚙️ Settings")
api_key = st.sidebar.text_input("🔑 Enter your OpenAI API Key", type="password")
uploaded_files = st.sidebar.file_uploader("📎 Upload PDF(s)", type="pdf", accept_multiple_files=True)

# --- Initialize session state ---
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()

# --- Process uploaded files ---
if uploaded_files and api_key:
    os.environ["OPENAI_API_KEY"] = api_key
    st.success(f"✅ {len(uploaded_files)} PDF(s) uploaded successfully!")

    all_docs = []
    for uploaded_file in uploaded_files:
        # Save uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        # Load PDF using temporary file path
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        chunks = splitter.split_documents(pages)
        all_docs.extend(chunks)

        # Clean up temp file
        os.remove(tmp_path)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    st.session_state.vectorstore = Chroma.from_documents(all_docs, embeddings)
    st.session_state.chat_history = ChatMessageHistory()

# --- Chat logic ---
def chat_with_pdf(query):
    if not api_key:
        return "Please enter your OpenAI API key."
    if not st.session_state.vectorstore:
        return "Please upload and process a PDF first."

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})

    # Context-aware retriever
    context_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("human", "Generate a search query relevant to the current question.")
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, context_prompt)

    # Retrieval + QA chain
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use the following context to answer accurately:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    combine_docs_chain = create_stuff_documents_chain(llm, qa_prompt)
    retrieval_chain = create_retrieval_chain(history_aware_retriever, combine_docs_chain)

    result = retrieval_chain.invoke({
        "input": query,
        "chat_history": st.session_state.chat_history.messages
    })

    # Update chat history
    st.session_state.chat_history.add_user_message(query)
    st.session_state.chat_history.add_ai_message(result["answer"])

    return result["answer"]

# --- Sidebar: Chat history viewer ---
st.sidebar.markdown("---")
st.sidebar.subheader("🕘 Chat History")
if len(st.session_state.chat_history.messages) == 0:
    st.sidebar.info("No messages yet.")
else:
    for msg in st.session_state.chat_history.messages[::-1]:
        role = "👩‍💻 You" if msg.type == "human" else "🤖 Bot"
        st.sidebar.markdown(f"**{role}:** {msg.content}")

if st.sidebar.button("🧹 Clear Chat History"):
    st.session_state.chat_history = ChatMessageHistory()
    st.sidebar.success("Chat history cleared!")

# --- Main Chat UI ---
st.subheader("💭 Chat Window")
for msg in st.session_state.chat_history.messages:
    role = "🧑‍💻 You" if msg.type == "human" else "🤖 Bot"
    with st.chat_message(role):
        st.markdown(msg.content)

# --- Chat input ---
if prompt := st.chat_input("Ask a question about your PDFs..."):
    with st.chat_message("🧑‍💻 You"):
        st.markdown(prompt)
    with st.chat_message("🤖 Bot"):
        with st.spinner("Thinking..."):
            response = chat_with_pdf(prompt)
            st.markdown(response)
