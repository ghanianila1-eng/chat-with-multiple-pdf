import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain.memory import ChatMessageHistory


# ---------------------------- CONFIG ----------------------------
st.set_page_config(page_title="📚 Chat with Multiple PDFs", layout="wide")
st.title("📚 Professional PDF Chatbot — LangChain + Chroma + GPT-3.5")
st.markdown("Upload multiple PDFs, build a persistent knowledge base, and chat naturally — conversation history is on the sidebar.")


# ---------------------------- SESSION STATE ----------------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()
if "api_key" not in st.session_state:
    st.session_state.api_key = ""


# ---------------------------- SIDEBAR ----------------------------
st.sidebar.header("⚙️ Settings")

api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")
uploaded_files = st.sidebar.file_uploader(
    "📎 Upload one or more PDFs", type=["pdf"], accept_multiple_files=True
)
process_btn = st.sidebar.button("Process PDFs")

if process_btn:
    if not api_key:
        st.sidebar.error("⚠️ Please enter your OpenAI API key first.")
    elif not uploaded_files:
        st.sidebar.error("⚠️ Please upload at least one PDF.")
    else:
        os.environ["OPENAI_API_KEY"] = api_key
        st.session_state.api_key = api_key
        st.session_state.chat_history = ChatMessageHistory()

        docs = []
        for pdf in uploaded_files:
            loader = PyPDFLoader(pdf)
            pages = loader.load()
            docs.extend(pages)

        splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        st.session_state.vectorstore = Chroma.from_documents(chunks, embeddings)

        st.sidebar.success("✅ PDFs processed and indexed! You can now chat.")


# ---------------------------- MAIN CHAT AREA ----------------------------
st.divider()
st.markdown("### 💬 Chat (continuous conversation)")
user_query = st.chat_input("Ask something about your PDFs...")

if st.session_state.vectorstore is not None and user_query:
    os.environ["OPENAI_API_KEY"] = st.session_state.api_key
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})

    # 1️⃣ History-aware retriever (understands context)
    context_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("human", "Given the above conversation, generate a search query to get relevant context for the current question."),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, context_prompt)

    # 2️⃣ QA prompt — must include {context}
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an assistant for question-answering tasks. "
         "Use the following context to answer the user's question. "
         "If you don't know the answer, just say you don't know.\n\n{context}"
         ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    combine_docs_chain = create_stuff_documents_chain(llm, qa_prompt)

    # 3️⃣ Final retrieval chain
    retrieval_chain = create_retrieval_chain(history_aware_retriever, combine_docs_chain)

    # 4️⃣ Run query
    with st.chat_message("user"):
        st.markdown(user_query)
    result = retrieval_chain.invoke({
        "input": user_query,
        "chat_history": st.session_state.chat_history.messages
    })

    # 5️⃣ Show answer
    answer = result["answer"]
    with st.chat_message("assistant"):
        st.markdown(answer)

    # 6️⃣ Update chat history
    st.session_state.chat_history.add_user_message(user_query)
    st.session_state.chat_history.add_ai_message(answer)

elif user_query and st.session_state.vectorstore is None:
    st.warning("⚠️ Please upload and process PDFs first.")


# ---------------------------- SIDEBAR CHAT HISTORY ----------------------------
if st.session_state.chat_history.messages:
    st.sidebar.markdown("### 🧠 Chat History")
    for msg in st.session_state.chat_history.messages[-10:]:
        role = "👩‍💼 You" if msg.type == "human" else "🤖 Assistant"
        st.sidebar.markdown(f"**{role}:** {msg.content}")
