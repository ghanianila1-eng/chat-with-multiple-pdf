🧠 Chat with Your PDF — AI-Powered Document Assistant

Interact with your PDF documents using natural language!
This project uses LangChain, OpenAI, and Streamlit to let you ask questions directly from your PDF — no manual reading required.

🚀 Demo Preview

💡 Upload your PDF, ask questions, and get instant answers powered by GPT!

✨ Features

✅ Conversational Memory – Maintains context between multiple questions
✅ PDF Understanding – Reads and interprets PDF content
✅ Smart Chunking – Splits large documents for efficient retrieval
✅ Modern UI – Beautiful chat interface built with Streamlit
✅ Private and Secure – Uses your own OpenAI API key

🧩 Tech Stack
Component	Technology
UI	Streamlit
AI Framework	LangChain
Model Provider	OpenAI (GPT-3.5 / GPT-4)
Vector Database	ChromaDB
Document Processing	PyPDFLoader
Memory	ChatMessageHistory
⚙️ Setup Instructions
1️⃣ Clone this Repository
git clone https://github.com/yourusername/chat-with-your-pdf.git
cd chat-with-your-pdf

2️⃣ Create a Virtual Environment
python -m venv venv
source venv/bin/activate    # (on Mac/Linux)
venv\Scripts\activate       # (on Windows)

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Add Your OpenAI API Key

You can get your key here
.

Then set it in your terminal:

export OPENAI_API_KEY="your_api_key_here"   # (Mac/Linux)
setx OPENAI_API_KEY "your_api_key_here"     # (Windows)


or you can paste it directly in the app sidebar when running Streamlit.

🖥️ Run the App
streamlit run app.py


Then open your browser at:

http://localhost:8501

📁 Project Structure
📦 chat-with-your-pdf
 ┣ 📜 app.py                # Main Streamlit app
 ┣ 📜 requirements.txt      # Dependencies
 ┣ 📜 README.md             # Project documentation
 ┗ 📂 data/ (optional)      # PDFs or sample data

🧠 How It Works

Upload your PDF document

The text is split into chunks using LangChain’s text splitter

Each chunk is embedded into a vector space using OpenAI embeddings

A retriever finds the most relevant chunks for your question

GPT generates a context-aware answer from the relevant sections

🖼️ UI Preview

A modern, professional chatbot interface:

💬 Clean chat bubbles

🎨 Gradient background

⚡ Real-time responses

🧹 Clear chat history button

🤝 Contributing

Pull requests are welcome!
If you’d like to add features like:

PDF summarization

Multi-file chat

Custom theme (Anila AI branding 🌸)

just fork the repo and open a PR.

🧾 License

This project is open-source under the MIT License.
Feel free to use and modify it for your own projects.

💖 Credits

Built with 💡 by Anila Ghani

“AI that understands your documents.”
