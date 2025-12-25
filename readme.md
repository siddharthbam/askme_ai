# 🌳 AskMe AI: Intelligent Document Assistant

**AskMe AI** is a full-stack Retrieval-Augmented Generation (RAG) application that allows users to upload documents and have context-aware conversations with their data. 

---

## 🏗 Architecture
The system follows a decoupled **Client-Server** architecture optimized for AI workloads:

* **Frontend**: Built with **React.js**, featuring a chat interface and a custom-designed sidebar for "Knowledge Base" management.
* **Backend**: A **Flask (Python)** server handling file processing (PDF/Docx), document indexing, and AI model orchestration.
* **AI Engine**: Powered by **LangChain** and **Mistral-7B-Instruct-v0.2** (via Hugging Face) for natural language generation.
* **Vector Store**: Uses **FAISS** for fast semantic retrieval of document chunks.



---

## 🧠 Key Decisions & Challenges

### 🛠 Technical Security
* **Environment Variables**: Migrated from hardcoded API tokens to **OS-level environment variables** (`HF_TOKEN`) using `python-dotenv` to follow industry security standards.
* **Performance**: Chose `RecursiveCharacterTextSplitter` to ensure context chunks remain coherent, improving the quality of the AI's answers.

---

## 🚀 How to Run Locally

### 1. Backend Setup
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Create a .env file and add: HF_TOKEN=your_token_here
python app.py
```


### 2. Frontend Setup
```bash
cd frontend
npm install
npm start
```
## 🤖 AI Stack Details

* **Embeddings**: all-MiniLM-L6-v2 (Converts text into 384-dimensional vectors).
* **LLM**: Mistral-7B-Instruct-v0.2 (Synthesizes answers based strictly on provided context).
* **Retrieval**: Leverages the RetrievalQA chain from LangChain to automate the prompt-construction process.
