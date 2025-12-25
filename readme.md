# 🌳 AskMe AI: Intelligent Document Assistant

AskMe AI is a full-stack **Retrieval-Augmented Generation (RAG)** application. It allows users to upload documents (PDF/Docx) and have context-aware, real-time conversations with their data using local vector embeddings and cloud-based LLMs.

---

## 🏗 Architecture

The system follows a decoupled **Client-Server** architecture optimized for AI workloads:

* **Frontend**: Built with **React.js**, featuring a chat interface and a custom-designed sidebar for "Knowledge Base" management.
* **Backend**: A **Flask (Python)** server handling file processing (PDF/Docx), document indexing, and AI model orchestration.
* **AI Engine**: Powered by **LangChain** and **Mistral-7B-Instruct-v0.2** (via Hugging Face) for natural language generation.
* **Vector Store**: Uses **FAISS** (Facebook AI Similarity Search) for high-speed semantic retrieval of document chunks.



---

## 🧠 Key Decisions & Challenges

### 🛠 Technical Security
* **Environment Variables**: Migrated from hardcoded API tokens to **OS-level variables** using `python-dotenv` to meet industry security standards.
* **Performance**: Utilized `RecursiveCharacterTextSplitter` to ensure document chunks remain coherent, which significantly improved the AI's accuracy.

---

## 🚀 How to Run Locally

### 1. Backend Configuration
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Important: Add your HF_TOKEN to a .env file in this folder
python app.py
```

### 2. Frontend Configuration
```bash
cd frontend
npm install
npm start
```