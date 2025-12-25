# 🌳 AskMe AI: Intelligent Document Assistant

AskMe AI is a full-stack **Retrieval-Augmented Generation (RAG)** application. It allows users to upload documents (PDF/Docx) and have context-aware, real-time conversations with their data using local vector embeddings and cloud-based LLMs.

---

## 🏗 Architecture

The system follows a decoupled **Client-Server** architecture optimized for AI workloads:

<<<<<<< HEAD
* **Frontend**: Built with **React.js**, featuring a "Gemini-inspired" chat interface and a custom-designed sidebar for knowledge base management.
* **Backend**: A **Flask (Python)** server handling file processing, text extraction, and AI model orchestration.
=======
* **Frontend**: Built with **React.js**, featuring a chat interface and a custom-designed sidebar for "Knowledge Base" management.
* **Backend**: A **Flask (Python)** server handling file processing (PDF/Docx), document indexing, and AI model orchestration.
>>>>>>> 22ac31d4d7f9ac9fd0883670829d647e56c7506c
* **AI Engine**: Powered by **LangChain** and **Mistral-7B-Instruct-v0.2** (via Hugging Face) for natural language generation.
* **Vector Store**: Uses **FAISS** (Facebook AI Similarity Search) for high-speed semantic retrieval of document chunks.



---

## 🧠 Key Decisions & Challenges

<<<<<<< HEAD
### 🎨 Design & Proportionality
* **Visual Balance**: Refined CSS margins to group the **Logo and Title** closer together for a more cohesive visual identity.
* **Branding**: Implemented a warm, off-white sidebar (**#f6f6ea**) to provide a sophisticated "paper-like" feel that contrasts with the clean white chat area.

=======
>>>>>>> 22ac31d4d7f9ac9fd0883670829d647e56c7506c
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

<<<<<<< HEAD
### 2. Frontend Configuration
=======

### 2. Frontend Setup
>>>>>>> 22ac31d4d7f9ac9fd0883670829d647e56c7506c
```bash
cd frontend
npm install
npm start
```
<<<<<<< HEAD

---

## 🤖 AI Stack Details

* **Embeddings**: all-MiniLM-L6-v2 (Converts text into 384-dimensional vectors).
* **LLM**: Mistral-7B-Instruct-v0.2 (Synthesizes answers based strictly on provided context).
* **Retrieval**: Leverages the Retrieval QA chain from LangChain to automate the prompt-construction process.

=======
## 🤖 AI Stack Details

* **Embeddings**: all-MiniLM-L6-v2 (Converts text into 384-dimensional vectors).
* **LLM**: Mistral-7B-Instruct-v0.2 (Synthesizes answers based strictly on provided context).
* **Retrieval**: Leverages the RetrievalQA chain from LangChain to automate the prompt-construction process.
>>>>>>> 22ac31d4d7f9ac9fd0883670829d647e56c7506c
