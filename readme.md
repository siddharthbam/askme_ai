🌳 AskMe AI: Intelligent Document Assistant
AskMe AI is a full-stack Retrieval-Augmented Generation (RAG) application that allows users to upload documents and have context-aware conversations with their data.

🏗 Architecture
The system follows a decoupled Client-Server architecture optimized for AI workloads:

Frontend: Built with React.js, featuring a "Gemini-inspired" chat interface and a custom-designed sidebar for "Knowledge Base" management.

Backend: A Flask (Python) server that handles file processing, document indexing, and AI model orchestration.

AI Engine: Powered by LangChain for the RAG pipeline and Mistral-7B (via Hugging Face) for natural language generation.

Vector Store: Uses FAISS for fast semantic retrieval.

🧠 Key Decisions & Challenges
🎨 Design & Proportionality

Visual Balance: Refined CSS margins to group the "Brand Unit" (Logo + Title) closer together (8px gap) for a more cohesive visual identity.

Sidebar Branding: Scaled the logo to 110px to establish brand authority without overwhelming the navigation.

User Experience: Implemented a warm, off-white sidebar (#f6f6ea) to provide a sophisticated "paper-like" feel.

🛠 Technical Security

GitHub Push Protection: Encountered and resolved a security block regarding hardcoded API tokens.

Solution: Migrated to Environment Variables (os.getenv("HF_TOKEN")) and established a .gitignore workflow to ensure sensitive credentials never reach the public repository.

🚀 How to Run Locally
1. Backend Setup

Bash
# Navigate to backend folder
cd backend

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create a .env file and add your token
echo "HF_TOKEN=your_huggingface_token_here" > .env

# Start the Flask server
python app.py
2. Frontend Setup

Bash
# Open a new terminal tab and navigate to frontend
cd frontend

# Install packages
npm install

# Start the React app
npm start
🤖 AI Usage
Embeddings: all-MiniLM-L6-v2 converts text into vectors.

LLM: Mistral-7B-Instruct-v0.2 synthesizes answers based only on provided context to prevent hallucinations.

Retrieval: RetrievalQA chain from LangChain manages the link between FAISS and the LLM.