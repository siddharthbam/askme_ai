import os
import io
from flask import Flask, request, jsonify
from flask_cors import CORS
import PyPDF2
from docx import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate

app = Flask(__name__)
CORS(app)

# Ensure your Hugging Face Token has 'Inference' permissions enabled
# Replace your actual token string with this:
hf_token = os.getenv("HF_TOKEN")

vector_db = None
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

template = """Answer the question based ONLY on the following context. If you don't know, say you don't know.
Context: {context}
Question: {question}
Answer:"""
PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])

@app.route('/upload', methods=['POST'])
def upload_file():
    global vector_db
    try:
        file = request.files['file']
        filename = file.filename.lower()
        file_content = ""

        if filename.endswith('.pdf'):
            pdf_reader = PyPDF2.PdfReader(file)
            file_content = "".join([p.extract_text() or "" for p in pdf_reader.pages])
        elif filename.endswith('.docx'):
            doc = Document(io.BytesIO(file.read()))
            file_content = "\n".join([para.text for para in doc.paragraphs])
        elif filename.endswith('.txt'):
            file_content = file.read().decode('utf-8')
        else:
            return jsonify({"error": "Unsupported format"}), 400

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_text(file_content)
        
        if vector_db is None:
            vector_db = FAISS.from_texts(docs, embeddings)
        else:
            vector_db.add_texts(docs)
            
        return jsonify({"message": f"Indexed {file.filename}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/delete_file', methods=['POST'])
def delete_file():
    global vector_db
    # For this local FAISS implementation, deleting resets the session
    vector_db = None 
    return jsonify({"message": "Knowledge base reset."})

@app.route('/ask', methods=['POST'])
def ask():
    global vector_db
    if not vector_db:
        return jsonify({"answer": "Please upload a document first!"})

    try:
        data = request.get_json() or {}
        llm = HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.2",
            huggingfacehub_api_token=hf_token,
            temperature=0.2,
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vector_db.as_retriever(),
            chain_type_kwargs={"prompt": PROMPT},
        )

        response = qa_chain.invoke({"query": data.get("question", "")})
        return jsonify({"answer": response.get("result", str(response))})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)