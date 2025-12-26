import streamlit as st
import os
import PyPDF2
from docx import Document
from dotenv import load_dotenv

# ---------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
load_dotenv()
st.set_page_config(page_title="Ask Llama 3.1", page_icon="🦙")

# Llama 3 specific prompt template (Standard Format)
template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant. Answer the user's question based ONLY on the context provided below. If you don't know, say you don't know.<|eot_id|><|start_header_id|>user<|end_header_id|>

Context:
{context}

Question:
{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])

# ---------------------------------------------------------
# STATE MANAGEMENT
# ---------------------------------------------------------
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------------
# SIDEBAR: DOCUMENT LOADING
# ---------------------------------------------------------
with st.sidebar:
    st.header("📁 Document Manager")
    uploaded_file = st.file_uploader("Upload PDF, DOCX, or TXT", type=["pdf", "docx", "txt"])
    
    if uploaded_file is not None:
        if st.button("Process File"):
            with st.spinner("Indexing document..."):
                try:
                    # 1. Extract Text
                    file_content = ""
                    if uploaded_file.name.endswith(".pdf"):
                        pdf_reader = PyPDF2.PdfReader(uploaded_file)
                        file_content = "".join([p.extract_text() or "" for p in pdf_reader.pages])
                    elif uploaded_file.name.endswith(".docx"):
                        doc = Document(uploaded_file)
                        file_content = "\n".join([para.text for para in doc.paragraphs])
                    elif uploaded_file.name.endswith(".txt"):
                        file_content = uploaded_file.read().decode("utf-8")
                    
                    # 2. Split Text
                    # We use 500 characters to be safe on the Free Tier bandwidth
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=500, 
                        chunk_overlap=50,
                        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
                    )
                    docs = text_splitter.split_text(file_content)
                    
                    # 3. Create Vector Store
                    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                    
                    if st.session_state.vector_db is None:
                        st.session_state.vector_db = FAISS.from_texts(docs, embeddings)
                    else:
                        st.session_state.vector_db.add_texts(docs)
                        
                    st.success(f"Successfully indexed {uploaded_file.name}!")
                    
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")

    st.markdown("---")
    if st.button("🗑️ Reset Knowledge Base"):
        st.session_state.vector_db = None
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------
# MAIN CHAT INTERFACE
# ---------------------------------------------------------
st.title("🦙 Ask Llama 3.1")

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle new user input
if prompt := st.chat_input("Ask a question about your documents..."):
    # 1. Display User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Generate Assistant Response
    if st.session_state.vector_db is None:
        st.error("Please upload and process a document first!")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Llama is thinking..."):
                try:
                    hf_token = os.getenv("HF_TOKEN")
                    
                    # --- LLAMA 3.1 CONFIGURATION ---
                    llm = HuggingFaceEndpoint(
                        repo_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
                        huggingfacehub_api_token=hf_token,
                        task="text-generation",
                        
                        # Top-level parameters
                        temperature=0.1,
                        max_new_tokens=512,
                        do_sample=False,
                        repetition_penalty=1.1
                    )
                    # -------------------------------

                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        # We retrieve top 3 chunks for better context
                        retriever=st.session_state.vector_db.as_retriever(search_kwargs={"k": 3}),
                        chain_type_kwargs={"prompt": PROMPT},
                    )

                    response = qa_chain.invoke({"query": prompt})
                    answer = response["result"]
                    
                    # Clean up the response (Llama sometimes repeats the prompt)
                    if "<|start_header_id|>assistant<|end_header_id|>" in answer:
                        answer = answer.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
                    
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.warning("If you see a '403 Client Error', make sure you accepted the terms on the Llama 3.1 Hugging Face page!")