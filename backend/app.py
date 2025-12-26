import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
import os
import tempfile

# 1. Page Config
st.set_page_config(page_title="Ask the Doc", layout="wide")
st.title("🤖 Ask the Doc (Flan-T5 Edition)")

# 2. Sidebar for Secrets
with st.sidebar:
    st.header("Setup")
    hf_token = st.text_input("Hugging Face Token", type="password")
    if not hf_token and "HF_TOKEN" in st.secrets:
        hf_token = st.secrets["HF_TOKEN"]
    
    st.markdown("---")
    st.markdown("### 💡 Why this model?")
    st.info("We are using 'Flan-T5' because it is stable on the free tier and reads documents without 'Task Not Supported' errors.")

# 3. File Uploader
uploaded_file = st.file_uploader("Upload a PDF or Word Doc", type=["pdf", "docx"])

if uploaded_file and hf_token:
    try:
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        # Load Document
        with st.spinner("Reading file..."):
            if uploaded_file.name.endswith(".pdf"):
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
            else:
                loader = Docx2txtLoader(tmp_path)
                docs = loader.load()
            
            # CRITICAL FIX: Smaller chunks to prevent "Context Length" crashes
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=250,  # Tiny chunks to fit in memory
                chunk_overlap=50
            )
            splits = text_splitter.split_documents(docs)

            # Create Vector DB
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectorstore = FAISS.from_documents(splits, embeddings)
            
            # Setup LLM - Google Flan-T5 Large (Stable & Free)
            llm = HuggingFaceEndpoint(
                repo_id="google/flan-t5-large",
                huggingfacehub_api_token=hf_token,
                temperature=0.1,
                max_new_tokens=200
            )

            # Create Chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=vectorstore.as_retriever(search_kwargs={"k": 2}) # Only get 2 best chunks
            )

            st.success("✅ File processed! Ask away.")
            
            # Chat Interface
            if "messages" not in st.session_state:
                st.session_state.messages = []

            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if prompt := st.chat_input("Ask a question about your document..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = qa_chain.invoke({"query": prompt})
                        st.markdown(response["result"])
                        st.session_state.messages.append({"role": "assistant", "content": response["result"]})

    except Exception as e:
        st.error(f"An error occurred: {e}")

elif not hf_token:
    st.warning("Please enter your Hugging Face Token to continue.")