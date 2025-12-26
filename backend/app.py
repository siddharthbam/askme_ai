import os
import io
from typing import List, Optional

import numpy as np
import faiss
import PyPDF2
from docx import Document as DocxDocument
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from huggingface_hub import InferenceClient

# -----------------------
# Config
# -----------------------
HF_TOKEN = os.getenv("HF_TOKEN")
LLM_MODEL = os.getenv("LLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "200"))
OVERLAP = int(os.getenv("OVERLAP", "50"))

# -----------------------
# App
# -----------------------
app = FastAPI(title="AskMe AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# In-memory stores
# -----------------------
embedder = SentenceTransformer(EMBED_MODEL, device="cpu")
dim = embedder.get_sentence_embedding_dimension()
index = faiss.IndexFlatIP(dim)

chunks: List[str] = []
sources: List[str] = []

hf = InferenceClient(token=HF_TOKEN)

PROMPT_TEMPLATE = """Answer the question based ONLY on the following context. If you don't know, say you don't know.

Context:
{context}

Question: {question}

Answer:"""

def normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1e-9
    return x / n

def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=OVERLAP) -> List[str]:
    words = text.split()
    out = []
    start = 0
    while start < len(words):
        out.append(" ".join(words[start:start + chunk_size]))
        start += max(1, chunk_size - overlap)
    return [c for c in out if c.strip()]

def read_pdf(data: bytes) -> str:
    reader = PyPDF2.PdfReader(io.BytesIO(data))
    return "\n".join([(p.extract_text() or "") for p in reader.pages])

def read_docx(data: bytes) -> str:
    doc = DocxDocument(io.BytesIO(data))
    return "\n".join([p.text for p in doc.paragraphs])

def read_txt(data: bytes) -> str:
    return data.decode("utf-8", errors="ignore")

def add_chunks(text_chunks: List[str], source: str):
    if not text_chunks:
        return

    embs = embedder.encode(text_chunks, convert_to_numpy=True, show_progress_bar=False)
    embs = normalize(embs).astype("float32")

    index.add(embs)
    chunks.extend(text_chunks)
    sources.extend([source] * len(text_chunks))

def retrieve(question: str, k: int):
    if len(chunks) == 0:
        return [], []

    q = embedder.encode([question], convert_to_numpy=True, show_progress_bar=False)
    q = normalize(q).astype("float32")

    D, I = index.search(q, k)
    ctx = []
    evid = []
    for score, idx in zip(D[0], I[0]):
        if 0 <= idx < len(chunks):
            ctx.append(chunks[idx])
            evid.append({"source": sources[idx], "score": float(score), "chunk": chunks[idx]})
    return ctx, evid

def answer_with_llm(question: str, context_chunks: List[str]) -> str:
    if not HF_TOKEN:
        return "HF_TOKEN missing on backend. Add it to environment variables/secrets."

    context = "\n\n---\n\n".join(context_chunks[:4])
    if len(context) > 12000:
        context = context[:12000] + "\n\n[Truncated]"

    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    resp = hf.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=350,
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

# -----------------------
# API
# -----------------------
@app.get("/health")
def health():
    return {"ok": True, "chunks": len(chunks)}

@app.post("/delete_file")
def delete_file():
    global index, chunks, sources
    index = faiss.IndexFlatIP(dim)
    chunks = []
    sources = []
    return {"message": "Knowledge base reset."}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    data = await file.read()
    name = (file.filename or "uploaded").lower()

    if name.endswith(".pdf"):
        text = read_pdf(data)
    elif name.endswith(".docx"):
        text = read_docx(data)
    elif name.endswith(".txt"):
        text = read_txt(data)
    else:
        return {"error": "Unsupported format"}

    parts = chunk_text(text)
    add_chunks(parts, file.filename or "uploaded")
    return {"message": f"Indexed {file.filename}", "added_chunks": len(parts), "total_chunks": len(chunks)}

class AskBody(BaseModel):
    question: str
    top_k: Optional[int] = 4

@app.post("/ask")
def ask(body: AskBody):
    if len(chunks) == 0:
        return {"answer": "Please upload a document first!"}

    ctx, evid = retrieve(body.question, body.top_k or 4)
    if not ctx:
        return {"answer": "No relevant context found in the documents.", "evidence": []}

    ans = answer_with_llm(body.question, ctx)
    return {"answer": ans, "evidence": evid}