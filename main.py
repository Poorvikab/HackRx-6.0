from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import shutil
import os
import pickle
import faiss
import re
import requests
import torch
from sentence_transformers import SentenceTransformer
from processors.file_loader import process_file
from dotenv import load_dotenv
from typing import List
import hashlib
import time
import json

# ---------------- load env ----------------
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
LLM_MODEL = "mistral-medium"  # change to the specific Mistral model if needed

print(f"Using Mistral model: {LLM_MODEL}")

# ---------------- config ----------------
UPLOAD_DIR = "sample_docs"
DB_PATH = "vector_store"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DB_PATH, exist_ok=True)

app = FastAPI(title="RAG Query-Retrieval API")

# ---------- embedding model -----------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device for embeddings: {device}")

embedding_model = SentenceTransformer(
    "intfloat/multilingual-e5-large",
    device=device
)

# Store active file hash globally
active_file_hash = None

# ---------------- Pydantic request ----------------
class QueryRequest(BaseModel):
    question: str
    top_k: int = 5

# ------------------ helpers ------------------
_sentence_split_re = re.compile(r'(?<=[.!?])\s+')

def _shorten_to_sentences(text: str, max_tokens: int = 120) -> str:
    max_chars = max_tokens * 4
    parts = _sentence_split_re.split(text.strip())

    result = []
    char_count = 0

    for sentence in parts:
        sentence = sentence.strip()
        if not sentence:
            continue
        if char_count + len(sentence) + 1 > max_chars:
            break
        result.append(sentence)
        char_count += len(sentence) + 1

    return " ".join(result).strip()

def file_hash(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, 'rb') as f:
        hasher.update(f.read())
    return hasher.hexdigest()

def get_index_paths(file_hash_val: str):
    return (
        os.path.join(DB_PATH, f"index_{file_hash_val}.faiss"),
        os.path.join(DB_PATH, f"chunks_{file_hash_val}.pkl")
    )

def create_and_store_embeddings(chunks, db_path, file_hash_val):
    embeddings = embedding_model.encode(chunks, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)
    
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    
    idx_path, chunks_path = get_index_paths(file_hash_val)
    faiss.write_index(index, idx_path)
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)

def search_faiss(query: str, top_k: int = 5, file_hash_val: str = None) -> List[str]:
    if file_hash_val is None:
        global active_file_hash
        if active_file_hash is None:
            raise FileNotFoundError("No document uploaded yet. Upload a document first via /upload.")
        file_hash_val = active_file_hash

    idx_path, chunks_path = get_index_paths(file_hash_val)
    if not os.path.exists(idx_path) or not os.path.exists(chunks_path):
        raise FileNotFoundError("No index found. Upload a document first via /upload.")

    index = faiss.read_index(idx_path)
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)

    q_emb = embedding_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)

    D, I = index.search(q_emb, top_k)
    results = []
    for idx in I[0]:
        if idx >= 0 and idx < len(chunks):
            results.append(chunks[idx])
    return results

def get_dynamic_max_tokens(prompt: str, max_limit: int = 250) -> int:
    estimated_prompt_tokens = max(len(prompt) // 4, 1)
    max_tokens = min(int(estimated_prompt_tokens * 0.4), max_limit)
    return max(50, max_tokens)

# UPDATED: Call Mistral API
def call_mistral_inference(prompt: str) -> str:
    if not MISTRAL_API_KEY:
        return "[ERROR] MISTRAL_API_KEY not configured. Add it to .env."

    max_tokens = get_dynamic_max_tokens(prompt)
    prompt_with_limit = prompt + f"\n\nPlease keep your answer concise and within {max_tokens} tokens."

    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt_with_limit}],
        "max_tokens": max_tokens,
        "temperature": 0.4
    }

    try:
        r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
        r.raise_for_status()
        response = r.json()
        gen_text = response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[ERROR] Mistral API request failed: {str(e)}"

    short = _shorten_to_sentences(gen_text, max_tokens=max_tokens)
    return short

def estimate_accuracy(answer: str, question: str) -> float:
    keywords = question.lower().split()
    answer_lower = answer.lower()
    if not keywords:
        return 0.0
    hits = sum(1 for kw in keywords if kw in answer_lower)
    return min(1.0, hits / len(keywords))

# ------------------ endpoints ------------------
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global active_file_hash
    start_time = time.time()
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    file_hash_val = file_hash(file_path)
    idx_path, chunks_path = get_index_paths(file_hash_val)

    if os.path.exists(idx_path) and os.path.exists(chunks_path):
        elapsed_time = time.time() - start_time
        active_file_hash = file_hash_val
        return {
            "status": "success",
            "message": "File embeddings already exist. Skipped processing.",
            "processing_time_seconds": round(elapsed_time, 2),
            "file_hash": file_hash_val
        }

    chunks = process_file(file_path)
    create_and_store_embeddings(chunks, db_path=DB_PATH, file_hash_val=file_hash_val)

    elapsed_time = time.time() - start_time
    active_file_hash = file_hash_val

    return {
        "status": "success",
        "message": f"Processed {len(chunks)} chunks.",
        "processing_time_seconds": round(elapsed_time, 2),
        "file_hash": file_hash_val
    }

@app.post("/query")
async def query_document(req: QueryRequest):
    start_time = time.time()

    question = req.question
    top_k = req.top_k if req.top_k and req.top_k > 0 else 5

    try:
        relevant_chunks = search_faiss(question, top_k=top_k)
    except FileNotFoundError as e:
        return {"error": str(e)}

    truncated_chunks = []
    for i, c in enumerate(relevant_chunks):
        if not isinstance(c, str):
            continue
        trimmed = c.strip()
        if len(trimmed) > 700:
            trimmed = trimmed[:700].rsplit(" ", 1)[0] + " ...[truncated]"
        truncated_chunks.append(f"CHUNK[{i}]: {trimmed}")

    context = "\n\n---\n\n".join(truncated_chunks)

    prompt = f"""
You are a helpful and friendly document assistant. Please answer the question in a clear, natural, and slightly detailed way,
using a conversational tone. Provide just enough explanation to make the answer informative and easy to understand,
but keep it concise and relevant.

Use ONLY the document context below — do not invent facts. If the answer is not present, say: "The document does not provide this information."

Document context:
{context}

Question: {question}

Answer:
"""

    answer = call_mistral_inference(prompt)
    elapsed_time = time.time() - start_time
    accuracy = estimate_accuracy(answer, question)

    return {
        "elapsed_time_seconds": round(elapsed_time, 2),
        "accuracy_estimate": round(accuracy, 2),
        "question": question,
        "answer": answer,
        "sources": truncated_chunks
    }

@app.get("/mistral_health")
def mistral_health():
    if not MISTRAL_API_KEY:
        return {"ok": False, "error": "MISTRAL_API_KEY not set"}
    return {"ok": True, "model": LLM_MODEL}

@app.get("/")
def root():
    return {"message": "RAG Query-Retrieval API running with Mistral"}
