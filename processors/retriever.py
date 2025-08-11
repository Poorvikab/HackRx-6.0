# processors/retriever.py
import os
import pickle
import json
import requests
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import faiss

# model used locally for query embeddings (fast CPU-friendly)
EMBED_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"  # or use all-MiniLM-L6-v2
# HF reasoning model (change to a model you have access to)
HF_REASONER = "mistralai/Mixtral-8x7B-Instruct-v0.1"
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

# cached objects (keeps memory warm between requests)
_sentence_model = None
_faiss_index = None
_chunks = None

def _load_sentence_model():
    global _sentence_model
    if _sentence_model is None:
        _sentence_model = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")
    return _sentence_model

def load_index_and_chunks(db_path="vector_store"):
    """
    Loads FAISS index and chunks.pkl. Raises FileNotFoundError if not present.
    """
    global _faiss_index, _chunks
    idx_path = os.path.join(db_path, "index.faiss")
    chunks_path = os.path.join(db_path, "chunks.pkl")
    if _faiss_index is None or _chunks is None:
        if not os.path.exists(idx_path) or not os.path.exists(chunks_path):
            raise FileNotFoundError("Index or chunks not found. Upload and index a document first.")
        _faiss_index = faiss.read_index(idx_path)
        with open(chunks_path, "rb") as f:
            _chunks = pickle.load(f)
    return _faiss_index, _chunks

def semantic_search(query: str, top_k: int = 5, db_path="vector_store") -> List[Dict]:
    """
    Returns top_k matches: list of dicts with chunk_id, score, text.
    """
    model = _load_sentence_model()
    index, chunks = load_index_and_chunks(db_path)
    q_emb = model.encode([query], convert_to_numpy=True)
    # normalize since IndexFlatIP was built with normalized vectors
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, top_k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        results.append({
            "chunk_id": int(idx),
            "score": float(score),
            "text": chunks[idx]
        })
    return results

def build_reasoner_prompt(question: str, matches: List[Dict], max_chars_per_chunk: int = 1200) -> str:
    """
    Create a concise prompt for the reasoning LLM. Truncates chunks for token efficiency.
    """
    chunk_blocks = []
    for m in matches:
        text = m["text"]
        if len(text) > max_chars_per_chunk:
            text = text[:max_chars_per_chunk].rsplit(" ", 1)[0] + " ...[truncated]"
        chunk_blocks.append(f"CHUNK_ID:{m['chunk_id']} SCORE:{m['score']:.4f}\n{text}")
    chunks_combined = "\n\n---\n\n".join(chunk_blocks)

    prompt = f"""
You are an expert policy / claims analyst. The user asked: "{question}"

Below are the most relevant document chunks (each labelled CHUNK_ID:<id>). Use only the information present here to decide.
Do 3 things:
1) Parse the user's query into structured fields (age, gender, procedure, location, policy_duration, any other relevant fields).
2) Based on the chunks, determine whether the procedure is covered (approved / rejected / needs_more_info). If a payout or limit applies, state the amount or formula.
3) Return a concise justification and list the matched chunk ids used.

Return only valid JSON (no explanation outside JSON) matching this schema exactly:

{
  "decision": "approved" | "rejected" | "needs_more_info",
  "amount": "<number or null>",
  "justification": "<1-3 sentence justification referencing clause ids>",
  "matched_clauses": [
      { "chunk_id": 12, "score": 0.93, "text_excerpt": "..." }
  ],
  "structured_query": {
      "age": 46,
      "gender": "male",
      "procedure": "knee surgery",
      "location": "Pune",
      "policy_duration": "3 months"
  }
}

Document chunks:
{chunks_combined}

Remember: be conservative, cite the chunk_id(s) you used, and do not hallucinate extra rules.
"""
    return prompt

def call_hf_reasoner(prompt: str, model_name: str = HF_REASONER, max_tokens: int = 512) -> Dict:
    """
    Call Hugging Face Inference API and try to parse JSON from the model output.
    Falls back to a simple extracted response if JSON parsing fails.
    """
    if HF_API_TOKEN is None:
        return {
            "decision": "needs_more_info",
            "amount": None,
            "justification": "No HF_API_TOKEN configured; reasoning not performed.",
            "matched_clauses": [],
            "structured_query": {}
        }

    url = f"https://api-inference.huggingface.co/models/{model_name}"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": max_tokens, "temperature": 0.0}}

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        # HuggingFace text response formats vary by model; try to find generated text
        if isinstance(data, list) and len(data) > 0:
            gen = data[0].get("generated_text") or data[0].get("text") or data[0].get("generated_text")
            if gen is None and isinstance(data[0], dict):
                # some models return {'generated_text': '...'}
                gen = data[0].get("generated_text") or next(iter(data[0].values()))
        elif isinstance(data, dict):
            gen = data.get("generated_text") or data.get("text") or json.dumps(data)
        else:
            gen = str(data)
    except Exception as e:
        # HF call failed
        return {
            "decision": "needs_more_info",
            "amount": None,
            "justification": f"Hugging Face call failed: {str(e)}",
            "matched_clauses": [],
            "structured_query": {}
        }

    # Try to extract JSON from generated text
    gen_text = gen if isinstance(gen, str) else str(gen)
    # Some models prefix with newlines etc. Extract first JSON-like substring
    try:
        # naive way: find first '{' and last '}' and parse
        start = gen_text.find('{')
        end = gen_text.rfind('}')
        json_text = gen_text[start:end+1] if start != -1 and end != -1 else gen_text
        parsed = json.loads(json_text)
        return parsed
    except Exception:
        # fallback: return raw text in justification
        return {
            "decision": "needs_more_info",
            "amount": None,
            "justification": f"Raw model output (could not parse JSON): {gen_text[:600]}",
            "matched_clauses": [],
            "structured_query": {}
        }
