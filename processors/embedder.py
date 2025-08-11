# from sentence_transformers import SentenceTransformer
# import faiss
# import numpy as np
# import pickle
# import os

# def create_and_store_embeddings(chunks, db_path="vector_store"):
#     # Use a much faster multilingual embedding model for CPU
#     model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")  

#     # Create embeddings (larger batch size to speed up)
#     print("[INFO] Generating embeddings...")
#     embeddings = model.encode(
#         chunks,
#         show_progress_bar=True,
#         convert_to_numpy=True,
#         batch_size=128
#     )

#     # Create FAISS index (cosine similarity)
#     dimension = embeddings.shape[1]
#     index = faiss.IndexFlatIP(dimension)

#     # Normalize for cosine similarity
#     faiss.normalize_L2(embeddings)

#     # Add vectors to index
#     index.add(embeddings)

#     # Save FAISS index
#     os.makedirs(db_path, exist_ok=True)
#     faiss.write_index(index, os.path.join(db_path, "index.faiss"))

#     # Save chunks alongside for retrieval
#     with open(os.path.join(db_path, "chunks.pkl"), "wb") as f:
#         pickle.dump(chunks, f)

#     print(f"[INFO] Stored {len(chunks)} chunks in vector DB at '{db_path}'")

# if __name__ == "__main__":
#     from processors.file_loader import process_file
#     chunks = process_file("sample_docs/Sample1.pdf")
#     create_and_store_embeddings(chunks)


from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os

def create_and_store_embeddings(chunks, db_path="vector_store"):
    model = SentenceTransformer("intfloat/multilingual-e5-large", device="cpu")

    print("[INFO] Generating embeddings...")
    embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True, batch_size=16)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    os.makedirs(db_path, exist_ok=True)
    faiss.write_index(index, os.path.join(db_path, "index.faiss"))

    with open(os.path.join(db_path, "chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)

    print(f"[INFO] Stored {len(chunks)} chunks in vector DB at '{db_path}'")
