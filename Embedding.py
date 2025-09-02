from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

model = SentenceTransformer("bkai-foundation-models/vietnamese-bi-encoder")

with open("./documents/luat.txt", "r", encoding="utf-8") as f:
    text = f.read()

def chunk_text(text, chunk_size=200, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

chunks = chunk_text(text, chunk_size=200, overlap=50)

embeddings = model.encode(chunks, convert_to_numpy=True)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

faiss.write_index(index, "./data/luat.index")

with open("./data/luat_meta.pkl", "wb") as f:
    pickle.dump({"chunks": chunks}, f)

print(f"Đã lưu {len(chunks)} chunks vào FAISS index và metadata.")
