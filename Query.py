from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import torch
import faiss
import pickle
import numpy as np

model_path = "VietAI/gpt-neo-1.3B-vietnamese-news"

bi_model = SentenceTransformer("bkai-foundation-models/vietnamese-bi-encoder",
                               device='cuda')

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)

index = faiss.read_index("./data/nghidinh.index")
with open("./data/nghidinh_meta.pkl", "rb") as f:
    metadata = pickle.load(f)
chunks = metadata["chunks"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
model.to(device)

def search_faiss(query, top_k=3):
    q_emb = bi_model.encode([query])
    D, I = index.search(np.array(q_emb).astype("float32"), top_k)
    results = [chunks[i] for i in I[0]]
    return results

def ask_sth(query):
    contexts = search_faiss(query, top_k=3)
    context_text = "\n".join(contexts)

    prompt = f"""### Hãy trả lời câu hỏi ngắn gọn và đủ ý với ngữ cảnh:
{context_text}

### Câu hỏi:
{query}

### Trả lời:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    gen_tokens = model.generate(**inputs,
                                max_length=1250,
                                do_sample=True,
                                temperature=0.9,
                                top_k=20,
                            )
    answer = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
    answer = answer.split("### Trả lời:")[-1].strip()
    return answer
