from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "NKhanh/NK106_bid_law_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

prompt = "điều nghiêm cấm trong đấu thầu."

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        top_p=0.9,
        temperature=0.7
    )

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("=== Output ===")
print(result)
