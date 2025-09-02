import re

input_file = "./data/luat.txt"

output_file = "./data/luat.txt"

with open(input_file, "r", encoding="utf-8") as f:
    text = f.read()

text = re.sub(r"=+\s*Trang\s*\d+\s*=+", "", text)

text = re.sub(r"\n\s*\n+", "\n\n", text)

lines = text.split("\n")
new_lines = []
buffer = ""

for line in lines:
    line = line.strip()
    if not line:
        if buffer:
            new_lines.append(buffer)
            buffer = ""
        new_lines.append("")
        continue
    
    if buffer:
        if not re.search(r"[.?!:…]$", buffer):
            buffer += " " + line
        else:
            new_lines.append(buffer)
            buffer = line
    else:
        buffer = line

if buffer:
    new_lines.append(buffer)

clean_text = "\n".join(new_lines)

with open(output_file, "w", encoding="utf-8") as f:
    f.write(text)

print("✅ Đã xử lý xong, lưu tại:", output_file)
