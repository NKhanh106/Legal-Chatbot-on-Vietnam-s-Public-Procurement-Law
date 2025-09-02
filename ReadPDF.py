import pytesseract
from pdf2image import convert_from_path
import re

#pdf_path = "./documents/214_2025_ND-CP_04082025.pdf"
pdf_path = r"./documents/Luat DT so 22.2023.QH15.pdf"
output_path = "./documents/luat.txt"

pages = convert_from_path(pdf_path)

full_text = ""

for i, page in enumerate(pages):
    text = pytesseract.image_to_string(page, lang="vie") 
    text = re.sub(r'\-\d+\-', '', text)
    full_text += f"\n=== Trang {i+1} ===\n{text}"

with open(output_path, "w", encoding="utf-8") as f:
    f.write(full_text)

print("✅ OCR hoàn tất! Văn bản đã được lưu vào", output_path)
