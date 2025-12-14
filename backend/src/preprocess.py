import re
import os
from pathlib import Path
from typing import List, Tuple, Dict

# Get the project root directory (parent of backend/src)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATA_TEXT_DIR = os.path.join(DATA_DIR, "text")  # Thư mục lưu file .txt sau khi preprocess
DOCUMENTS_DIR = os.path.join(PROJECT_ROOT, "documents")
MARKDOWN_DIR = os.path.join(DOCUMENTS_DIR, "markdown")  # Thư mục chứa file markdown từ Word

# Tạo thư mục nếu chưa có
os.makedirs(DATA_TEXT_DIR, exist_ok=True)
os.makedirs(MARKDOWN_DIR, exist_ok=True)

def clean_markdown(text: str) -> str:
    """
    Làm sạch markdown, loại bỏ các phần không cần thiết.
    
    Args:
        text: Markdown text cần xử lý
    
    Returns:
        Markdown đã được làm sạch
    """
    # Loại bỏ số trang và header/footer
    text = re.sub(r"=+\s*Trang\s*\d+\s*=+", "", text)
    text = re.sub(r"-\s*\d+\s*-", "", text)  # Loại bỏ số trang dạng -1-, -2-
    
    # Loại bỏ các markdown artifacts không cần thiết
    text = re.sub(r"```[\s\S]*?```", "", text)  # Loại bỏ code blocks
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)  # Loại bỏ images
    
    # Chuẩn hóa khoảng trắng
    text = re.sub(r"\n{3,}", "\n\n", text)  # Tối đa 2 dòng trống
    text = re.sub(r"[ \t]+", " ", text)  # Chuẩn hóa spaces
    
    return text.strip()

def parse_markdown_structure(text: str) -> List[Dict]:
    """
    Parse markdown để tách thành các section có cấu trúc.
    
    Args:
        text: Markdown text
    
    Returns:
        List các section với metadata
    """
    lines = text.split("\n")
    sections = []
    current_section = {
        "level": 0,
        "type": "paragraph",
        "content": [],
        "header": None
    }
    
    for line in lines:
        line = line.strip()
        
        if not line:
            if current_section["content"]:
                sections.append(current_section)
                current_section = {
                    "level": current_section["level"],
                    "type": "paragraph",
                    "content": [],
                    "header": current_section["header"]
                }
            continue
        
        # Phát hiện markdown headers (# ## ###)
        header_match = re.match(r"^(#{1,6})\s+(.+)$", line)
        if header_match:
            # Lưu section cũ
            if current_section["content"]:
                sections.append(current_section)
            
            # Tạo section mới
            level = len(header_match.group(1))
            header_text = header_match.group(2).strip()
            
            current_section = {
                "level": level,
                "type": "header",
                "content": [],
                "header": header_text
            }
            continue
        
        # Phát hiện list items
        list_match = re.match(r"^[-*+]\s+(.+)$", line)
        ordered_list_match = re.match(r"^\d+\.\s+(.+)$", line)
        
        if list_match or ordered_list_match:
            if current_section["type"] != "list":
                if current_section["content"]:
                    sections.append(current_section)
                current_section = {
                    "level": current_section["level"],
                    "type": "list",
                    "content": [],
                    "header": current_section["header"]
                }
            content = list_match.group(1) if list_match else ordered_list_match.group(1)
            current_section["content"].append(content)
            continue
        
        # Phát hiện cấu trúc pháp luật (Điều, Khoản, Điểm)
        legal_match = re.match(r"^(Điều|Khoản|Điểm|Chương|Mục|Phần)\s+(\d+[a-z]?)[\.:]?\s*(.+)?$", line, re.IGNORECASE)
        if legal_match:
            if current_section["content"]:
                sections.append(current_section)
            
            legal_type = legal_match.group(1)
            legal_number = legal_match.group(2)
            legal_title = legal_match.group(3) if legal_match.group(3) else ""
            
            current_section = {
                "level": 2 if legal_type in ["Điều", "Chương", "Phần"] else 3,
                "type": "legal_structure",
                "content": [],
                "header": f"{legal_type} {legal_number}",
                "legal_type": legal_type,
                "legal_number": legal_number,
                "legal_title": legal_title.strip()
            }
            continue
        
        # Thêm vào content của section hiện tại
        current_section["content"].append(line)
    
    # Thêm section cuối cùng
    if current_section["content"]:
        sections.append(current_section)
    
    return sections

def restructure_markdown(text: str) -> str:
    """
    Tái cấu trúc markdown để phù hợp với embeddings.
    Tận dụng cấu trúc markdown để tạo các chunk có ý nghĩa.
    
    Args:
        text: Markdown text đã được làm sạch
    
    Returns:
        Markdown đã được tái cấu trúc
    """
    sections = parse_markdown_structure(text)
    restructured_lines = []
    
    for section in sections:
        # Thêm header nếu có
        if section["header"]:
            header_prefix = "#" * section["level"]
            restructured_lines.append(f"{header_prefix} {section['header']}")
            restructured_lines.append("")
        
        # Xử lý content dựa trên type
        if section["type"] == "list":
            # Giữ nguyên list structure
            for item in section["content"]:
                restructured_lines.append(f"- {item}")
        elif section["type"] == "legal_structure":
            # Thêm title nếu có
            if section.get("legal_title"):
                restructured_lines.append(section["legal_title"])
                restructured_lines.append("")
            
            # Xử lý nội dung của điều/khoản
            content_text = " ".join(section["content"])
            # Tách thành các câu nếu quá dài
            sentences = re.split(r"([.!?]+\s+)", content_text)
            current_paragraph = ""
            
            for i in range(0, len(sentences), 2):
                sentence = sentences[i] + (sentences[i+1] if i+1 < len(sentences) else "")
                sentence = sentence.strip()
                
                if not sentence:
                    continue
                
                if len(current_paragraph) + len(sentence) > 500:
                    # Lưu paragraph hiện tại và bắt đầu paragraph mới
                    if current_paragraph:
                        restructured_lines.append(current_paragraph)
                        restructured_lines.append("")
                    current_paragraph = sentence
                else:
                    if current_paragraph:
                        current_paragraph += " " + sentence
                    else:
                        current_paragraph = sentence
            
            if current_paragraph:
                restructured_lines.append(current_paragraph)
                restructured_lines.append("")
        else:
            # Xử lý paragraph thông thường
            content_text = " ".join(section["content"])
            
            # Tách thành các đoạn hợp lý
            if len(content_text) > 500:
                # Tách theo câu
                sentences = re.split(r"([.!?]+\s+)", content_text)
                current_paragraph = ""
                
                for i in range(0, len(sentences), 2):
                    sentence = sentences[i] + (sentences[i+1] if i+1 < len(sentences) else "")
                    sentence = sentence.strip()
                    
                    if not sentence:
                        continue
                    
                    if len(current_paragraph) + len(sentence) > 400:
                        if current_paragraph:
                            restructured_lines.append(current_paragraph)
                            restructured_lines.append("")
                        current_paragraph = sentence
                    else:
                        if current_paragraph:
                            current_paragraph += " " + sentence
                        else:
                            current_paragraph = sentence
                
                if current_paragraph:
                    restructured_lines.append(current_paragraph)
                    restructured_lines.append("")
            else:
                if content_text:
                    restructured_lines.append(content_text)
                    restructured_lines.append("")
    
    return "\n".join(restructured_lines)

def preprocess_file(input_path: str, output_path: str = None) -> str:
    """
    Xử lý một file markdown hoặc text.
    
    Args:
        input_path: Đường dẫn file input
        output_path: Đường dẫn file output (nếu None thì ghi đè file input)
    
    Returns:
        Text đã được xử lý
    """
    if output_path is None:
        output_path = input_path
    
    try:
        # Đọc file
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        file_ext = Path(input_path).suffix.lower()
        is_markdown = file_ext in [".md", ".markdown"]
        
        print(f"📄 Đang xử lý: {os.path.basename(input_path)}")
        print(f"   Loại file: {'Markdown' if is_markdown else 'Text'}")
        print(f"   Kích thước ban đầu: {len(text)} ký tự")
        
        if is_markdown:
            # Xử lý markdown
            cleaned_text = clean_markdown(text)
            restructured_text = restructure_markdown(cleaned_text)
        else:
            # Xử lý text thuần (fallback cho .txt)
            # Loại bỏ số trang
            cleaned_text = re.sub(r"=+\s*Trang\s*\d+\s*=+", "", text)
            cleaned_text = re.sub(r"-\s*\d+\s*-", "", cleaned_text)
            cleaned_text = re.sub(r"\n\s*\n+", "\n\n", cleaned_text)
            
            # Tái cấu trúc đơn giản
            lines = cleaned_text.split("\n")
            paragraphs = []
            current_para = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    if current_para:
                        paragraphs.append(" ".join(current_para))
                        current_para = []
                else:
                    current_para.append(line)
            
            if current_para:
                paragraphs.append(" ".join(current_para))
            
            restructured_text = "\n\n".join(paragraphs)
        
        # Lưu file (luôn lưu dưới dạng .txt cho embeddings)
        if output_path.endswith(".md"):
            output_path = output_path.replace(".md", ".txt")
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(restructured_text)
        
        print(f"   ✅ Đã xử lý xong: {len(restructured_text)} ký tự")
        print(f"   💾 Đã lưu tại: {output_path}\n")
        
        return restructured_text
        
    except Exception as e:
        print(f"   ❌ Lỗi khi xử lý {input_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def preprocess_all_files(data_dir: str = None, markdown_dir: str = None):
    """
    Xử lý tất cả file .md trong thư mục markdown và lưu vào data/text/.
    
    Args:
        data_dir: Đường dẫn thư mục data/text để lưu output (nếu None thì dùng DATA_TEXT_DIR)
        markdown_dir: Đường dẫn thư mục markdown để đọc input (nếu None thì dùng MARKDOWN_DIR)
    """
    if data_dir is None:
        data_dir = DATA_TEXT_DIR
    if markdown_dir is None:
        markdown_dir = MARKDOWN_DIR
    
    all_files = []
    
    # Tìm file markdown trong documents/markdown/
    markdown_path = Path(markdown_dir)
    if markdown_path.exists():
        md_files = list(markdown_path.glob("*.md"))
        # Chuyển markdown từ documents/markdown/ sang data/text/ dưới dạng .txt
        for md_file in md_files:
            output_name = md_file.stem + ".txt"
            output_path = Path(data_dir) / output_name
            all_files.append((str(md_file), str(output_path)))
    else:
        print(f"⚠️  Thư mục markdown không tồn tại: {markdown_dir}")
        print(f"   Vui lòng chạy: python backend/src/read_word.py để tạo file markdown từ Word")
        return
    
    # Cũng tìm file markdown trong documents/ (từ PDF, backward compatibility)
    docs_path = Path(DOCUMENTS_DIR)
    if docs_path.exists():
        md_files = list(docs_path.glob("*.md"))
        for md_file in md_files:
            output_name = md_file.stem + ".txt"
            output_path = Path(data_dir) / output_name
            all_files.append((str(md_file), str(output_path)))
    
    if not all_files:
        print(f"⚠️  Không tìm thấy file .md nào trong {markdown_dir}")
        print(f"   Vui lòng chạy: python backend/src/read_word.py để tạo file markdown từ Word")
        return
    
    print(f"🔍 Tìm thấy {len(all_files)} file để xử lý\n")
    print("=" * 60)
    
    processed_count = 0
    for file_info in all_files:
        if isinstance(file_info, tuple):
            input_path, output_path = file_info
        else:
            input_path = str(file_info)
            output_path = None
        
        # Bỏ qua file backup hoặc temp
        if Path(input_path).name.startswith("~") or Path(input_path).name.startswith("."):
            continue
        
        result = preprocess_file(input_path, output_path)
        if result is not None:
            processed_count += 1
    
    print("=" * 60)
    print(f"✅ Hoàn tất! Đã xử lý {processed_count}/{len(all_files)} file")

if __name__ == "__main__":
    print("🚀 Bắt đầu xử lý các file markdown và text...\n")
    print("📂 Đọc file markdown từ:")
    print(f"   - {MARKDOWN_DIR}")
    print(f"📂 Lưu file text vào:")
    print(f"   - {DATA_TEXT_DIR}\n")
    preprocess_all_files()

