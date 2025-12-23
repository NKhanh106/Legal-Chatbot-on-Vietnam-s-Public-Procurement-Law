"""
Advanced embedding system for RAG (Retrieval-Augmented Generation).
Tối ưu hóa cho tài liệu pháp luật Việt Nam.
"""
import torch
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
# tqdm và json không được sử dụng trong code hiện tại, có thể dùng trong tương lai

# Helper function để lấy short path trên Windows (tránh lỗi Unicode với FAISS)
def get_short_path(long_path: str) -> str:
    """
    Lấy short path (8.3 format) trên Windows để tránh lỗi Unicode với FAISS.
    Trên Linux/Mac, trả về path gốc.
    """
    if sys.platform == 'win32':
        try:
            import win32api
            return win32api.GetShortPathName(long_path)
        except ImportError:
            # Nếu không có pywin32, thử dùng ctypes
            try:
                import ctypes
                from ctypes import wintypes
                
                GetShortPathNameW = ctypes.windll.kernel32.GetShortPathNameW
                GetShortPathNameW.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
                GetShortPathNameW.restype = wintypes.DWORD
                
                buffer = ctypes.create_unicode_buffer(260)
                result = GetShortPathNameW(long_path, buffer, 260)
                if result:
                    return buffer.value
            except (OSError, AttributeError, ValueError) as ctypes_err:
                # Log debug nếu cần
                pass
        except (OSError, AttributeError) as win32_err:
            # Log debug nếu cần
            pass
    # Nếu không thể lấy short path, trả về path gốc
    return long_path

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# DATA_DIR: Thư mục chứa file input (sau khi preprocess, file .txt sẽ ở đây)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
# DATA_TEXT_DIR: Thư mục chứa file .txt sau khi preprocess
DATA_TEXT_DIR = os.path.join(DATA_DIR, "text")
# STORE_DIR: Thư mục lưu index và metadata (cùng với data/)
STORE_DIR = os.path.join(PROJECT_ROOT, "data")

# Configuration
# Cho phép override qua ENV để dễ thử nghiệm model khác
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
EMBEDDING_MAX_LENGTH = int(os.getenv("EMBEDDING_MAX_LENGTH", "1024"))

# ============================================================================
# CHUNKING PARAMETERS - Tối ưu cho văn bản pháp luật lớn
# ============================================================================
# CHUNK_SIZE: Kích thước chunk mục tiêu (số từ)
# - Tăng từ 750 lên 1000 để:
#   + Giữ nguyên các định nghĩa dài và danh sách không bị cắt
#   + Giảm mất context ở đầu/đuôi chunk
#   + Phù hợp với văn bản pháp luật có cấu trúc phức tạp
#   + Vẫn trong khoảng tối ưu cho embedding models
CHUNK_SIZE = 1000  # Tối ưu cho legal documents lớn (tăng từ 750)

# CHUNK_OVERLAP: Số từ overlap giữa các chunks liên tiếp
# - Tăng từ 200 lên 300 (30% của CHUNK_SIZE) để:
#   + Giữ context tốt hơn giữa các chunks (quan trọng với văn bản có nhiều tham chiếu chéo)
#   + Đảm bảo không mất thông tin quan trọng ở ranh giới chunks
#   + Best practice: 20-30% của chunk size cho legal documents
CHUNK_OVERLAP = 300  # tối ưu cho legal documents (tăng từ 200)

# MIN_CHUNK_SIZE: Chunk tối thiểu (số từ)
# - Tăng từ 10 lên 20 để:
#   + Tránh chunks quá ngắn không có đủ context
#   + Với văn bản pháp luật, 50 từ là tối thiểu hợp lý để có ý nghĩa
#   + Giảm noise trong retrieval
MIN_CHUNK_SIZE = 10  # Chunk tối thiểu có ý nghĩa

# MAX_CHUNK_SIZE: Chunk tối đa (số từ)
# - Tăng từ 1000 lên 1500 để:
#   + Giữ nguyên các Điều rất dài mà không bị cắt
#   + Vẫn trong giới hạn hợp lý để tránh mất ngữ cảnh
#   + Một số Điều pháp luật có thể rất dài (1000+ từ)
MAX_CHUNK_SIZE = 1500  # Chunk tối đa để giữ nguyên Điều dài

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🖥️ Running on device: {DEVICE}")

# FAISS Index Configuration
USE_IVF_INDEX = True  # Sử dụng IndexIVFFlat cho hiệu suất tốt hơn với large datasets
USE_COMPRESSED_INDEX = False  # Sử dụng IndexPQ/IndexIVFPQ cho datasets rất lớn (>100k chunks) - giảm memory
N_CLUSTERS = 100  # Số clusters cho IVF index (tăng nếu có nhiều chunks)
PQ_M = 64  # Số subquantizers cho Product Quantization (chỉ dùng khi USE_COMPRESSED_INDEX=True)
PQ_BITS = 8  # Số bits cho mỗi subquantizer (chỉ dùng khi USE_COMPRESSED_INDEX=True)

# Performance Optimization
AUTO_BATCH_SIZE = True  # Tự động điều chỉnh batch size
DEFAULT_BATCH_SIZE = 16  # Giảm để tránh OOM với BGE-M3 (nặng hơn BKAI)
MAX_BATCH_SIZE = 64  # Giảm trần batch cho model lớn
MIN_BATCH_SIZE = 8  # Batch size tối thiểu (cho CPU hoặc GPU yếu)
VALIDATE_CHUNKS = True  # Validate chunks trước khi embed

class LegalDocumentChunker:
    """
    Chunker chuyên biệt cho tài liệu pháp luật.
    Chunk theo cấu trúc pháp luật: Chương > Mục > Điều > Khoản > Điểm
    """
    
    # Semantic boundaries - các từ khóa báo hiệu bắt đầu danh sách/định nghĩa
    SEMANTIC_BOUNDARIES = [
        r'gồm\s*:', r'bao gồm\s*:', r'như sau\s*:', r'sau đây\s*:',
        r'cụ thể\s*:', r'chi tiết\s*:', r'định nghĩa\s*:', r'quy định\s*:',
        r'bao gồm\s*:', r'gồm có\s*:', r'như\s*:', r'các\s*:'
    ]
    
    # TỐI ƯU: Compile regex patterns một lần để tránh recompile mỗi lần sử dụng
    _SEMANTIC_BOUNDARIES_COMPILED = [re.compile(pattern) for pattern in SEMANTIC_BOUNDARIES]
    # TỐI ƯU: Compile patterns với end anchor cho việc kiểm tra kết thúc chunk
    _SEMANTIC_BOUNDARIES_END_COMPILED = [re.compile(pattern + r'\s*$') for pattern in SEMANTIC_BOUNDARIES]
    
    @staticmethod
    def _detect_table_in_text(text: str) -> bool:
        """
        Phát hiện bảng markdown trong text.
        Format: | col1 | col2 | ... | với ít nhất 2 dòng có format này.
        """
        lines = text.split('\n')
        table_lines = 0
        for line in lines:
            line_stripped = line.strip()
            # Phát hiện markdown table: có ít nhất 2 dấu | và không phải separator (---)
            if '|' in line_stripped and line_stripped.count('|') >= 2:
                # Loại bỏ separator line (---)
                if not re.match(r'^\|\s*[-:|\s]+\s*\|', line_stripped):
                    table_lines += 1
        
        return table_lines >= 2  # Ít nhất 2 dòng có format bảng
    
    @staticmethod
    def _extract_table_from_text(text: str) -> Optional[str]:
        """
        Extract toàn bộ table từ text (từ dòng đầu tiên có | đến dòng cuối cùng).
        """
        lines = text.split('\n')
        table_start = None
        table_end = None
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if '|' in line_stripped and line_stripped.count('|') >= 2:
                if table_start is None:
                    table_start = i
                table_end = i
        
        if table_start is not None and table_end is not None:
            return '\n'.join(lines[table_start:table_end + 1])
        return None
    
    def __init__(self, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap
        # Import parse_markdown_structure từ preprocess
        try:
            sys.path.insert(0, os.path.join(PROJECT_ROOT, "backend", "src"))
            from preprocess import parse_markdown_structure
            self.parse_markdown_structure = parse_markdown_structure
        except ImportError:
            # Fallback nếu không import được
            self.parse_markdown_structure = None
    
    def semantic_chunk(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Chunk text theo cấu trúc pháp luật (Chương > Mục > Điều > Khoản > Điểm).
        Tôn trọng cấu trúc markdown và không chunk ngang qua các điều khoản.
        
        Args:
            text: Text cần chunk
            metadata: Metadata của document (source, headers, etc.)
        
        Returns:
            List các chunks với metadata phong phú
        """
        # Nếu có parse_markdown_structure, sử dụng nó
        if self.parse_markdown_structure:
            return self._chunk_by_legal_structure(text, metadata)
        else:
            # Fallback về phương pháp cũ
            return self._chunk_by_paragraphs(text, metadata)
    
    def _chunk_by_legal_structure(self, text: str, metadata: Dict = None) -> List[Dict]:
        """Chunk theo cấu trúc pháp luật từ parse_markdown_structure."""
        sections = self.parse_markdown_structure(text)
        
        chunks = []
        current_hierarchy = {
            "chapter": None,
            "section": None,
            "article": None,
            "article_number": None,
            "clause": None,
            "point": None
        }
        current_header = None
        
        for section in sections:
            # Cập nhật hierarchy dựa trên section type
            if section["type"] == "legal_structure":
                legal_type = section.get("legal_type", "")
                legal_number = section.get("legal_number", "")
                
                if legal_type == "Chương":
                    current_hierarchy["chapter"] = f"Chương {legal_number}"
                    current_hierarchy["section"] = None
                    current_hierarchy["article"] = None
                elif legal_type == "Mục":
                    current_hierarchy["section"] = f"Mục {legal_number}"
                elif legal_type == "Điều":
                    current_hierarchy["article"] = f"Điều {legal_number}"
                    try:
                        # Loại bỏ chữ "đ" trước khi parse int (ví dụ: "1đ" -> "1")
                        number_str = legal_number.rstrip('đĐ')
                        current_hierarchy["article_number"] = int(number_str)
                    except (ValueError, AttributeError):
                        current_hierarchy["article_number"] = None
                    current_hierarchy["clause"] = None
                    current_hierarchy["point"] = None
                elif legal_type == "Khoản":
                    current_hierarchy["clause"] = f"Khoản {legal_number}"
                    current_hierarchy["point"] = None
                elif legal_type == "Điểm":
                    current_hierarchy["point"] = f"Điểm {legal_number}"
                
                current_header = section.get("header", "")
            
            # Xử lý content của section
            content_text = "\n".join(section["content"]).strip()
            if not content_text:
                continue
            
            # Chunk content theo kích thước, nhưng không vượt qua boundary của section
            section_chunks = self._chunk_section_content(
                content_text, 
                current_hierarchy.copy(),
                current_header,
                section.get("header", ""),
                metadata
            )
            chunks.extend(section_chunks)
        
        return chunks
    
    def _chunk_section_content(self, content: str, hierarchy: Dict, 
                              section_header: str, legal_header: str,
                              metadata: Dict = None) -> List[Dict]:
        """
        Chunk nội dung của một section, tôn trọng kích thước chunk.
        Tối ưu cho văn bản pháp luật:
        - Ưu tiên giữ nguyên Điều/Khoản/Điểm làm một chunk nếu có thể
        - Chỉ tách khi quá dài (vượt MAX_CHUNK_SIZE)
        - Tôn trọng ranh giới cấu trúc pháp luật (Khoản, Điểm)
        - Giữ nguyên toàn bộ bảng biểu (không cắt bảng)
        """
        # Tách thành các đoạn (theo dòng trống hoặc markdown headers)
        # Phát hiện các ranh giới cấu trúc pháp luật (Khoản, Điểm) và bảng biểu
        lines = content.split("\n")
        paragraphs = []
        current_para = []
        in_table = False
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Phát hiện bảng markdown (bắt đầu bằng |)
            is_table_line = '|' in line_stripped and line_stripped.count('|') >= 2
            
            # Khởi tạo is_legal_boundary mặc định là False
            is_legal_boundary = False
            
            if is_table_line:
                # Nếu đang trong table, tiếp tục thêm vào
                if in_table:
                    current_para.append(line_stripped)
                else:
                    # Bắt đầu table mới
                    if current_para:
                        para_text = "\n".join(current_para).strip()
                        if para_text:
                            paragraphs.append(para_text)
                        current_para = []
                    current_para.append(line_stripped)
                    in_table = True
            elif in_table and not line_stripped:
                # Dòng trống sau table - kết thúc table
                para_text = "\n".join(current_para).strip()
                if para_text:
                    paragraphs.append(para_text)
                current_para = []
                in_table = False
            elif in_table:
                # Vẫn trong table (có thể có dòng separator ---)
                current_para.append(line_stripped)
            else:
                # Không phải table - xử lý legal boundary và paragraphs
                # Phát hiện ranh giới cấu trúc pháp luật (Khoản, Điểm)
                # Pattern: #### số. hoặc ##### chữ)
                is_legal_boundary = (
                    re.match(r"^####\s+\d+\.\s+", line_stripped) or  # Khoản
                    re.match(r"^#####\s+[a-zđ]\)\s+", line_stripped, re.IGNORECASE)  # Điểm
                )
            
            if is_legal_boundary:
                # Lưu paragraph hiện tại nếu có
                if current_para:
                    para_text = "\n".join(current_para).strip()
                    if para_text:
                        paragraphs.append(para_text)
                    current_para = []
                # Bắt đầu paragraph mới với ranh giới
                current_para.append(line_stripped)
            elif not line_stripped:
                # Dòng trống - kết thúc paragraph hiện tại
                if current_para:
                    para_text = "\n".join(current_para).strip()
                    if para_text:
                        paragraphs.append(para_text)
                    current_para = []
            else:
                current_para.append(line_stripped)
        
        # Thêm paragraph cuối cùng
        if current_para:
            para_text = "\n".join(current_para).strip()
            if para_text:
                paragraphs.append(para_text)
        
        chunks = []
        current_chunk_parts = []
        current_length = 0
        
        for para in paragraphs:
            para_words = len(para.split())
            
            # Kiểm tra xem paragraph có phải là bảng không
            is_table = self._detect_table_in_text(para)
            
            # Nếu là bảng, luôn giữ nguyên (không tách bảng)
            if is_table:
                # Nếu chunk hiện tại đã có nội dung, lưu lại trước khi thêm bảng
                if current_chunk_parts:
                    chunks.append(self._create_chunk_with_hierarchy(
                        current_chunk_parts, hierarchy, section_header, legal_header, metadata
                    ))
                    current_chunk_parts = []
                    current_length = 0
                
                # Thêm bảng vào chunk mới (bảng luôn là một chunk riêng)
                chunks.append(self._create_chunk_with_hierarchy(
                    [para], hierarchy, section_header, legal_header, metadata, is_table=True
                ))
                continue
            
            # Nếu đoạn quá lớn (vượt MAX_CHUNK_SIZE), bắt buộc tách
            if para_words > MAX_CHUNK_SIZE:
                # Lưu chunk hiện tại nếu có
                if current_chunk_parts:
                    chunks.append(self._create_chunk_with_hierarchy(
                        current_chunk_parts, hierarchy, section_header, legal_header, metadata
                    ))
                    current_chunk_parts = []
                    current_length = 0
                
                # Tách đoạn lớn thành sub-chunks theo câu
                sub_chunks = self._split_large_paragraph(para)
                for sub_chunk in sub_chunks:
                    sub_words = len(sub_chunk.split())
                    if current_length + sub_words > self.chunk_size and current_chunk_parts:
                        chunks.append(self._create_chunk_with_hierarchy(
                            current_chunk_parts, hierarchy, section_header, legal_header, metadata
                        ))
                        # Tạo overlap dựa trên CHUNK_OVERLAP
                        overlap = self._get_overlap_text(current_chunk_parts, target_words=self.overlap)
                        current_chunk_parts = [overlap, sub_chunk] if overlap else [sub_chunk]
                        current_length = len(" ".join(current_chunk_parts).split())
                    else:
                        current_chunk_parts.append(sub_chunk)
                        current_length += sub_words
            else:
                # Kiểm tra xem có thể thêm vào chunk hiện tại không
                # Ưu tiên: Nếu là Điều, cố gắng giữ nguyên trong một chunk nếu có thể
                is_article = hierarchy.get("article") is not None
                can_fit = current_length + para_words <= self.chunk_size
                # Cho phép vượt một chút (lên đến MAX_CHUNK_SIZE) để giữ nguyên Điều
                can_fit_with_tolerance = current_length + para_words <= MAX_CHUNK_SIZE
                
                # Kiểm tra xem chunk hiện tại có kết thúc bằng semantic boundary không
                # (ví dụ: "gồm:", "bao gồm:") - nếu có, bắt buộc phải thêm para tiếp theo
                # TỐI ƯU: Dùng compiled patterns với end anchor thay vì re.search mỗi lần
                current_text = " ".join(current_chunk_parts).lower()
                ends_with_boundary = any(pattern.search(current_text) 
                                        for pattern in self._SEMANTIC_BOUNDARIES_END_COMPILED)
                
                # Nếu chunk kết thúc bằng semantic boundary, bắt buộc phải thêm para tiếp theo
                if ends_with_boundary:
                    current_chunk_parts.append(para)
                    current_length += para_words
                # Nếu là Điều và có thể fit (kể cả với tolerance), luôn thêm vào (giữ nguyên Điều)
                elif is_article and can_fit_with_tolerance:
                    current_chunk_parts.append(para)
                    current_length += para_words
                # Nếu là Điều nhưng quá dài, vẫn thêm vào nhưng cảnh báo
                elif is_article and not can_fit_with_tolerance:
                    # Điều quá dài, bắt buộc phải tách
                    if current_chunk_parts:
                        chunks.append(self._create_chunk_with_hierarchy(
                            current_chunk_parts, hierarchy, section_header, legal_header, metadata
                        ))
                    # Tạo overlap lớn hơn cho Điều dài để giữ context
                    overlap = self._get_overlap_text(current_chunk_parts, target_words=self.overlap * 1.5) if current_chunk_parts else None
                    current_chunk_parts = [overlap, para] if overlap else [para]
                    current_length = len(" ".join(current_chunk_parts).split())
                # Nếu không phải Điều và không fit, tạo chunk mới
                elif current_length + para_words > self.chunk_size and current_chunk_parts:
                    chunks.append(self._create_chunk_with_hierarchy(
                        current_chunk_parts, hierarchy, section_header, legal_header, metadata
                    ))
                    # Tạo overlap dựa trên CHUNK_OVERLAP (tối ưu cho văn bản pháp luật)
                    # Overlap lớn hơn giúp giữ context tốt hơn với các tham chiếu chéo
                    overlap = self._get_overlap_text(current_chunk_parts, target_words=self.overlap)
                    current_chunk_parts = [overlap, para] if overlap else [para]
                    current_length = len(" ".join(current_chunk_parts).split())
                else:
                    current_chunk_parts.append(para)
                    current_length += para_words
        
        # Thêm chunk cuối cùng
        if current_chunk_parts:
            chunks.append(self._create_chunk_with_hierarchy(
                current_chunk_parts, hierarchy, section_header, legal_header, metadata
            ))
        
        return chunks
    
    def _chunk_by_paragraphs(self, text: str, metadata: Dict = None) -> List[Dict]:
        """Fallback: Chunk theo paragraphs (phương pháp cũ)."""
        paragraphs = self._split_into_paragraphs(text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for para in paragraphs:
            para_words = len(para.split())
            
            if para_words > self.chunk_size:
                sub_chunks = self._split_large_paragraph(para)
                for sub_chunk in sub_chunks:
                    if current_length + len(sub_chunk.split()) > self.chunk_size:
                        if current_chunk:
                            chunks.append(self._create_chunk(current_chunk, metadata))
                        current_chunk = [sub_chunk]
                        current_length = len(sub_chunk.split())
                    else:
                        current_chunk.append(sub_chunk)
                        current_length += len(sub_chunk.split())
            else:
                if current_length + para_words > self.chunk_size and current_chunk:
                    chunks.append(self._create_chunk(current_chunk, metadata))
                    overlap_text = self._get_overlap_text(current_chunk, target_words=self.overlap)
                    current_chunk = [overlap_text, para] if overlap_text else [para]
                    current_length = len(" ".join(current_chunk).split())
                else:
                    current_chunk.append(para)
                    current_length += para_words
        
        if current_chunk:
            chunks.append(self._create_chunk(current_chunk, metadata))
        
        return chunks
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Tách text thành các đoạn văn."""
        # Tách theo double newline
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Làm sạch và lọc
        cleaned_paragraphs = []
        for para in paragraphs:
            para = para.strip()
            # Lọc các đoạn quá ngắn (dùng MIN_CHUNK_SIZE để nhất quán)
            if para and len(para.split()) >= MIN_CHUNK_SIZE:
                cleaned_paragraphs.append(para)
        
        return cleaned_paragraphs
    
    def _split_large_paragraph(self, paragraph: str) -> List[str]:
        """
        Tách đoạn văn lớn thành các chunks nhỏ hơn.
        Tách theo câu để giữ nguyên ngữ nghĩa.
        Tối ưu: Tách theo ranh giới cấu trúc pháp luật nếu có.
        """
        # Kiểm tra xem có ranh giới cấu trúc pháp luật không (Khoản, Điểm)
        lines = paragraph.split("\n")
        has_legal_boundaries = False
        for line in lines:
            if re.match(r"^####\s+\d+\.\s+", line.strip()) or \
               re.match(r"^#####\s+[a-zđ]\)\s+", line.strip(), re.IGNORECASE):
                has_legal_boundaries = True
                break
        
        # Nếu có ranh giới, tách theo ranh giới trước
        if has_legal_boundaries:
            chunks = []
            current_chunk = []
            
            for line in lines:
                line_stripped = line.strip()
                is_boundary = (
                    re.match(r"^####\s+\d+\.\s+", line_stripped) or
                    re.match(r"^#####\s+[a-zđ]\)\s+", line_stripped, re.IGNORECASE)
                )
                
                if is_boundary and current_chunk:
                    # Lưu chunk hiện tại
                    chunk_text = "\n".join(current_chunk).strip()
                    if chunk_text:
                        chunks.append(chunk_text)
                    current_chunk = [line_stripped]
                else:
                    current_chunk.append(line_stripped)
            
            # Thêm chunk cuối
            if current_chunk:
                chunk_text = "\n".join(current_chunk).strip()
                if chunk_text:
                    chunks.append(chunk_text)
            
            # Nếu chunks vẫn quá lớn, tách tiếp theo câu
            final_chunks = []
            for chunk in chunks:
                chunk_words = len(chunk.split())
                if chunk_words > self.chunk_size:
                    # Tách tiếp theo câu
                    sub_chunks = self._split_by_sentences(chunk)
                    final_chunks.extend(sub_chunks)
                else:
                    final_chunks.append(chunk)
            
            return final_chunks
        
        # Không có ranh giới, tách theo câu
        return self._split_by_sentences(paragraph)
    
    def _split_by_sentences(self, text: str) -> List[str]:
        """
        Tách text thành chunks theo câu.
        """
        # Tách theo câu (giữ lại dấu câu)
        sentences = re.split(r'([.!?]+[)\]\}"\']*\s+)', text)
        
        # Ghép lại câu với dấu câu
        combined_sentences = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                combined_sentences.append(sentences[i] + sentences[i + 1])
            else:
                combined_sentences.append(sentences[i])
        
        # Nhóm các câu thành chunks hợp lý
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in combined_sentences:
            sent_words = len(sentence.split())
            
            if current_length + sent_words > self.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = sent_words
            else:
                current_chunk.append(sentence)
                current_length += sent_words
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def _get_overlap_text(self, chunk: List[str], target_words: int = None, max_sentences: int = None) -> str:
        """
        Lấy phần overlap từ chunk cuối cùng.
        Cải thiện: Không cắt giữa danh sách, đảm bảo giữ nguyên semantic boundaries.
        QUAN TRỌNG: Không bao giờ cắt bảng biểu trong overlap.
        
        Args:
            chunk: List các phần text
            target_words: Số từ mục tiêu cho overlap (ưu tiên, dùng CHUNK_OVERLAP)
            max_sentences: Số câu tối đa để lấy làm overlap (fallback nếu không có target_words)
        
        Returns:
            Text overlap với số từ gần target_words nhất, không cắt giữa danh sách hoặc bảng
        """
        if not chunk:
            return ""
        
        if target_words is None:
            target_words = self.overlap  # Dùng CHUNK_OVERLAP mặc định
        
        # Lấy text từ cuối chunk (từ các paragraph cuối)
        # Bắt đầu từ paragraph cuối và lùi dần để đạt target_words
        # TỐI ƯU: Dùng list và reverse sau thay vì insert(0) để tránh O(N) mỗi lần insert
        overlap_parts = []
        overlap_word_count = 0
        
        # TỐI ƯU: Compile regex patterns một lần
        semantic_patterns = self._SEMANTIC_BOUNDARIES_COMPILED
        
        # Lấy từ cuối lên
        for para in reversed(chunk):
            # TỐI ƯU: Cache split() và lower() để tránh tính lại nhiều lần
            para_words_list = para.split()
            para_words = len(para_words_list)
            para_lower = para.lower()
            
            # QUAN TRỌNG: Kiểm tra xem paragraph có phải là bảng không
            # Nếu là bảng, KHÔNG BAO GIỜ cắt - lấy cả bảng hoặc không lấy gì
            is_table_para = self._detect_table_in_text(para)
            if is_table_para:
                # Nếu đã có overlap_parts, dừng lại (không lấy bảng vào overlap)
                # Nếu chưa có gì, có thể lấy bảng nhưng chỉ khi không có lựa chọn khác
                if overlap_parts:
                    break  # Đã có overlap, không lấy bảng
                else:
                    # Chưa có overlap, nhưng không nên lấy bảng vào overlap
                    # (bảng nên là một chunk riêng)
                    break
            
            # Kiểm tra xem paragraph có chứa semantic boundary không
            # TỐI ƯU: Dùng compiled patterns thay vì re.search mỗi lần
            has_boundary = any(pattern.search(para_lower) for pattern in semantic_patterns)
            
            # Nếu thêm paragraph này vẫn chưa vượt quá target_words * 1.5, thêm vào
            # (cho phép vượt một chút để giữ nguyên câu)
            if overlap_word_count + para_words <= target_words * 1.5:
                # TỐI ƯU: Append thay vì insert(0), sẽ reverse sau
                overlap_parts.append(para)
                overlap_word_count += para_words
                
                # Nếu paragraph có semantic boundary và chưa đủ target_words, tiếp tục lấy thêm
                if has_boundary and overlap_word_count < target_words:
                    continue
                
                # Nếu đã đạt target_words, dừng lại
                if overlap_word_count >= target_words:
                    break
            else:
                # Nếu paragraph quá lớn, tách theo câu
                # Nhưng không cắt nếu có semantic boundary
                if has_boundary:
                    # Nếu có boundary, lấy cả paragraph (quan trọng hơn target_words)
                    overlap_parts.append(para)
                    overlap_word_count += para_words
                    break
                
                # Tách paragraph thành các câu
                sentences = re.split(r'([.!?]+[)\]\}"\']*\s+)', para)
                combined_sentences = []
                for i in range(0, len(sentences) - 1, 2):
                    if i + 1 < len(sentences):
                        combined_sentences.append(sentences[i] + sentences[i + 1])
                    else:
                        combined_sentences.append(sentences[i])
                
                # Lấy các câu cuối để đạt target_words
                for sent in reversed(combined_sentences):
                    # TỐI ƯU: Cache split() để tránh tính lại
                    sent_words = len(sent.split())
                    if overlap_word_count + sent_words <= target_words * 1.5:
                        overlap_parts.append(sent)
                        overlap_word_count += sent_words
                        if overlap_word_count >= target_words:
                            break
                
                break
        
        # TỐI ƯU: Reverse một lần thay vì insert(0) nhiều lần
        overlap_parts.reverse()
        overlap_text = " ".join(overlap_parts).strip()
        
        # Đảm bảo overlap không quá lớn (tối đa 1.5x target_words)
        # Nhưng nếu có semantic boundary, cho phép vượt một chút
        overlap_words = overlap_text.split()
        max_overlap_words = int(target_words * 1.5)
        # TỐI ƯU: Cache lower() và dùng compiled patterns
        overlap_text_lower = overlap_text.lower()
        has_boundary_in_overlap = any(pattern.search(overlap_text_lower) for pattern in semantic_patterns)
        
        if len(overlap_words) > max_overlap_words and not has_boundary_in_overlap:
            overlap_text = " ".join(overlap_words[-max_overlap_words:])
        
        return overlap_text
    
    def _create_chunk_with_hierarchy(self, text_parts: List[str], hierarchy: Dict,
                                     section_header: str, legal_header: str,
                                     metadata: Dict = None, is_table: bool = False) -> Dict:
        """
        Tạo chunk với metadata phong phú về cấu trúc pháp luật.
        
        Args:
            text_parts: List các phần text
            hierarchy: Hierarchy metadata
            section_header: Section header
            legal_header: Legal header
            metadata: Document metadata
            is_table: True nếu chunk này là một bảng biểu
        """
        # Nếu là bảng, giữ nguyên format markdown (không join bằng space)
        if is_table:
            chunk_text = "\n".join(text_parts).strip()
        else:
            chunk_text = " ".join(text_parts).strip()
        
        # Xây dựng hierarchy list
        hierarchy_list = []
        if hierarchy.get("chapter"):
            hierarchy_list.append(hierarchy["chapter"])
        if hierarchy.get("section"):
            hierarchy_list.append(hierarchy["section"])
        if hierarchy.get("article"):
            hierarchy_list.append(hierarchy["article"])
        if hierarchy.get("clause"):
            hierarchy_list.append(hierarchy["clause"])
        if hierarchy.get("point"):
            hierarchy_list.append(hierarchy["point"])
        
        # Phát hiện bảng trong chunk (nếu chưa được đánh dấu)
        # Nếu is_table=True, đã biết chắc là bảng, không cần detect lại
        has_table = is_table or self._detect_table_in_text(chunk_text)
        table_data = None
        if has_table:
            table_data = self._extract_table_from_text(chunk_text)
            # Đảm bảo table_data không None
            if table_data is None:
                table_data = chunk_text  # Fallback: dùng toàn bộ chunk_text
        
        chunk_data = {
            "text": chunk_text,
            "word_count": len(chunk_text.split()),
            "char_count": len(chunk_text),
            # Metadata về cấu trúc pháp luật
            "chapter": hierarchy.get("chapter", ""),
            "section": hierarchy.get("section", ""),
            "article": hierarchy.get("article", ""),
            "article_number": hierarchy.get("article_number"),
            "clause": hierarchy.get("clause", ""),
            "point": hierarchy.get("point", ""),
            "hierarchy": hierarchy_list,
            "section_header": section_header or "",
            "legal_header": legal_header or "",
            # Metadata về bảng biểu
            "has_table": has_table,
            "table_data": table_data,  # Raw table markdown nếu có
        }
        
        # Thêm metadata từ document level
        if metadata:
            chunk_data.update({
                "source": metadata.get("source", ""),
                "file_path": metadata.get("file_path", ""),
                # Metadata về năm và loại văn bản (quan trọng cho temporal filtering)
                "year": metadata.get("year"),
                "document_type": metadata.get("document_type"),
                "document_number": metadata.get("document_number"),
                "is_active": True,  # Mặc định là active, có thể parse từ văn bản sau
            })
        
        return chunk_data
    
    def _create_chunk(self, text_parts: List[str], metadata: Dict = None) -> Dict:
        """Tạo chunk với metadata (fallback method)."""
        chunk_text = " ".join(text_parts).strip()
        
        chunk_data = {
            "text": chunk_text,
            "word_count": len(chunk_text.split()),
            "char_count": len(chunk_text),
        }
        
        # Thêm metadata nếu có
        if metadata:
            chunk_data.update({
                "source": metadata.get("source", ""),
                "section": metadata.get("section", ""),
                "header": metadata.get("header", ""),
                "legal_type": metadata.get("legal_type", ""),
                "legal_number": metadata.get("legal_number", ""),
            })
        
        return chunk_data

class EmbeddingSystem:
    """Hệ thống embedding tối ưu cho RAG."""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL, device: str = DEVICE, load_model: bool = True):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.dimension = None
        if load_model:
            self._ensure_model_loaded()
        
        self.chunker = LegalDocumentChunker()

    def _ensure_model_loaded(self):
        """Lazy load model khi cần để tránh load 2 lần khi chỉ chunking."""
        if self.model is not None:
            return
        print(f"🔄 Đang load embedding model: {self.model_name}")
        model_kwargs = {}
        if self.device == 'cuda':
            # Sử dụng dtype thay vì torch_dtype (deprecated)
            model_kwargs["dtype"] = torch.float16  # Giảm VRAM cho BGE-M3
        self.model = SentenceTransformer(self.model_name, device=self.device, model_kwargs=model_kwargs)
        # BGE-M3 hỗ trợ sequence length tới 8192; dùng 1024 để cân bằng hiệu năng
        try:
            self.model.max_seq_length = EMBEDDING_MAX_LENGTH
        except Exception:
            # Không phải model nào cũng expose max_seq_length, nên ignore nếu không set được
            pass
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"✅ Model loaded. Dimension: {self.dimension}")
    
    def process_file(self, file_path: str) -> Tuple[List[Dict], str]:
        """
        Xử lý một file và tạo chunks.
        
        Args:
            file_path: Đường dẫn đến file
        
        Returns:
            Tuple (chunks, file_name)
        """
        file_name = Path(file_path).stem
        
        print(f"\n📄 Đang xử lý: {file_name}")
        
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        # Tạo metadata
        metadata = {
            "source": file_name,
            "file_path": file_path
        }
        
        # Extract metadata từ filename (năm, loại văn bản, số văn bản)
        filename_metadata = self._extract_metadata_from_filename(file_name)
        metadata.update(filename_metadata)
        
        # Parse cấu trúc nếu có
        metadata.update(self._extract_structure(text))
        
        # Chunk text
        chunks = self.chunker.semantic_chunk(text, metadata)
        
        # Thống kê chunks
        total_words = sum(c['word_count'] for c in chunks)
        avg_words = total_words // len(chunks) if chunks else 0
        chunks_with_article = sum(1 for c in chunks if c.get('article'))
        chunks_with_chapter = sum(1 for c in chunks if c.get('chapter'))
        
        print(f"   ✅ Đã tạo {len(chunks)} chunks")
        print(f"   📊 Tổng {total_words:,} từ (trung bình {avg_words} từ/chunk)")
        if chunks_with_article > 0:
            print(f"   📑 {chunks_with_article} chunks có thông tin Điều")
        if chunks_with_chapter > 0:
            print(f"   📚 {chunks_with_chapter} chunks có thông tin Chương")
        
        return chunks, file_name
    
    def _extract_metadata_from_filename(self, file_name: str) -> Dict:
        """
        Extract metadata từ filename.
        Pattern: "02_2024_TT-BKHDT_591580.txt" -> year=2024, doc_type=TT, number=591580
        """
        metadata = {}
        
        # Extract năm (4 chữ số)
        year_match = re.search(r'_(\d{4})_', file_name)
        if year_match:
            try:
                metadata["year"] = int(year_match.group(1))
            except ValueError:
                pass
        
        # Extract loại văn bản (TT, ND, QD, CT, etc.)
        doc_type_match = re.search(r'_([A-Z]{2,3})-', file_name)
        if doc_type_match:
            metadata["document_type"] = doc_type_match.group(1)
        
        # Extract số văn bản
        number_match = re.search(r'_(\d+)(?:\.txt)?$', file_name)
        if number_match:
            try:
                metadata["document_number"] = number_match.group(1)
            except ValueError:
                pass
        
        return metadata
    
    def _extract_structure(self, text: str) -> Dict:
        """Trích xuất cấu trúc từ text với thông tin chi tiết hơn."""
        structure = {}
        
        # Tìm các header markdown
        headers = re.findall(r'^(#{1,6})\s+(.+)$', text, re.MULTILINE)
        if headers:
            structure["has_headers"] = True
            structure["header_count"] = len(headers)
        
        # Tìm cấu trúc pháp luật với số lượng chi tiết
        # Bao gồm cả chữ "đ" trong tiếng Việt (ví dụ: "Điều 1đ", "Khoản 2đ")
        chapters = re.findall(r'Chương\s+(\d+[a-zđ]?)', text, re.IGNORECASE)
        sections = re.findall(r'Mục\s+(\d+[a-zđ]?)', text, re.IGNORECASE)
        articles = re.findall(r'Điều\s+(\d+[a-zđ]?)', text, re.IGNORECASE)
        clauses = re.findall(r'Khoản\s+(\d+[a-zđ]?)', text, re.IGNORECASE)
        points = re.findall(r'Điểm\s+([a-zđ])', text, re.IGNORECASE)
        
        if chapters or sections or articles:
            structure["has_legal_structure"] = True
            structure["chapter_count"] = len(chapters)
            structure["section_count"] = len(sections)
            structure["article_count"] = len(articles)
            structure["clause_count"] = len(clauses)
            structure["point_count"] = len(points)
            structure["legal_structure_count"] = len(chapters) + len(sections) + len(articles) + len(clauses) + len(points)
        
        return structure
    
    def _validate_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Validate và filter chunks trước khi embed."""
        if not VALIDATE_CHUNKS:
            return chunks
        
        validated_chunks = []
        filtered_count = 0
        
        for chunk in chunks:
            text = chunk.get("text", "").strip()
            word_count = chunk.get("word_count", 0)
            
            # Lọc chunks quá ngắn hoặc quá dài
            if not text:
                filtered_count += 1
                continue
            
            if word_count < MIN_CHUNK_SIZE:
                filtered_count += 1
                continue
            
            if word_count > MAX_CHUNK_SIZE:
                # Cảnh báo nhưng vẫn giữ lại
                print(f"   ⚠️  Chunk có {word_count} từ (vượt MAX_CHUNK_SIZE={MAX_CHUNK_SIZE})")
            
            validated_chunks.append(chunk)
        
        if filtered_count > 0:
            print(f"   ℹ️  Đã lọc {filtered_count} chunks không hợp lệ")
        
        return validated_chunks
    
    def _calculate_optimal_batch_size(self, chunks: List[Dict], device: str) -> int:
        """Tính toán batch size tối ưu dựa trên chunks và device."""
        if not AUTO_BATCH_SIZE:
            return DEFAULT_BATCH_SIZE
        
        # Tính độ dài trung bình của chunks
        avg_length = sum(len(c.get("text", "").split()) for c in chunks) / len(chunks) if chunks else 0
        
        # Điều chỉnh batch size dựa trên độ dài trung bình
        if device == 'cuda':
            # GPU: batch size lớn hơn
            if avg_length < 200:
                batch_size = min(MAX_BATCH_SIZE, 64)  # Chunks ngắn -> batch lớn
            elif avg_length < 400:
                batch_size = min(MAX_BATCH_SIZE, 32)  # Chunks trung bình
            else:
                batch_size = min(MAX_BATCH_SIZE, 16)  # Chunks dài -> batch nhỏ
        else:
            # CPU: batch size nhỏ hơn
            if avg_length < 200:
                batch_size = min(16, 16)
            elif avg_length < 400:
                batch_size = min(8, 8)
            else:
                batch_size = MIN_BATCH_SIZE
        
        return max(MIN_BATCH_SIZE, min(batch_size, MAX_BATCH_SIZE))
    
    def create_embeddings(self, chunks: List[Dict], batch_size: Optional[int] = None) -> np.ndarray:
        """
        Tạo embeddings cho các chunks với tối ưu tự động.
        
        Args:
            chunks: List các chunks
            batch_size: Batch size cho encoding (None = tự động)
        
        Returns:
            Numpy array của embeddings
        """
        self._ensure_model_loaded()
        # Validate chunks
        validated_chunks = self._validate_chunks(chunks)
        
        if not validated_chunks:
            raise ValueError("Không có chunks hợp lệ để tạo embeddings")
        
        # Safe get để tránh KeyError nếu chunk không có "text" key
        texts = [chunk.get("text", "") for chunk in validated_chunks]
        
        # Tự động tính batch size nếu không được chỉ định
        if batch_size is None:
            device = 'cuda' if os.getenv("CUDA_VISIBLE_DEVICES") else 'cpu'
            batch_size = self._calculate_optimal_batch_size(validated_chunks, device)
            print(f"🔄 Đang tạo embeddings cho {len(texts)} chunks (batch_size={batch_size})...")
        else:
            print(f"🔄 Đang tạo embeddings cho {len(texts)} chunks...")
        
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,  # Normalize để tối ưu cosine similarity
                device=self.model.device  # Sử dụng device của model
            )
            print(f"✅ Đã tạo embeddings: shape {embeddings.shape}")
            return embeddings
        except RuntimeError as e:
            # Nếu lỗi do OOM, thử giảm batch size
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                print(f"   ⚠️  GPU OOM, thử giảm batch size...")
                if batch_size > MIN_BATCH_SIZE:
                    new_batch_size = max(MIN_BATCH_SIZE, batch_size // 2)
                    print(f"   🔄 Retry với batch_size={new_batch_size}")
                    return self.create_embeddings(validated_chunks, batch_size=new_batch_size)
            raise
        except KeyboardInterrupt:
            print("\n⚠️  Đã dừng quá trình tạo embeddings (KeyboardInterrupt)")
            raise
        except Exception as e:
            print(f"\n❌ Lỗi khi tạo embeddings: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def create_faiss_index(self, embeddings: np.ndarray, use_ivf: bool = USE_IVF_INDEX) -> faiss.Index:
        """
        Tạo FAISS index với chiến lược "Accuracy First" cho dữ liệu pháp luật.
        - Ưu tiên IndexFlatIP (exact) cho dataset < 50k vectors.
        - Dùng IndexIVFFlat (không nén) cho 50k - 1M vectors (nhanh hơn, vẫn giữ nguyên vector).
        - Chỉ dùng IndexIVFPQ (nén, giảm độ chính xác) khi >1M vectors hoặc bắt buộc do bộ nhớ.
        Normalize + Inner Product = cosine similarity chuẩn cho embeddings đã normalize.
        
        Args:
            embeddings: Embeddings array
            use_ivf: Có sử dụng IVF index không (tốt hơn cho large datasets)
        
        Returns:
            FAISS index
        """
        dimension = embeddings.shape[1]
        n_vectors = embeddings.shape[0]
        
        print(f"\n🔧 Đang tạo FAISS index...")
        print(f"   Dimension: {dimension}")
        print(f"   Vectors: {n_vectors}")
        
        # Điều chỉnh số clusters dựa trên số lượng vectors
        optimal_clusters = min(N_CLUSTERS, max(4, n_vectors // 10))
        
        # Nếu tắt IVF, ép dùng exact search
        if not use_ivf:
            print("   🧭 USE_IVF_INDEX=False -> dùng IndexFlatIP (exact, accuracy-first)")
            index = faiss.IndexFlatIP(dimension)
            index.add(embeddings.astype('float32'))
            print("   ✅ IndexFlatIP đã được tạo")
            return index
        
        # Chiến lược Accuracy First
        if n_vectors < 50_000:
            # Small/medium dataset: exact search
            print("   🧭 Accuracy First: dùng IndexFlatIP (exact) cho <50k vectors")
            print("   ⚡ Normalize + IP = cosine similarity chuẩn, không mất mát thông tin")
            index = faiss.IndexFlatIP(dimension)
            index.add(embeddings.astype('float32'))
            print("   ✅ IndexFlatIP đã được tạo")
        elif n_vectors <= 1_000_000:
            # Medium-large dataset: IVF without compression
            print(f"   🧭 Accuracy First: dùng IndexIVFFlat (no compression) cho 50k-1M vectors")
            print("   ⚡ Không nén vector -> giữ nguyên độ chính xác của BGE-M3, vẫn tăng tốc nhờ clustering")
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, optimal_clusters, faiss.METRIC_INNER_PRODUCT)
            
            print("   🔄 Đang train index (IVFFlat)...")
            index.train(embeddings.astype('float32'))
            index.add(embeddings.astype('float32'))
            
            # nprobe: cân bằng speed/accuracy, ưu tiên accuracy
            index.nprobe = min(max(8, optimal_clusters // 8), 64)
            print(f"   ✅ IndexIVFFlat đã được tạo (nprobe={index.nprobe}, metric=INNER_PRODUCT)")
        else:
            # Very large dataset: allow PQ but warn about accuracy loss
            print("   ⚠️  Dataset >1M vectors -> dùng IndexIVFPQ (compressed) để tiết kiệm bộ nhớ")
            print("   ⚠️  PQ có thể giảm độ chính xác cho các điều khoản luật gần nghĩa; chỉ dùng khi bắt buộc.")
            print(f"   PQ parameters: m={PQ_M}, bits={PQ_BITS}")
            
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFPQ(quantizer, dimension, optimal_clusters, PQ_M, PQ_BITS)
            index.metric_type = faiss.METRIC_INNER_PRODUCT
            
            print("   🔄 Đang train index (IVFPQ)...")
            index.train(embeddings.astype('float32'))
            index.add(embeddings.astype('float32'))
            
            index.nprobe = min(max(16, optimal_clusters // 6), 80)
            print(f"   ✅ IndexIVFPQ đã được tạo (nprobe={index.nprobe}, metric=INNER_PRODUCT)")
            print(f"   💾 Approx memory (compressed): ~{n_vectors * dimension * PQ_BITS / 8 / (1024**2):.1f} MB")
        
        return index
    
    def save_index(self, index: faiss.Index, chunks: List[Dict], 
                   file_name: str, output_dir: str = STORE_DIR):
        """
        Lưu index và metadata.
        
        Args:
            index: FAISS index
            chunks: List chunks với metadata
            file_name: Tên file để lưu
            output_dir: Thư mục output
        """
        # Đảm bảo thư mục output tồn tại
        os.makedirs(output_dir, exist_ok=True)
        
        # Sử dụng pathlib để xử lý đường dẫn Unicode tốt hơn
        index_path_obj = Path(output_dir) / f"{file_name}.index"
        metadata_path_obj = Path(output_dir) / f"{file_name}_meta.pkl"
        
        # Lưu index
        print(f"\n💾 Đang lưu index: {index_path_obj}")
        try:
            # Đảm bảo thư mục parent tồn tại
            index_path_obj.parent.mkdir(parents=True, exist_ok=True)
            
            # Chuyển Path object thành string cho FAISS
            abs_index_path = str(index_path_obj.resolve())
            
            # Trên Windows, sử dụng short path để tránh lỗi Unicode với FAISS
            if sys.platform == 'win32':
                try:
                    short_path = get_short_path(abs_index_path)
                    if short_path and short_path != abs_index_path:
                        print(f"   📝 Sử dụng short path: {short_path}")
                        abs_index_path = short_path
                except Exception as short_path_err:
                    print(f"   ⚠️  Không thể lấy short path, dùng path gốc: {short_path_err}")
            
            # Test write trước để đảm bảo có quyền ghi
            test_file = index_path_obj.parent / "test_write.tmp"
            try:
                test_file.write_bytes(b'test')
                test_file.unlink()
            except Exception as test_err:
                print(f"   ⚠️  Không thể ghi vào thư mục: {index_path_obj.parent}")
                print(f"   Lỗi test: {str(test_err)}")
                raise
            
            # Lưu index với FAISS
            # Sử dụng short path trên Windows để tránh lỗi Unicode
            try:
                faiss.write_index(index, abs_index_path)
            except Exception as faiss_err:
                # Nếu vẫn lỗi, thử lưu vào thư mục tạm rồi copy
                if "could not open" in str(faiss_err).lower() or "unicode" in str(faiss_err).lower():
                    print(f"   ⚠️  Lỗi Unicode với FAISS, thử giải pháp dự phòng...")
                    import tempfile
                    import shutil
                    
                    # Tạo file tạm trong thư mục tạm (không có Unicode)
                    temp_dir = tempfile.gettempdir()
                    temp_index = os.path.join(temp_dir, f"{file_name}.index")
                    
                    # Lưu vào thư mục tạm
                    faiss.write_index(index, temp_index)
                    
                    # Copy sang thư mục đích
                    shutil.copy2(temp_index, str(index_path_obj))
                    os.remove(temp_index)
                    
                    print(f"   ✅ Đã lưu index qua thư mục tạm thành công")
                else:
                    raise
            
            # Verify file đã được tạo
            if index_path_obj.exists():
                file_size = index_path_obj.stat().st_size
                print(f"   ✅ Đã lưu index thành công ({file_size:,} bytes)")
            else:
                print(f"   ⚠️  Cảnh báo: File có thể chưa được tạo đúng")
        except Exception as e:
            print(f"   ❌ Lỗi khi lưu index: {str(e)}")
            print(f"   Đường dẫn: {abs_index_path}")
            print(f"   Thư mục tồn tại: {index_path_obj.parent.exists()}")
            import traceback
            traceback.print_exc()
            raise
        
        # Lưu metadata
        metadata = {
            "chunks": [chunk.get("text", "") for chunk in chunks],  # Chỉ lưu text cho backward compatibility (safe get)
            "chunks_full": chunks,  # Lưu full metadata
            "file_name": file_name,
            "total_chunks": len(chunks),
            "model": EMBEDDING_MODEL,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
        }
        
        try:
            # Đảm bảo thư mục tồn tại
            metadata_path_obj.parent.mkdir(parents=True, exist_ok=True)
            
            # Lưu metadata sử dụng pathlib
            with open(metadata_path_obj, "wb") as f:
                pickle.dump(metadata, f)
            
            # Verify file đã được tạo
            if metadata_path_obj.exists():
                file_size = metadata_path_obj.stat().st_size
                print(f"💾 Đã lưu metadata: {metadata_path_obj} ({file_size:,} bytes)")
            else:
                print(f"💾 Đã lưu metadata: {metadata_path_obj}")
            
            print(f"✅ Hoàn tất! Index có {index.ntotal} vectors")
        except Exception as e:
            print(f"❌ Lỗi khi lưu metadata: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

def process_all_files(data_dir: str = DATA_TEXT_DIR, 
                      file_patterns: List[str] = None,
                      combine_all: bool = True) -> Tuple[List[Dict], str]:
    """
    Xử lý tất cả file trong thư mục data/text và gom lại thành một.
    
    Args:
        data_dir: Thư mục chứa files (mặc định: data/text/)
        file_patterns: List các pattern để tìm file (mặc định: ["*.txt"])
        combine_all: Nếu True, gom tất cả file thành một index duy nhất
    
    Returns:
        Tuple (all_chunks, file_name) nếu combine_all=True (file_name sẽ là "data_for_rag")
        Hoặc Dict mapping file_name -> (chunks, file_name) nếu combine_all=False
    """
    if file_patterns is None:
        file_patterns = ["*.txt"]  # Chỉ đọc file .txt từ data/text/
    
    data_path = Path(data_dir)
    all_files = []
    
    for pattern in file_patterns:
        all_files.extend(data_path.glob(pattern))
    
    # Loại bỏ file backup và metadata
    files_to_process = [
        f for f in all_files 
        if not f.name.startswith("~") 
        and not f.name.endswith("_meta.pkl")
        and not f.name.endswith(".index")
        and not f.name.endswith("_corrected")  # Bỏ qua file corrected nếu có file gốc
    ]
    
    if not files_to_process:
        print(f"⚠️  Không tìm thấy file nào trong {data_dir}")
        return ([], "data_for_rag") if combine_all else {}
    
    print(f"🔍 Tìm thấy {len(files_to_process)} file để xử lý")
    if combine_all:
        print("📚 Sẽ gom tất cả file thành một index duy nhất")
    print("=" * 60)
    
    embedding_system = EmbeddingSystem(load_model=False)
    all_chunks = []
    results_dict = {}  # Dùng cho legacy mode
    
    for file_path in files_to_process:
        try:
            chunks, file_name = embedding_system.process_file(str(file_path))
            
            if combine_all:
                # Thêm source file vào metadata của mỗi chunk
                for chunk in chunks:
                    chunk["source_file"] = file_name
                    chunk["source_path"] = str(file_path)
                all_chunks.extend(chunks)
            else:
                # Giữ nguyên cách cũ (từng file riêng)
                results_dict[file_name] = (chunks, file_name)
        except Exception as e:
            print(f"❌ Lỗi khi xử lý {file_path}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    if combine_all:
        print(f"\n✅ Đã gom {len(files_to_process)} file thành {len(all_chunks)} chunks")
        return (all_chunks, "data_for_rag")
    else:
        return results_dict

def create_embeddings_for_files(results, output_dir: str = STORE_DIR):
    """
    Tạo embeddings và index cho tất cả files đã xử lý.
    Hỗ trợ cả mode gom tất cả thành một index hoặc từng file riêng.
    
    Args:
        results: 
            - Nếu là Tuple (all_chunks, file_name): Gom tất cả thành một index
            - Nếu là Dict: Tạo index riêng cho từng file (legacy mode)
        output_dir: Thư mục output
    """
    # Load model một lần cho tất cả files (tối ưu memory)
    embedding_system = EmbeddingSystem()
    
    print("\n" + "=" * 60)
    print("🚀 Bắt đầu tạo embeddings và index...")
    print("=" * 60)
    
    # Kiểm tra xem results là tuple (combined mode) hay dict (legacy mode)
    if isinstance(results, tuple):
        # Combined mode: Gom tất cả thành một index (tên file: data_for_rag)
        all_chunks, file_name = results
        
        if not all_chunks:
            print("⚠️  Không có chunks nào để xử lý")
            return
        
        print(f"\n📊 Đang tạo embeddings cho {len(all_chunks)} chunks (từ tất cả files)...")
        
        try:
            # Tạo embeddings cho tất cả chunks cùng lúc
            embeddings = embedding_system.create_embeddings(all_chunks)
            
            # Tạo một index duy nhất
            index = embedding_system.create_faiss_index(embeddings)
            
            # Lưu index và metadata
            embedding_system.save_index(index, all_chunks, file_name, output_dir)
            
            # Thống kê
            source_files = set(chunk.get("source_file", "unknown") for chunk in all_chunks)
            print(f"\n📊 Thống kê:")
            print(f"   - Tổng số chunks: {len(all_chunks):,}")
            print(f"   - Số file nguồn: {len(source_files)}")
            print(f"   - Files: {', '.join(sorted(source_files))}")
            
            # Giải phóng memory
            del embeddings
            del index
            import gc
            gc.collect()
            
        except KeyboardInterrupt:
            print(f"\n⚠️  Đã dừng quá trình (KeyboardInterrupt)")
            raise
        except Exception as e:
            print(f"❌ Lỗi khi tạo embeddings: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    else:
        # Legacy mode: Tạo index riêng cho từng file
        total_files = len(results)
        processed_files = 0
        
        for file_name, (chunks, _) in results.items():
            try:
                processed_files += 1
                print(f"\n📊 Xử lý file {processed_files}/{total_files}: {file_name}")
                
                # Tạo embeddings (tự động tối ưu batch size)
                embeddings = embedding_system.create_embeddings(chunks)
                
                # Tạo index
                index = embedding_system.create_faiss_index(embeddings)
                
                # Lưu
                embedding_system.save_index(index, chunks, file_name, output_dir)
                
                # Giải phóng memory (xóa embeddings và index sau khi đã lưu)
                del embeddings
                del index
                import gc
                gc.collect()
                
            except KeyboardInterrupt:
                print(f"\n⚠️  Đã dừng quá trình (KeyboardInterrupt)")
                print(f"   File {file_name} chưa được xử lý xong")
                print(f"   Các file đã xử lý trước đó vẫn được lưu")
                break  # Dừng xử lý các file còn lại
            except Exception as e:
                print(f"❌ Lỗi khi tạo embeddings cho {file_name}: {str(e)}")
                import traceback
                traceback.print_exc()
                # Tiếp tục với file tiếp theo thay vì dừng toàn bộ
                continue
        
        print(f"\n✅ Đã xử lý {processed_files}/{total_files} files")

if __name__ == "__main__":
    print("🚀 Hệ thống Embedding cho RAG - Legal Documents")
    print("=" * 60)
    
    # Xử lý tất cả files và gom lại thành một index
    results = process_all_files(combine_all=True)
    
    if results and results[0]:  # Kiểm tra có chunks không
        # Tạo embeddings và index (một index duy nhất cho tất cả)
        create_embeddings_for_files(results)
        
        print("\n" + "=" * 60)
        print("✅ Hoàn tất! Đã tạo một index duy nhất cho tất cả files")
        print("=" * 60)
    else:
        print("\n⚠️  Không có file nào để xử lý")
