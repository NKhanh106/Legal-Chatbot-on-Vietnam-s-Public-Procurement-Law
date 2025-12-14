"""
Advanced embedding system for RAG (Retrieval-Augmented Generation).
Tối ưu hóa cho tài liệu pháp luật Việt Nam.
"""
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
EMBEDDING_MODEL = "bkai-foundation-models/vietnamese-bi-encoder"
CHUNK_SIZE = 500  # Số từ tối ưu cho legal documents (tăng để giữ nguyên Điều khi có thể)
CHUNK_OVERLAP = 50  # Overlap để giữ context (giảm để tránh trùng lặp quá nhiều)
MIN_CHUNK_SIZE = 20  # Chunk tối thiểu (tăng để tránh chunks quá ngắn, không có ý nghĩa)
MAX_CHUNK_SIZE = 1000  # Chunk tối đa (tăng để giữ nguyên các Điều dài)
DEVICE = 'cuda' if os.getenv("CUDA_VISIBLE_DEVICES") else 'cpu'

# FAISS Index Configuration
USE_IVF_INDEX = True  # Sử dụng IndexIVFFlat cho hiệu suất tốt hơn với large datasets
USE_COMPRESSED_INDEX = False  # Sử dụng IndexPQ/IndexIVFPQ cho datasets rất lớn (>100k chunks) - giảm memory
N_CLUSTERS = 100  # Số clusters cho IVF index (tăng nếu có nhiều chunks)
PQ_M = 64  # Số subquantizers cho Product Quantization (chỉ dùng khi USE_COMPRESSED_INDEX=True)
PQ_BITS = 8  # Số bits cho mỗi subquantizer (chỉ dùng khi USE_COMPRESSED_INDEX=True)

# Performance Optimization
AUTO_BATCH_SIZE = True  # Tự động điều chỉnh batch size
DEFAULT_BATCH_SIZE = 32  # Batch size mặc định
MAX_BATCH_SIZE = 128  # Batch size tối đa (cho GPU mạnh)
MIN_BATCH_SIZE = 8  # Batch size tối thiểu (cho CPU hoặc GPU yếu)
VALIDATE_CHUNKS = True  # Validate chunks trước khi embed

class LegalDocumentChunker:
    """
    Chunker chuyên biệt cho tài liệu pháp luật.
    Chunk theo cấu trúc pháp luật: Chương > Mục > Điều > Khoản > Điểm
    """
    
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
        """
        # Tách thành các đoạn (theo dòng trống hoặc markdown headers)
        # Phát hiện các ranh giới cấu trúc pháp luật (Khoản, Điểm)
        lines = content.split("\n")
        paragraphs = []
        current_para = []
        
        for line in lines:
            line_stripped = line.strip()
            
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
                        # Tạo overlap nhỏ (chỉ 1-2 câu cuối)
                        overlap = self._get_overlap_text(current_chunk_parts, max_sentences=2)
                        current_chunk_parts = [overlap, sub_chunk] if overlap else [sub_chunk]
                        current_length = len(" ".join(current_chunk_parts).split())
                    else:
                        current_chunk_parts.append(sub_chunk)
                        current_length += sub_words
            else:
                # Kiểm tra xem có thể thêm vào chunk hiện tại không
                # Ưu tiên: Nếu là Điều và vẫn trong giới hạn hợp lý, giữ nguyên
                is_article = hierarchy.get("article") is not None
                can_fit = current_length + para_words <= self.chunk_size
                
                # Nếu là Điều và có thể fit, luôn thêm vào (giữ nguyên Điều)
                if is_article and can_fit:
                    current_chunk_parts.append(para)
                    current_length += para_words
                # Nếu không fit và đã có chunk, tạo chunk mới
                elif current_length + para_words > self.chunk_size and current_chunk_parts:
                    chunks.append(self._create_chunk_with_hierarchy(
                        current_chunk_parts, hierarchy, section_header, legal_header, metadata
                    ))
                    # Tạo overlap nhỏ (chỉ 1-2 câu cuối)
                    overlap = self._get_overlap_text(current_chunk_parts, max_sentences=2)
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
                    overlap_text = self._get_overlap_text(current_chunk)
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
            # Tăng threshold từ 10 lên 15 từ để bỏ qua các đoạn quá ngắn
            if para and len(para.split()) >= 15:
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
    
    def _get_overlap_text(self, chunk: List[str], max_sentences: int = 2) -> str:
        """
        Lấy phần overlap từ chunk cuối cùng.
        
        Args:
            chunk: List các phần text
            max_sentences: Số câu tối đa để lấy làm overlap (mặc định: 2 để giảm trùng lặp)
        
        Returns:
            Text overlap (1-2 câu cuối, tối đa 150 từ)
        """
        if not chunk:
            return ""
        
        # Lấy câu cuối cùng
        last_para = chunk[-1]
        sentences = re.split(r'([.!?]+[)\]\}"\']*\s+)', last_para)
        combined = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                combined.append(sentences[i] + sentences[i + 1])
        
        # Lấy max_sentences câu cuối (mặc định 2)
        overlap_sentences = combined[-min(max_sentences, len(combined)):]
        overlap_text = " ".join(overlap_sentences).strip()
        
        # Giới hạn độ dài overlap (tối đa 150 từ để tránh overlap quá lớn)
        overlap_words = overlap_text.split()
        if len(overlap_words) > 150:
            overlap_text = " ".join(overlap_words[-150:])
        
        return overlap_text
    
    def _create_chunk_with_hierarchy(self, text_parts: List[str], hierarchy: Dict,
                                     section_header: str, legal_header: str,
                                     metadata: Dict = None) -> Dict:
        """Tạo chunk với metadata phong phú về cấu trúc pháp luật."""
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
        }
        
        # Thêm metadata từ document level
        if metadata:
            chunk_data.update({
                "source": metadata.get("source", ""),
                "file_path": metadata.get("file_path", ""),
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
    
    def __init__(self, model_name: str = EMBEDDING_MODEL, device: str = DEVICE):
        print(f"🔄 Đang load embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"✅ Model loaded. Dimension: {self.dimension}")
        
        self.chunker = LegalDocumentChunker()
    
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
        # Validate chunks
        validated_chunks = self._validate_chunks(chunks)
        
        if not validated_chunks:
            raise ValueError("Không có chunks hợp lệ để tạo embeddings")
        
        texts = [chunk["text"] for chunk in validated_chunks]
        
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
        Tạo FAISS index tối ưu với auto-tuning.
        
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
        
        # QUAN TRỌNG: Embeddings đã được normalize, nên dùng Inner Product (IP) thay vì L2
        # Normalize + IP = cosine similarity chuẩn (tối ưu nhất)
        # L2 + normalize = gần đúng nhưng không tối ưu
        
        # Quyết định loại index dựa trên kích thước dataset
        if n_vectors < 1000:
            # Small dataset: sử dụng IndexFlatIP (exact search, nhanh cho datasets nhỏ)
            print("   Sử dụng IndexFlatIP (Inner Product - exact search, phù hợp cho datasets nhỏ)")
            print("   ⚡ Tối ưu: Normalize + IP = cosine similarity chuẩn")
            index = faiss.IndexFlatIP(dimension)
            index.add(embeddings.astype('float32'))
            print("   ✅ IndexFlatIP đã được tạo")
        elif USE_COMPRESSED_INDEX and n_vectors > 100000:
            # Very large dataset (>100k): sử dụng IndexIVFPQ (compressed, tiết kiệm memory)
            # Lưu ý: IndexIVFPQ có thể không hỗ trợ Inner Product tốt, cần kiểm tra
            print(f"   Sử dụng IndexIVFPQ (compressed index - phù hợp cho datasets rất lớn >100k)")
            print(f"   PQ parameters: m={PQ_M}, bits={PQ_BITS}")
            print("   ⚠️  Lưu ý: IndexIVFPQ với IP có thể không tối ưu, cân nhắc dùng IndexIVFFlat với IP")
            
            # Vẫn dùng IP quantizer cho consistency
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFPQ(quantizer, dimension, optimal_clusters, PQ_M, PQ_BITS)
            index.metric_type = faiss.METRIC_INNER_PRODUCT
            
            # Train index
            print("   🔄 Đang train index...")
            index.train(embeddings.astype('float32'))
            
            # Add vectors
            index.add(embeddings.astype('float32'))
            
            # Tự động điều chỉnh nprobe
            index.nprobe = min(optimal_clusters // 4, 50)
            
            print(f"   ✅ IndexIVFPQ đã được tạo và train (nprobe={index.nprobe}, metric=INNER_PRODUCT)")
            print(f"   💾 Memory: ~{n_vectors * dimension * PQ_BITS / 8 / (1024**2):.1f} MB (compressed)")
        elif use_ivf and n_vectors > optimal_clusters * 10:
            # Medium to large dataset: sử dụng IndexIVFFlat với Inner Product (cân bằng tốt)
            print(f"   Sử dụng IndexIVFFlat với {optimal_clusters} clusters (phù hợp cho datasets vừa và lớn)")
            print("   ⚡ Tối ưu: Normalize + IP = cosine similarity chuẩn")
            
            # Sử dụng Inner Product quantizer và metric
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, optimal_clusters, faiss.METRIC_INNER_PRODUCT)
            
            # Train index
            print("   🔄 Đang train index...")
            index.train(embeddings.astype('float32'))
            
            # Add vectors
            index.add(embeddings.astype('float32'))
            
            # Tự động điều chỉnh nprobe dựa trên số lượng vectors
            # nprobe càng lớn thì search càng chính xác nhưng chậm hơn
            if n_vectors < 10000:
                index.nprobe = min(optimal_clusters // 2, 20)  # Medium dataset
            else:
                index.nprobe = min(optimal_clusters // 4, 50)  # Large dataset: balance speed/accuracy
            
            print(f"   ✅ IndexIVFFlat đã được tạo và train (nprobe={index.nprobe}, metric=INNER_PRODUCT)")
        else:
            # Fallback: IndexFlatIP
            print("   Sử dụng IndexFlatIP (Inner Product - exact search)")
            print("   ⚡ Tối ưu: Normalize + IP = cosine similarity chuẩn")
            index = faiss.IndexFlatIP(dimension)
            index.add(embeddings.astype('float32'))
            print("   ✅ IndexFlatIP đã được tạo")
        
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
            "chunks": [chunk["text"] for chunk in chunks],  # Chỉ lưu text cho backward compatibility
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
    
    embedding_system = EmbeddingSystem()
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
