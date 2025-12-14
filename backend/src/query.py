import google.generativeai as genai
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import faiss
import pickle
import numpy as np
import os
import sys
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from collections import Counter, defaultdict
import math
import time
from functools import lru_cache

# BM25 optimization: Use rank-bm25 for fast inverted index search
try:
    from rank_bm25 import BM25Okapi
    HAS_RANK_BM25 = True
except ImportError:
    HAS_RANK_BM25 = False
    print("⚠️  rank-bm25 not installed. Install with: pip install rank-bm25")
    print("   Falling back to manual BM25 (slower for large datasets)")

# Vietnamese tokenizer (optional, fallback to simple regex if not available)
_pyvi_tokenizer = None
try:
    try:
        from pyvi.ViTokenizer import ViTokenizer  # type: ignore
        _pyvi_tokenizer = ViTokenizer()  # Create instance once
        HAS_PYVI = True
        VIETNAMESE_TOKENIZER = "pyvi"
    except ImportError:
        try:
            from underthesea import word_tokenize  # type: ignore
            HAS_PYVI = True
            VIETNAMESE_TOKENIZER = "underthesea"
        except ImportError:
            HAS_PYVI = False
            VIETNAMESE_TOKENIZER = None
except:
    HAS_PYVI = False
    VIETNAMESE_TOKENIZER = None

if not HAS_PYVI:
    print("⚠️  Vietnamese tokenizer (pyvi/underthesea) không có. Sử dụng regex tokenizer (kém chính xác hơn).")
    print("   Cài đặt: pip install pyvi hoặc pip install underthesea")

# Get the project root directory (parent of backend/src)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CONFIG_DIR = os.path.join(PROJECT_ROOT, "backend", "config")
sys.path.insert(0, CONFIG_DIR)

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
            except:
                pass
        except:
            pass
    # Nếu không thể lấy short path, trả về path gốc
    return long_path

def safe_read_faiss_index(index_path: str):
    """
    Đọc FAISS index an toàn với đường dẫn Unicode.
    
    Args:
        index_path: Đường dẫn đến file index
    
    Returns:
        FAISS index object
    """
    index_path_obj = Path(index_path)
    
    if not index_path_obj.exists():
        raise FileNotFoundError(f"Index file không tồn tại: {index_path}")
    
    # Chuyển sang absolute path
    abs_index_path = str(index_path_obj.resolve())
    
    # Trên Windows, sử dụng short path để tránh lỗi Unicode với FAISS
    if sys.platform == 'win32':
        try:
            short_path = get_short_path(abs_index_path)
            if short_path and short_path != abs_index_path:
                abs_index_path = short_path
        except Exception as e:
            print(f"⚠️  Không thể lấy short path, dùng path gốc: {e}")
    
    try:
        return faiss.read_index(abs_index_path)
    except Exception as e:
        # Fallback: thử đọc từ thư mục tạm nếu vẫn lỗi
        if "could not open" in str(e).lower() or "unicode" in str(e).lower():
            print(f"⚠️  Lỗi Unicode với FAISS, thử giải pháp dự phòng...")
            import tempfile
            import shutil
            
            # Copy vào thư mục tạm (không có Unicode)
            temp_dir = tempfile.gettempdir()
            temp_index = os.path.join(temp_dir, f"faiss_index_{os.getpid()}.index")
            
            shutil.copy2(str(index_path_obj), temp_index)
            index = faiss.read_index(temp_index)
            os.remove(temp_index)
            
            return index
        else:
            raise

# Import configuration
try:
    from config import GEMINI_API_KEY, GEMINI_MODEL_NAME
except ImportError:
    # Fallback nếu không có config file
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-pro")

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)

# Configuration for Advanced Multi-Stage Retrieval
STAGE1_TOP_K = 100  # Số chunks lấy từ FAISS (stage 1) - tăng lên để có nhiều candidates
STAGE1_BM25_TOP_K = 100  # Số chunks lấy từ BM25 (stage 1) - hybrid search
STAGE1_HYBRID_TOP_K = 150  # Tổng số chunks sau khi merge FAISS + BM25 (deduplicate)
STAGE2_TOP_K = 30  # Số chunks sau cross-encoder re-ranking (stage 2) - tăng để có nhiều candidates
STAGE3_TOP_K = 15  # Số chunks sau keyword + metadata scoring (stage 3)
STAGE4_TOP_K = 10  # Số chunks sau diversity filtering (stage 4)
FINAL_TOP_K = 3  # Số chunks cuối cùng trả về
USE_CROSS_ENCODER = True  # Sử dụng cross-encoder để re-rank
USE_KEYWORD_BOOST = True  # Boost chunks có từ khóa
USE_METADATA_FILTER = True  # Filter theo metadata (điều khoản)
USE_BM25 = True  # Sử dụng BM25 cho hybrid search
USE_DIVERSITY_FILTER = True  # Lọc để tránh nhiều chunks từ cùng điều khoản
USE_DEDUPLICATION = True  # Loại bỏ chunks trùng lặp
BM25_K1 = 1.5  # BM25 parameter k1
BM25_B = 0.75  # BM25 parameter b

# Load bi-encoder for semantic search (fast, for initial retrieval)
bi_model = SentenceTransformer("bkai-foundation-models/vietnamese-bi-encoder",
                               device='cuda' if os.getenv("CUDA_VISIBLE_DEVICES") else 'cpu')

# Cross-encoder sẽ được load lazy (chỉ khi cần)
cross_encoder_model = None

# Load FAISS index and metadata
# Sử dụng safe_read_faiss_index để xử lý đường dẫn Unicode
def load_rag_system(index_name: str = None):
    """
    Load RAG system (FAISS index và metadata).
    Ưu tiên load index "data_for_rag" nếu có, nếu không thì load index được chỉ định hoặc "nghidinh".
    Xử lý trường hợp file không tồn tại hoặc lỗi Unicode.
    
    Args:
        index_name: Tên index cụ thể (nếu None, tự động tìm)
    
    Returns:
        Tuple (index, chunks)
    """
    # Ưu tiên load index "data_for_rag" (gom tất cả files)
    if index_name is None:
        data_for_rag_index = Path(DATA_DIR) / "data_for_rag.index"
        data_for_rag_meta = Path(DATA_DIR) / "data_for_rag_meta.pkl"
        
        if data_for_rag_index.exists() and data_for_rag_meta.exists():
            index_path = str(data_for_rag_index)
            metadata_path = data_for_rag_meta
            print("📚 Đang load index 'data_for_rag' (gom tất cả files)...")
        else:
            # Fallback về nghidinh.index (legacy)
            index_path = os.path.join(DATA_DIR, "nghidinh.index")
            metadata_path = Path(DATA_DIR) / "nghidinh_meta.pkl"
            print("📚 Đang load index 'nghidinh' (legacy)...")
    else:
        index_path = os.path.join(DATA_DIR, f"{index_name}.index")
        metadata_path = Path(DATA_DIR) / f"{index_name}_meta.pkl"
        print(f"📚 Đang load index '{index_name}'...")
    
    # Kiểm tra file có tồn tại không
    if not Path(index_path).exists():
        raise FileNotFoundError(
            f"❌ Index file không tồn tại: {index_path}\n"
            f"   Vui lòng chạy: python backend/src/embedding.py để tạo index"
        )
    
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"❌ Metadata file không tồn tại: {metadata_path}\n"
            f"   Vui lòng chạy: python backend/src/embedding.py để tạo metadata"
        )
    
    # Load index với xử lý Unicode
    index = safe_read_faiss_index(index_path)
    
    # Đọc metadata (pickle không có vấn đề với Unicode)
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    
    # Ưu tiên sử dụng chunks_full (có metadata đầy đủ) nếu có
    if "chunks_full" in metadata and metadata["chunks_full"]:
        chunks = metadata["chunks_full"]
        # Thống kê số file nguồn nếu là data_for_rag index
        if index_name == "data_for_rag" or (index_name is None and "data_for_rag" in str(metadata_path)):
            source_files = set(chunk.get("source_file", "unknown") for chunk in chunks)
            print(f"✅ RAG System loaded: {index.ntotal} vectors, {len(chunks)} chunks từ {len(source_files)} files")
            print(f"   Files: {', '.join(sorted(source_files))}")
        else:
            print(f"✅ RAG System loaded: {index.ntotal} vectors, {len(chunks)} chunks (với metadata đầy đủ)")
    else:
        # Fallback về chunks cũ (chỉ có text)
        chunks = metadata.get("chunks", [])
        print(f"✅ RAG System loaded: {index.ntotal} vectors, {len(chunks)} chunks (metadata cơ bản)")
    
    return index, chunks

# Load RAG system
try:
    index, chunks = load_rag_system()
except FileNotFoundError as e:
    print(str(e))
    print("\n⚠️  RAG System chưa được khởi tạo!")
    print("   Chạy lệnh sau để tạo index:")
    print("   python backend/src/embedding.py")
    # Tạo dummy để tránh lỗi import, nhưng sẽ báo lỗi khi gọi hàm
    index = None
    chunks = []
except Exception as e:
    print(f"❌ Lỗi khi load RAG system: {str(e)}")
    import traceback
    traceback.print_exc()
    index = None
    chunks = []

def _load_cross_encoder():
    """Lazy load cross-encoder model (chỉ load khi cần)."""
    global cross_encoder_model
    if cross_encoder_model is None:
        print("🔄 Đang load cross-encoder model cho re-ranking...")
        # Sử dụng cross-encoder tiếng Việt nếu có, hoặc fallback về model tiếng Anh
        try:
            # Thử load Vietnamese cross-encoder nếu có
            cross_encoder_model = CrossEncoder("bkai-foundation-models/vietnamese-cross-encoder",
                                              device='cuda' if os.getenv("CUDA_VISIBLE_DEVICES") else 'cpu')
        except:
            # Fallback về multilingual model
            try:
                cross_encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2",
                                                  device='cuda' if os.getenv("CUDA_VISIBLE_DEVICES") else 'cpu')
            except:
                # Nếu không có cross-encoder, sử dụng bi-encoder để re-rank
                print("⚠️  Không thể load cross-encoder, sử dụng bi-encoder để re-rank")
                cross_encoder_model = bi_model
    return cross_encoder_model

def _extract_legal_references(query: str) -> Dict:
    """Trích xuất thông tin về điều khoản, chương từ query."""
    references = {
        "article_numbers": [],
        "chapter_numbers": [],
        "clause_numbers": [],
        "keywords": []
    }
    
    # Tìm Điều
    articles = re.findall(r'Điều\s+(\d+)', query, re.IGNORECASE)
    references["article_numbers"] = [int(a) for a in articles]
    
    # Tìm Chương
    chapters = re.findall(r'Chương\s+(\d+)', query, re.IGNORECASE)
    references["chapter_numbers"] = [int(c) for c in chapters]
    
    # Tìm Khoản
    clauses = re.findall(r'Khoản\s+(\d+)', query, re.IGNORECASE)
    references["clause_numbers"] = [int(c) for c in clauses]
    
    # Tìm từ khóa quan trọng (loại bỏ stop words)
    stop_words = {'và', 'của', 'cho', 'với', 'từ', 'đến', 'trong', 'là', 'có', 'được', 'theo', 'về', 'nào', 'gì', 'thế', 'như'}
    words = re.findall(r'\b\w+\b', query.lower())
    keywords = [w for w in words if len(w) > 3 and w not in stop_words]
    references["keywords"] = keywords[:10]  # Giới hạn 10 từ khóa
    
    return references

# BM25 Index (lazy load)
_bm25_index = None
_bm25_doc_freqs = None
_bm25_idf = None
_bm25_avg_doc_len = 0.0
_bm25_rank_model = None  # rank-bm25 model (optimized)

def _build_bm25_index():
    """
    Xây dựng BM25 index từ chunks (lazy load).
    Tối ưu: Sử dụng rank-bm25 với inverted index (O(1) lookup thay vì O(N)).
    """
    global _bm25_index, _bm25_doc_freqs, _bm25_idf, _bm25_avg_doc_len, _bm25_rank_model
    
    if _bm25_index is not None:
        return
    
    if not chunks:
        return
    
    print("🔄 Đang xây dựng BM25 index...")
    
    def tokenize_vietnamese(text: str) -> List[str]:
        """Tokenize tiếng Việt sử dụng tokenizer chuyên biệt hoặc fallback."""
        if HAS_PYVI:
            try:
                if VIETNAMESE_TOKENIZER == "pyvi":
                    # pyvi tokenizer
                    tokens = _pyvi_tokenizer.tokenize(text).split()
                elif VIETNAMESE_TOKENIZER == "underthesea":
                    # underthesea tokenizer
                    tokens = word_tokenize(text)
                else:
                    # Fallback
                    tokens = re.findall(r'\b\w+\b', text.lower())
            except Exception:
                # Nếu tokenizer lỗi, fallback về regex
                tokens = re.findall(r'\b\w+\b', text.lower())
        else:
            # Fallback: regex tokenizer (kém chính xác hơn)
            tokens = re.findall(r'\b\w+\b', text.lower())
        return [t.lower() for t in tokens if t.strip()]
    
    # Tokenize tất cả chunks
    tokenized_docs = []
    for chunk in chunks:
        text = chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
        tokens = tokenize_vietnamese(text)
        tokenized_docs.append(tokens)
    
    _bm25_index = tokenized_docs
    
    # Sử dụng rank-bm25 nếu có (tối ưu với inverted index)
    if HAS_RANK_BM25:
        print("   ⚡ Sử dụng rank-bm25 (tối ưu với inverted index)")
        _bm25_rank_model = BM25Okapi(tokenized_docs, k1=BM25_K1, b=BM25_B)
        print(f"✅ BM25 index (rank-bm25) đã được xây dựng: {len(chunks)} documents")
    else:
        # Fallback về manual BM25 (chậm hơn cho large datasets)
        print("   ⚠️  Sử dụng manual BM25 (chậm hơn, nên cài rank-bm25)")
        all_terms = set()
        doc_freqs = defaultdict(int)
        total_doc_length = 0
        
        for tokens in tokenized_docs:
            all_terms.update(tokens)
            total_doc_length += len(tokens)
            # Đếm document frequency
            for term in set(tokens):
                doc_freqs[term] += 1
        
        # Tính average document length
        _bm25_avg_doc_len = total_doc_length / len(chunks) if chunks else 0
        
        # Tính IDF (Inverse Document Frequency)
        _bm25_doc_freqs = doc_freqs
        num_docs = len(chunks)
        _bm25_idf = {}
        for term in all_terms:
            df = doc_freqs.get(term, 0)
            if df > 0:
                # IDF = log((N - df + 0.5) / (df + 0.5))
                _bm25_idf[term] = math.log((num_docs - df + 0.5) / (df + 0.5))
            else:
                _bm25_idf[term] = 0.0
        
        print(f"✅ BM25 index (manual) đã được xây dựng: {num_docs} documents, {len(all_terms)} unique terms")

def _bm25_score(query_terms: List[str], doc_tokens: List[str]) -> float:
    """Tính BM25 score cho một document."""
    if _bm25_index is None or not query_terms or not doc_tokens:
        return 0.0
    
    doc_len = len(doc_tokens)
    if doc_len == 0:
        return 0.0
    
    score = 0.0
    term_freqs = Counter(doc_tokens)
    
    for term in query_terms:
        if term not in _bm25_idf:
            continue
        
        tf = term_freqs.get(term, 0)
        if tf == 0:
            continue
        
        # BM25 formula
        idf = _bm25_idf[term]
        numerator = idf * tf * (BM25_K1 + 1)
        denominator = tf + BM25_K1 * (1 - BM25_B + BM25_B * (doc_len / _bm25_avg_doc_len))
        
        score += numerator / denominator
    
    return score

def _search_bm25(query: str, top_k: int = STAGE1_BM25_TOP_K) -> List[Tuple[int, float]]:
    """
    Tìm kiếm bằng BM25 và trả về top K chunks với scores.
    Tối ưu: Sử dụng rank-bm25 với inverted index (O(1) lookup) thay vì O(N) loop.
    """
    if not USE_BM25:
        return []
    
    # Build index nếu chưa có
    _build_bm25_index()
    
    if _bm25_index is None:
        return []
    
    # Tokenize query (sử dụng cùng tokenizer với BM25 index)
    def tokenize_query(text: str) -> List[str]:
        if HAS_PYVI:
            try:
                if VIETNAMESE_TOKENIZER == "pyvi":
                    tokens = _pyvi_tokenizer.tokenize(text).split()
                elif VIETNAMESE_TOKENIZER == "underthesea":
                    tokens = word_tokenize(text)
                else:
                    tokens = re.findall(r'\b\w+\b', text.lower())
            except Exception:
                tokens = re.findall(r'\b\w+\b', text.lower())
        else:
            tokens = re.findall(r'\b\w+\b', text.lower())
        return [t.lower() for t in tokens if t.strip()]
    
    query_terms = tokenize_query(query)
    if not query_terms:
        return []
    
    # Sử dụng rank-bm25 nếu có (tối ưu với inverted index)
    if HAS_RANK_BM25 and _bm25_rank_model is not None:
        # rank-bm25 sử dụng inverted index, nhanh hơn nhiều cho large datasets
        scores = _bm25_rank_model.get_scores(query_terms)
        # Lấy top_k với scores
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]
        return results
    else:
        # Fallback: Manual BM25 (O(N) - chậm cho large datasets)
        scores = []
        for idx, doc_tokens in enumerate(_bm25_index):
            score = _bm25_score(query_terms, doc_tokens)
            if score > 0:
                scores.append((idx, score))
        
        # Sort theo score giảm dần
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

def _calculate_keyword_score(chunk_text: str, keywords: List[str]) -> float:
    """Tính điểm keyword matching (BM25-like, cải thiện)."""
    if not keywords:
        return 0.0
    
    chunk_lower = chunk_text.lower()
    keyword_counts = Counter()
    
    for keyword in keywords:
        # Tìm exact match và partial match
        count = chunk_lower.count(keyword.lower())
        keyword_counts[keyword] = count
    
    # Tính điểm: tổng số lần xuất hiện / số từ khóa (cải thiện)
    total_matches = sum(keyword_counts.values())
    if total_matches == 0:
        return 0.0
    
    # Normalize: điểm từ 0 đến 1 (cải thiện formula)
    # Sử dụng log để tránh quá phụ thuộc vào số lần xuất hiện
    score = min(1.0, math.log(1 + total_matches) / (len(keywords) * 1.5))
    return score

def _calculate_metadata_score(chunk: Dict, references: Dict) -> float:
    """Tính điểm dựa trên metadata (điều khoản, chương) - cải thiện cho legal documents."""
    score = 0.0
    
    # Boost mạnh nếu chunk thuộc điều khoản được mention (quan trọng nhất cho legal docs)
    if references["article_numbers"]:
        article_num = chunk.get("article_number")
        if article_num and article_num in references["article_numbers"]:
            score += 0.8  # Tăng từ 0.5 lên 0.8 - rất quan trọng cho legal docs
    
    # Boost nếu chunk thuộc chương được mention
    if references["chapter_numbers"]:
        chapter = chunk.get("chapter", "")
        for ch_num in references["chapter_numbers"]:
            if f"Chương {ch_num}" in chapter or f"Chương {ch_num} " in chapter:
                score += 0.4  # Tăng từ 0.3 lên 0.4
    
    # Boost nếu chunk thuộc khoản được mention
    if references["clause_numbers"]:
        clause = chunk.get("clause", "")
        for cl_num in references["clause_numbers"]:
            if f"Khoản {cl_num}" in clause:
                score += 0.3  # Tăng từ 0.2 lên 0.3
    
    return min(1.0, score)  # Normalize về 0-1

def _check_diversity_constraints(chunk: Dict, selected_chunks: List[Dict]) -> Tuple[bool, float]:
    """
    Kiểm tra diversity constraints với hard constraint và soft constraint.
    
    Hard constraint: Nếu đã có >= 2 chunks từ cùng điều khoản → skip (return True)
    Soft constraint: Nếu cosine similarity > 0.85 → skip (return True)
    
    Returns:
        (should_skip: bool, penalty: float)
        - should_skip: True nếu vi phạm hard/soft constraint (nên skip chunk này)
        - penalty: Penalty score (0.0 - 1.0) để giảm score nếu cần
    """
    if not USE_DIVERSITY_FILTER or not selected_chunks:
        return False, 0.0
    
    chunk_article = chunk.get("article_number")
    chunk_chapter = chunk.get("chapter", "")
    
    # ========== HARD CONSTRAINT: Đếm số chunks từ cùng điều khoản ==========
    same_article_count = 0
    for selected in selected_chunks:
        if chunk_article and selected.get("article_number") == chunk_article:
            same_article_count += 1
    
    # Hard constraint: Nếu đã có >= 2 chunks từ cùng điều khoản → skip
    if same_article_count >= 2:
        return True, 1.0  # Skip chunk này
    
    # ========== SOFT CONSTRAINT: Cosine similarity với chunks đã chọn ==========
    max_similarity = 0.0
    for selected in selected_chunks:
        similarity = _calculate_similarity(chunk, selected)
        max_similarity = max(max_similarity, similarity)
    
    # Soft constraint: Nếu cosine similarity > 0.85 → skip
    if max_similarity > 0.85:
        return True, max_similarity  # Skip chunk này
    
    # ========== SOFT PENALTY: Giảm score nếu có penalty nhỏ ==========
    penalty = 0.0
    for selected in selected_chunks:
        # Penalty nếu cùng điều khoản (nhưng chưa đủ 2 để hard skip)
        if chunk_article and selected.get("article_number") == chunk_article:
            penalty += 0.3
        
        # Penalty nếu cùng chương
        if chunk_chapter and selected.get("chapter", "") == chunk_chapter:
            penalty += 0.1
    
    return False, min(1.0, penalty)  # Không skip, nhưng có penalty

# Cache embeddings theo chunk_id để tránh rủi ro RAM với text dài
_embedding_cache = {}  # {chunk_id: embedding}

def _get_chunk_embedding_optimized(chunk: Dict) -> np.ndarray:
    """
    Lấy embedding cho chunk với cache tối ưu.
    Tối ưu: Cache theo chunk_id thay vì text để tránh rủi ro RAM với text dài.
    """
    # Ưu tiên: Cache theo chunk_id (nếu có trong metadata)
    chunk_id = chunk.get("chunk_idx") or chunk.get("id")
    
    if chunk_id is not None and chunk_id in _embedding_cache:
        return _embedding_cache[chunk_id]
    
    # Encode text
    text = chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
    embedding = bi_model.encode([text], normalize_embeddings=True, convert_to_numpy=True)[0]
    
    # Cache theo chunk_id nếu có
    if chunk_id is not None:
        # Giới hạn cache size (1000 entries)
        if len(_embedding_cache) >= 1000:
            # Xóa entry cũ nhất (FIFO)
            oldest_key = next(iter(_embedding_cache))
            del _embedding_cache[oldest_key]
        _embedding_cache[chunk_id] = embedding
    
    return embedding

def _calculate_similarity(chunk1: Dict, chunk2: Dict) -> float:
    """
    Tính similarity giữa 2 chunks sử dụng cosine similarity của embeddings.
    Tối ưu hơn Jaccard word-level (chính xác hơn, ít false-positive cho văn bản pháp luật).
    """
    text1 = chunk1.get("text", "") if isinstance(chunk1, dict) else str(chunk1)
    text2 = chunk2.get("text", "") if isinstance(chunk2, dict) else str(chunk2)
    
    if not text1 or not text2:
        return 0.0
    
    # Sử dụng cosine similarity của embeddings (chính xác hơn Jaccard)
    try:
        # Tối ưu: Sử dụng chunk objects để có thể cache theo chunk_id
        chunk1_dict = chunk1 if isinstance(chunk1, dict) else {"text": str(chunk1)}
        chunk2_dict = chunk2 if isinstance(chunk2, dict) else {"text": str(chunk2)}
        
        # Cache embeddings để tránh tính lại (tối ưu với chunk_id)
        emb1 = _get_chunk_embedding_optimized(chunk1_dict)
        emb2 = _get_chunk_embedding_optimized(chunk2_dict)
        
        # Cosine similarity (embeddings đã được normalize)
        similarity = np.dot(emb1, emb2)
        return float(similarity)
    except Exception:
        # Fallback về Jaccard nếu embedding lỗi
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0

def _deduplicate_chunks(chunks: List[Dict], similarity_threshold: float = 0.8) -> List[Dict]:
    """
    Loại bỏ chunks trùng lặp hoặc quá giống nhau.
    Tối ưu: Dedup theo article_number trước để giảm O(N²) complexity.
    """
    if not USE_DEDUPLICATION or len(chunks) <= 1:
        return chunks
    
    # Tối ưu: Nhóm chunks theo article_number trước (giảm số lượng so sánh)
    chunks_by_article = defaultdict(list)
    chunks_without_article = []
    
    for chunk in chunks:
        article_num = chunk.get("article_number") if isinstance(chunk, dict) else None
        if article_num is not None:
            chunks_by_article[article_num].append(chunk)
        else:
            chunks_without_article.append(chunk)
    
    deduplicated = []
    
    # Dedup trong từng nhóm article (giảm complexity)
    for article_num, article_chunks in chunks_by_article.items():
        article_dedup = []
        for chunk in article_chunks:
            is_duplicate = False
            for existing in article_dedup:
                similarity = _calculate_similarity(chunk, existing)
                if similarity >= similarity_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                article_dedup.append(chunk)
        
        deduplicated.extend(article_dedup)
    
    # Dedup chunks không có article_number
    for chunk in chunks_without_article:
        is_duplicate = False
        for existing in deduplicated:
            similarity = _calculate_similarity(chunk, existing)
            if similarity >= similarity_threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            deduplicated.append(chunk)
    
    return deduplicated

def _re_rank_with_cross_encoder(query: str, candidate_chunks: List[Dict], 
                                 top_k: int = STAGE2_TOP_K, batch_size: int = 16) -> List[Tuple[Dict, float, bool]]:
    """
    Re-rank candidates sử dụng cross-encoder (chính xác hơn nhưng chậm hơn).
    Tối ưu: Batch processing để tránh GPU memory spike và giảm latency.
    
    Returns:
        List[Tuple[Dict, float, bool]]: (chunk, score, is_cross_encoder)
        - is_cross_encoder: True nếu dùng cross-encoder thật, False nếu fallback bi-encoder
    """
    if not USE_CROSS_ENCODER or len(candidate_chunks) == 0:
        return [(chunk, 0.0, False) for chunk in candidate_chunks[:top_k]]
    
    try:
        cross_model = _load_cross_encoder()
        
        # Tạo pairs (query, chunk_text) cho cross-encoder
        pairs = [[query, chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)] 
                 for chunk in candidate_chunks]
        
        # Tính scores với batch processing
        is_cross_encoder = False
        if isinstance(cross_model, CrossEncoder):
            is_cross_encoder = True
            # Batch processing để tránh GPU memory spike
            all_scores = []
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i + batch_size]
                batch_scores = cross_model.predict(batch_pairs, show_progress_bar=False)
                all_scores.extend(batch_scores)
            
            scores = np.array(all_scores)
            # QUAN TRỌNG: Không normalize cross-encoder scores theo min-max (làm méo ranking)
            # Cross-encoder đã là relative scorer, normalize per-query làm chunk "trung bình" trông tốt giả
            # Chỉ dùng sigmoid để đưa về [0, 1] mà không làm méo relative ranking
            try:
                import torch
                # Sigmoid: giữ nguyên relative ranking, chỉ scale về [0, 1]
                scores = torch.sigmoid(torch.from_numpy(scores)).numpy()
            except ImportError:
                # Fallback: nếu không có torch, dùng scipy hoặc không normalize
                try:
                    from scipy.special import expit  # expit = sigmoid
                    scores = expit(scores)
                except ImportError:
                    # Nếu không có cả hai, chỉ clip về [0, 1] (không normalize)
                    # Cross-encoder scores thường đã trong range hợp lý
                    scores = np.clip(scores, 0, 1)
        else:
            # Fallback: sử dụng bi-encoder (cosine similarity)
            is_cross_encoder = False
            query_emb = cross_model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
            chunk_texts = [chunk.get("text", "") if isinstance(chunk, dict) else str(chunk) 
                          for chunk in candidate_chunks]
            # Batch encoding cho chunks
            chunk_embs = cross_model.encode(chunk_texts, normalize_embeddings=True, 
                                          convert_to_numpy=True, batch_size=batch_size)
            # Sử dụng util.cos_sim từ sentence-transformers (rõ ràng và tối ưu hơn)
            scores = util.cos_sim(query_emb, chunk_embs)[0].cpu().numpy()
            # Cosine similarity đã là -1 đến 1, normalize về 0-1
            scores = (scores + 1) / 2
        
        # Sort theo score giảm dần
        scored_chunks = list(zip(candidate_chunks, scores, [is_cross_encoder] * len(candidate_chunks)))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        return scored_chunks[:top_k]
    
    except Exception as e:
        print(f"⚠️  Lỗi khi re-rank với cross-encoder: {e}")
        # Fallback: trả về top_k đầu tiên với flag False
        return [(chunk, 0.0, False) for chunk in candidate_chunks[:top_k]]

def search_faiss(query, top_k=FINAL_TOP_K, use_multi_stage=True, return_metadata=False):
    """
    Advanced Multi-stage retrieval: Tìm kiếm và lọc nhiều lần để lấy chunks giàu ý nghĩa nhất.
    
    Pipeline cải thiện:
    1. Stage 1: Hybrid search (FAISS + BM25) lấy K lớn (150 chunks)
    2. Stage 2: Cross-encoder re-ranking (30 chunks)
    3. Stage 3: Keyword + Metadata scoring (15 chunks)
    4. Stage 4: Diversity filtering (10 chunks)
    5. Stage 5: Deduplication
    6. Stage 6: Final scoring và selection (top K)
    
    Args:
        query: Câu hỏi cần tìm kiếm
        top_k: Số lượng kết quả trả về cuối cùng
        use_multi_stage: Có sử dụng multi-stage retrieval không
    
    Returns:
        List các đoạn văn bản liên quan (đã được sắp xếp theo độ liên quan)
    """
    if index is None or not chunks:
        raise RuntimeError(
            "❌ RAG System chưa được khởi tạo!\n"
            "   Vui lòng chạy: python backend/src/embedding.py để tạo index"
        )
    
    if not use_multi_stage:
        # Fallback về phương pháp cũ
        # QUAN TRỌNG: Normalize query embeddings để match với index (đã normalize)
        # Normalize + Inner Product = cosine similarity chuẩn
        q_emb = bi_model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        D, I = index.search(np.array(q_emb).astype("float32"), top_k)
        results = [chunks[i] if isinstance(chunks[i], str) else chunks[i].get("text", "") 
                   for i in I[0]]
        return results
    
    # ========== STAGE 1: Hybrid Search (FAISS + BM25) ==========
    # Lấy nhiều candidates từ cả FAISS (semantic) và BM25 (keyword)
    candidate_chunks_dict = {}  # Dùng dict để deduplicate theo index
    
    # 1.1: FAISS semantic search
    # QUAN TRỌNG: Normalize query embeddings để match với index (đã normalize)
    # Normalize + Inner Product = cosine similarity chuẩn
    q_emb = bi_model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
    D, I = index.search(np.array(q_emb).astype("float32"), STAGE1_TOP_K)
    
    for idx, score in zip(I[0], D[0]):
        chunk_data = chunks[idx]
        if isinstance(chunk_data, str):
            chunk_dict = {"text": chunk_data, "faiss_score": float(score), "chunk_idx": idx}
        else:
            chunk_dict = chunk_data.copy()
            chunk_dict["faiss_score"] = float(score)
            chunk_dict["chunk_idx"] = idx
        
        candidate_chunks_dict[idx] = chunk_dict
    
    # 1.2: BM25 keyword search
    if USE_BM25:
        bm25_results = _search_bm25(query, top_k=STAGE1_BM25_TOP_K)
        for idx, bm25_score in bm25_results:
            if idx in candidate_chunks_dict:
                # Merge: thêm BM25 score vào chunk đã có
                candidate_chunks_dict[idx]["bm25_score"] = float(bm25_score)
            else:
                # Thêm chunk mới từ BM25
                chunk_data = chunks[idx]
                if isinstance(chunk_data, str):
                    chunk_dict = {"text": chunk_data, "bm25_score": float(bm25_score), "faiss_score": 0.0, "chunk_idx": idx}
                else:
                    chunk_dict = chunk_data.copy()
                    chunk_dict["bm25_score"] = float(bm25_score)
                    chunk_dict["faiss_score"] = 0.0
                    chunk_dict["chunk_idx"] = idx
                candidate_chunks_dict[idx] = chunk_dict
    
    # Convert dict về list
    candidate_chunks = list(candidate_chunks_dict.values())
    
    # ========== RECIPROCAL RANK FUSION (RRF) ==========
    # Thay vì normalize và weighted sum (bị bias theo query), dùng RRF để kết hợp ranks
    # RRF: rrf_score = 1/(k + rank_faiss) + 1/(k + rank_bm25)
    # Ưu điểm: Ổn định giữa các query, không phụ thuộc vào score distribution
    
    if candidate_chunks:
        # Tách chunks có FAISS score và BM25 score
        faiss_chunks = []
        bm25_chunks = []
        
        for chunk in candidate_chunks:
            if chunk.get("faiss_score") is not None:
                faiss_chunks.append(chunk)
            if chunk.get("bm25_score") is not None:
                bm25_chunks.append(chunk)
        
        # Sort và gán rank cho FAISS
        faiss_chunks.sort(key=lambda x: x.get("faiss_score", -999), reverse=True)
        for rank, chunk in enumerate(faiss_chunks, start=1):
            chunk["faiss_rank"] = rank
        
        # Sort và gán rank cho BM25
        bm25_chunks.sort(key=lambda x: x.get("bm25_score", 0.0), reverse=True)
        for rank, chunk in enumerate(bm25_chunks, start=1):
            chunk["bm25_rank"] = rank
        
        # Tính RRF score cho mỗi chunk
        RRF_K = 60  # Constant cho RRF (thường dùng 60)
        for chunk in candidate_chunks:
            rrf_score = 0.0
            
            # RRF từ FAISS rank
            if "faiss_rank" in chunk:
                rrf_score += 1.0 / (RRF_K + chunk["faiss_rank"])
            
            # RRF từ BM25 rank
            if "bm25_rank" in chunk:
                rrf_score += 1.0 / (RRF_K + chunk["bm25_rank"])
            
            chunk["hybrid_score"] = rrf_score
            # Lưu lại scores gốc để debug
            chunk["faiss_score_norm"] = (chunk.get("faiss_score", 0.0) + 1.0) / 2.0 if chunk.get("faiss_score", -1.0) > -1.0 else 0.0
            chunk["bm25_score_norm"] = chunk.get("bm25_score", 0.0)
    
    # Sort theo RRF score và lấy top K
    candidate_chunks.sort(key=lambda x: x.get("hybrid_score", 0.0), reverse=True)
    candidate_chunks = candidate_chunks[:STAGE1_HYBRID_TOP_K]
    
    if len(candidate_chunks) == 0:
        return []
    
    # ========== STAGE 2: Cross-Encoder Re-ranking ==========
    # Re-rank bằng cross-encoder (chính xác hơn nhưng chậm hơn)
    re_ranked = _re_rank_with_cross_encoder(query, candidate_chunks, top_k=STAGE2_TOP_K)
    
    # ========== STAGE 3: Extract References & Calculate Additional Scores ==========
    references = _extract_legal_references(query)
    
    # Tính các loại scores
    scored_chunks = []
    for chunk, cross_score, is_cross_encoder in re_ranked:
        chunk_text = chunk.get("text", "")
        
        # Cross-encoder score đã được normalize trong _re_rank_with_cross_encoder
        cross_score_norm = float(cross_score) if isinstance(cross_score, (int, float)) else 0.5
        
        # Hybrid score (FAISS + BM25) - từ RRF
        hybrid_score = chunk.get("hybrid_score", 0.0)
        
        # Keyword score
        keyword_score = 0.0
        if USE_KEYWORD_BOOST and references["keywords"]:
            keyword_score = _calculate_keyword_score(chunk_text, references["keywords"])
        
        # Metadata score (tăng weight cho legal docs)
        metadata_score = 0.0
        if USE_METADATA_FILTER:
            metadata_score = _calculate_metadata_score(chunk, references)
        
        # Stage 3 score: Combine tất cả
        # QUAN TRỌNG: Điều chỉnh weight dựa trên is_cross_encoder
        # Nếu fallback về bi-encoder, giảm weight của cross_score vì không chính xác bằng
        if is_cross_encoder:
            # Cross-encoder thật: weight cao
            cross_weight = 0.45 if references["article_numbers"] else 0.50
        else:
            # Fallback bi-encoder: giảm weight (không chính xác bằng cross-encoder)
            cross_weight = 0.20 if references["article_numbers"] else 0.25
        
        # Adaptive weights: nếu có mention điều khoản, tăng metadata weight
        if references["article_numbers"]:
            # Có mention điều khoản: metadata quan trọng hơn
            stage3_score = (
                cross_weight * cross_score_norm +
                0.25 * hybrid_score +
                0.10 * keyword_score +
                0.20 * metadata_score
            )
        else:
            # Không có mention điều khoản: semantic quan trọng hơn
            stage3_score = (
                cross_weight * cross_score_norm +
                0.30 * hybrid_score +
                0.15 * keyword_score +
                0.05 * metadata_score
            )
        
        # Lưu flag để debug
        chunk["is_cross_encoder"] = is_cross_encoder
        
        chunk["stage3_score"] = stage3_score
        chunk["cross_score"] = cross_score_norm
        chunk["hybrid_score"] = hybrid_score
        chunk["keyword_score"] = keyword_score
        chunk["metadata_score"] = metadata_score
        
        scored_chunks.append((chunk, stage3_score))
    
    # Sort theo stage3_score
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    scored_chunks = scored_chunks[:STAGE3_TOP_K]
    
    # ========== STAGE 4: Diversity Filtering ==========
    # Lọc để tránh nhiều chunks từ cùng điều khoản
    # QUAN TRỌNG: Sử dụng hard constraint và soft constraint
    selected_chunks = []
    
    for chunk, score in scored_chunks:
        # Kiểm tra xem đã đủ số lượng chưa
        if len(selected_chunks) >= STAGE4_TOP_K:
            break
        
        # Kiểm tra diversity constraints (hard + soft)
        should_skip, penalty = _check_diversity_constraints(chunk, selected_chunks)
        
        # Hard/Soft constraint: Skip chunk này nếu vi phạm
        if should_skip:
            continue  # Skip chunk này, không append
        
        # Soft penalty: Giảm score nếu có penalty nhỏ (nhưng vẫn append)
        if penalty > 0.0:
            final_score = score * (1.0 - penalty * 0.3)  # Giảm tối đa 30%
        else:
            final_score = score
        
        chunk["final_score"] = final_score
        chunk["diversity_penalty"] = penalty
        
        # CHỈ APPEND KHI ĐÃ PASS DIVERSITY CHECK
        selected_chunks.append(chunk)
    
    # selected_chunks đã được sort theo score ban đầu và filtered, không cần sort lại
    diverse_chunks = [(chunk, chunk.get("final_score", 0.0)) for chunk in selected_chunks]
    
    # ========== STAGE 5: Deduplication ==========
    # Loại bỏ chunks trùng lặp
    final_chunks = [chunk for chunk, _ in diverse_chunks]
    final_chunks = _deduplicate_chunks(final_chunks, similarity_threshold=0.8)
    
    # ========== STAGE 6: Final Selection ==========
    # Lấy top K cuối cùng
    final_chunks = final_chunks[:top_k]
    
    # ========== Return Results ==========
    if return_metadata:
        # Trả về chunks với metadata đầy đủ (bao gồm scores)
        results = []
        for chunk in final_chunks:
            result_chunk = chunk.copy()
            results.append(result_chunk)
        return results
    else:
        # Trả về text của chunks (backward compatible)
        results = []
        for chunk in final_chunks:
            results.append(chunk.get("text", ""))
        return results

def _analyze_query_type(query: str) -> Dict:
    """Phân tích loại câu hỏi để tối ưu prompt."""
    query_lower = query.lower()
    
    query_type = {
        "is_definition": False,  # "là gì", "định nghĩa"
        "is_procedure": False,  # "quy trình", "cách", "như thế nào"
        "is_comparison": False,  # "khác nhau", "so sánh", "phân biệt"
        "is_condition": False,  # "khi nào", "trường hợp nào", "điều kiện"
        "is_prohibition": False,  # "nghiêm cấm", "không được"
        "is_requirement": False,  # "yêu cầu", "phải", "cần"
        "is_article_specific": False,  # Có mention điều khoản cụ thể
        "complexity": "medium"  # simple, medium, complex
    }
    
    # Phát hiện loại câu hỏi
    if any(word in query_lower for word in ["là gì", "định nghĩa", "khái niệm", "nghĩa là"]):
        query_type["is_definition"] = True
    if any(word in query_lower for word in ["quy trình", "cách", "như thế nào", "thực hiện", "tiến hành"]):
        query_type["is_procedure"] = True
    if any(word in query_lower for word in ["khác nhau", "so sánh", "phân biệt", "khác"]):
        query_type["is_comparison"] = True
    if any(word in query_lower for word in ["khi nào", "trường hợp nào", "điều kiện", "trong trường hợp"]):
        query_type["is_condition"] = True
    if any(word in query_lower for word in ["nghiêm cấm", "không được", "cấm"]):
        query_type["is_prohibition"] = True
    if any(word in query_lower for word in ["yêu cầu", "phải", "cần", "bắt buộc"]):
        query_type["is_requirement"] = True
    
    # Phát hiện mention điều khoản
    if re.search(r'điều\s+\d+', query_lower):
        query_type["is_article_specific"] = True
    
    # Đánh giá độ phức tạp
    word_count = len(query.split())
    if word_count < 5:
        query_type["complexity"] = "simple"
    elif word_count > 15:
        query_type["complexity"] = "complex"
    
    return query_type

def _calculate_confidence_score(contexts_metadata: List[Dict], query: str) -> float:
    """Tính confidence score dựa trên quality của contexts."""
    if not contexts_metadata:
        return 0.0
    
    # Tính điểm trung bình của contexts
    avg_score = sum(c.get("final_score", 0.0) for c in contexts_metadata) / len(contexts_metadata)
    
    # Boost nếu có metadata phù hợp
    references = _extract_legal_references(query)
    metadata_match = 0.0
    
    for ctx in contexts_metadata:
        if references["article_numbers"]:
            if ctx.get("article_number") in references["article_numbers"]:
                metadata_match += 0.2
        if references["chapter_numbers"]:
            chapter = ctx.get("chapter", "")
            for ch_num in references["chapter_numbers"]:
                if f"Chương {ch_num}" in chapter:
                    metadata_match += 0.1
    
    # Normalize metadata match
    metadata_match = min(1.0, metadata_match)
    
    # Combine scores
    confidence = (0.7 * avg_score) + (0.3 * metadata_match)
    
    return min(1.0, max(0.0, confidence))

# ========== SYSTEM PROMPT (CACHE - Gửi 1 lần hoặc session-level) ==========
SYSTEM_PROMPT = """Bạn là trợ lý AI chuyên tư vấn Luật Đấu thầu Việt Nam.

Nguyên tắc:
- Trả lời trực tiếp, không xã giao, không tự giới thiệu.
- Diễn đạt chính xác, đúng thuật ngữ pháp lý.
- Không bịa thông tin ngoài ngữ cảnh được cung cấp.
- Ưu tiên rõ ràng, logic, dễ đọc.

Phong cách:
- Markdown: **bold** cho thuật ngữ, - cho danh sách, ## cho tiêu đề.
- Đoạn văn cách nhau 1 dòng trống.
- Không xuống dòng trong cùng một gạch đầu dòng.
- Diễn đạt lại, không sao chép nguyên văn."""

# ========== STYLE RULES (CACHE - Rút gọn từ 30 dòng xuống ~6 dòng) ==========
STYLE_RULES = """Quy ước trình bày:
- Không chào hỏi, không nhắc vai trò, không dẫn nhập.
- Không sao chép nguyên văn, phải diễn đạt lại.
- Không xuống dòng giữa các câu trong cùng một gạch đầu dòng.
- Nếu thiếu thông tin, nói rõ là không có trong ngữ cảnh."""

def _get_task_instruction(query_type: Dict) -> str:
    """
    Tạo task-specific instruction ngắn gọn dựa trên query type.
    Rút gọn từ nhiều dòng xuống 1-2 dòng meta-instruction.
    """
    instructions = []
    
    if query_type.get("is_definition"):
        instructions.append("Đưa ra định nghĩa rõ ràng, so sánh nếu có nhiều định nghĩa.")
    
    if query_type.get("is_procedure"):
        instructions.append("Trình bày quy trình từng bước, nêu điều kiện ở mỗi bước.")
    
    if query_type.get("is_comparison"):
        instructions.append("So sánh chi tiết điểm giống/khác, có ví dụ minh họa.")
    
    if query_type.get("is_condition"):
        instructions.append("Liệt kê đầy đủ điều kiện, phân loại rõ ràng.")
    
    if query_type.get("is_prohibition"):
        instructions.append("Liệt kê hành vi bị nghiêm cấm, nêu hậu quả pháp lý.")
    
    if query_type.get("is_requirement"):
        instructions.append("Liệt kê yêu cầu, phân loại theo mức độ bắt buộc.")
    
    if query_type.get("is_article_specific"):
        instructions.append("Tập trung vào điều khoản được đề cập, giải thích chi tiết.")
    
    return " ".join(instructions) if instructions else "Trình bày theo quy trình, liệt kê điều kiện nếu có."

def _create_dynamic_prompt(query: str, contexts_metadata: List[Dict], 
                          query_type: Dict, conversation_history: List[Dict] = None) -> str:
    """
    Tạo prompt động theo nguyên tắc 3 lớp:
    1. System Prompt (cache) - đã định nghĩa ở trên
    2. Style Rules (cache) - đã định nghĩa ở trên
    3. Task Prompt (động) - chỉ phần này thay đổi mỗi query
    
    Tối ưu token economics: Giảm từ ~200 tokens xuống ~80 tokens cho phần cố định.
    """
    
    # Xây dựng context với citation
    context_parts = []
    for i, ctx in enumerate(contexts_metadata, 1):
        text = ctx.get("text", "")
        citation = []
        
        # Tạo citation từ metadata
        if ctx.get("article"):
            citation.append(ctx["article"])
        if ctx.get("clause"):
            citation.append(ctx["clause"])
        if ctx.get("chapter"):
            citation.append(ctx["chapter"])
        
        citation_str = " - ".join(citation) if citation else "Văn bản pháp luật"
        
        context_parts.append(f"[Nguồn {i}: {citation_str}]\n{text}")
    
    context_text = "\n\n".join(context_parts)
    
    # Format conversation history nếu có
    history_section = ""
    if conversation_history:
        history_text = _format_conversation_history(conversation_history, max_messages=8)
        if history_text:
            history_section = f"\n### Lịch sử cuộc trò chuyện:\n{history_text}\n"
    
    # Task-specific instruction (rút gọn từ nhiều dòng xuống 1-2 dòng)
    task_instruction = _get_task_instruction(query_type)
    
    # Task Prompt (NGẮN - ĐỘNG) - chỉ phần này gửi mỗi query
    # Giảm từ ~200 tokens xuống ~80 tokens cho phần cố định
    prompt = f"""{SYSTEM_PROMPT}

{STYLE_RULES}

### Ngữ cảnh pháp luật:
{context_text}{history_section}### Câu hỏi:
{query}

### Yêu cầu:
{task_instruction}

### Trả lời:"""
    
    return prompt

def _format_conversation_history(history: List[Dict], max_messages: int = 10) -> str:
    """
    Format conversation history để đưa vào prompt.
    Chỉ lấy N messages gần nhất để tránh prompt quá dài.
    
    Args:
        history: List of messages với format [{"role": "user"|"model", "content": "..."}]
        max_messages: Số lượng messages tối đa (mặc định 10)
    
    Returns:
        Formatted conversation history string
    """
    if not history:
        return ""
    
    # Chỉ lấy N messages gần nhất
    recent_history = history[-max_messages:] if len(history) > max_messages else history
    
    history_lines = []
    for msg in recent_history:
        role = msg.get("role", "")
        content = msg.get("content", "").strip()
        
        if not content:
            continue
        
        # Format role name
        if role == "user":
            role_name = "Người dùng"
        elif role == "model" or role == "assistant":
            role_name = "Trợ lý"
        else:
            role_name = role
        
        history_lines.append(f"{role_name}: {content}")
    
    if history_lines:
        return "\n".join(history_lines)
    return ""

def ask_sth(query, return_metadata=False, use_advanced=True, conversation_history=None):
    """
    Trả lời câu hỏi sử dụng RAG (Retrieval-Augmented Generation) với Gemini API.
    Phiên bản cải tiến với dynamic prompt, citation, confidence score, và conversation history.
    
    Args:
        query: Câu hỏi cần trả lời
        return_metadata: Có trả về metadata (sources, confidence) không
        use_advanced: Có sử dụng advanced features (dynamic prompt, citation) không
        conversation_history: List of previous messages [{"role": "user"|"model", "content": "..."}]
    
    Returns:
        Nếu return_metadata=False: str - Câu trả lời
        Nếu return_metadata=True: dict - {
            "answer": str,
            "sources": List[Dict],
            "confidence": float,
            "query_type": Dict
        }
    """
    try:
        # Tìm kiếm context với metadata
        contexts_metadata = search_faiss(query, top_k=3, use_multi_stage=True, return_metadata=True)
        
        if not contexts_metadata:
            if return_metadata:
                return {
                    "answer": "Xin lỗi, tôi không tìm thấy thông tin liên quan trong cơ sở dữ liệu để trả lời câu hỏi này.",
                    "sources": [],
                    "confidence": 0.0,
                    "query_type": {}
                }
            return "Xin lỗi, tôi không tìm thấy thông tin liên quan trong cơ sở dữ liệu để trả lời câu hỏi này."
        
        # Phân tích loại câu hỏi
        query_type = _analyze_query_type(query)
        
        # Tính confidence score
        confidence = _calculate_confidence_score(contexts_metadata, query)
        
        if use_advanced:
            # Tạo dynamic prompt với conversation history
            prompt = _create_dynamic_prompt(query, contexts_metadata, query_type, conversation_history)
        else:
            # Fallback về prompt cũ (cải thiện) với conversation history
            context_text = "\n".join([ctx.get("text", "") for ctx in contexts_metadata])
            
            # Format conversation history nếu có
            history_section = ""
            if conversation_history:
                history_text = _format_conversation_history(conversation_history, max_messages=8)
                if history_text:
                    history_section = f"""
### Lịch sử cuộc trò chuyện (để hiểu ngữ cảnh):
{history_text}

"""
            
            prompt = f"""Bạn là một chuyên gia tư vấn về Luật Đấu thầu Việt Nam. 
Hãy sử dụng thông tin ngữ cảnh sau đây để trả lời câu hỏi một cách chính xác, ngắn gọn và súc tích.

### Ngữ cảnh (từ các văn bản pháp luật):
{context_text}
{history_section}### Câu hỏi hiện tại:
{query}

### Yêu cầu:
- Trả lời TRỰC TIẾP, KHÔNG giới thiệu bản thân, KHÔNG nói "Chào bạn", "với vai trò", "xin trả lời", "tôi là"
- Bắt đầu NGAY với nội dung trả lời, KHÔNG nói "dựa trên thông tin được cung cấp" hay "theo ngữ cảnh"
- Diễn đạt bằng lời văn tự nhiên, dễ hiểu
- Sử dụng Markdown: **bold** cho thuật ngữ, - cho list, ## cho tiêu đề phụ
- **PHẢI XUỐNG DÒNG**: Mỗi đoạn văn phải cách nhau bằng 2 dòng trống (\\n\\n)
- **KHÔNG XUỐNG DÒNG**: Không xuống dòng sau dấu chấm trong cùng một câu hoặc trong danh sách markdown (gạch đầu dòng)
- Sử dụng tiêu đề ## để phân chia các phần lớn (ví dụ: ## Giải thích chi tiết, ## Kết luận)
- Không sao chép nguyên văn mà diễn đạt lại
- Kết thúc bằng một câu kết luận rõ ràng
- Nếu không có thông tin trong ngữ cảnh, hãy nói rõ

### Trả lời (BẮT ĐẦU NGAY với nội dung, KHÔNG giới thiệu, PHẢI XUỐNG DÒNG giữa các đoạn):"""
        
        # Gửi request đến Gemini API với retry và exponential backoff
        max_retries = 3
        base_delay = 1.0  # seconds
        
        answer = None
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # Gửi request đến Gemini API
                response = gemini_model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": 0.7,
                        "max_output_tokens": 8192,
                    }
                )
                answer = response.text.strip()
                break  # Thành công, thoát khỏi retry loop
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                # Chỉ retry cho các lỗi có thể recover (network, rate limit, timeout)
                retryable_errors = ["timeout", "rate limit", "quota", "network", "connection", "503", "429", "500"]
                is_retryable = any(err in error_str for err in retryable_errors)
                
                if attempt < max_retries - 1 and is_retryable:
                    # Exponential backoff: delay = base_delay * (2^attempt)
                    delay = base_delay * (2 ** attempt)
                    print(f"⚠️  Lỗi khi gọi Gemini API (attempt {attempt + 1}/{max_retries}): {str(e)}")
                    print(f"   Retry sau {delay:.1f} giây...")
                    time.sleep(delay)
                elif not is_retryable:
                    # Lỗi không thể retry (ví dụ: invalid API key, bad request)
                    print(f"❌ Lỗi không thể retry: {str(e)}")
                    raise
                else:
                    # Lần thử cuối cùng thất bại
                    print(f"❌ Lỗi khi gọi Gemini API sau {max_retries} lần thử: {str(e)}")
                    raise Exception(f"Không thể lấy response từ Gemini API sau {max_retries} lần thử: {str(e)}")
        
        if answer is None:
            raise Exception(f"Không thể lấy response từ Gemini API: {last_error}")
        
        # Post-processing: Loại bỏ các cụm từ không cần thiết
        unwanted_phrases = [
            "chào bạn, với vai trò là một chuyên gia tư vấn về luật đấu thầu việt nam, tôi xin trả lời",
            "chào bạn, với vai trò là",
            "với vai trò là một chuyên gia",
            "với vai trò là",
            "với vai trò",
            "tôi xin trả lời",
            "xin trả lời",
            "tôi là",
            "dựa trên thông tin được cung cấp",
            "dựa trên thông tin",
            "theo ngữ cảnh được cung cấp",
            "theo ngữ cảnh",
            "dựa trên ngữ cảnh",
            "theo thông tin",
            "dựa vào thông tin",
            "theo các thông tin",
            "dựa vào ngữ cảnh",
        ]
        
        # Loại bỏ các cụm từ không cần thiết ở đầu câu
        answer_lower = answer.lower()
        for phrase in unwanted_phrases:
            # Check cả startswith và tìm trong câu (cho các cụm từ dài)
            if answer_lower.startswith(phrase):
                # Tìm vị trí kết thúc của phrase và lấy phần sau
                idx = len(phrase)
                # Bỏ qua dấu phẩy, dấu chấm, dấu hai chấm sau phrase
                while idx < len(answer) and answer[idx] in [',', '.', ':', ';', ' ']:
                    idx += 1
                answer = answer[idx:].strip()
                # Viết hoa chữ cái đầu nếu cần
                if answer and answer[0].islower():
                    answer = answer[0].upper() + answer[1:]
                break
            # Nếu phrase xuất hiện ở đầu câu (sau dấu chấm hoặc xuống dòng)
            elif phrase in answer_lower[:200]:  # Check trong 200 ký tự đầu
                # Tìm vị trí của phrase
                phrase_idx = answer_lower.find(phrase)
                if phrase_idx >= 0:
                    # Kiểm tra xem có phải ở đầu câu không (sau dấu chấm, xuống dòng, hoặc đầu chuỗi)
                    before_phrase = answer_lower[:phrase_idx].strip()
                    if not before_phrase or before_phrase.endswith('.') or before_phrase.endswith('\n'):
                        # Loại bỏ phrase và lấy phần sau
                        idx = phrase_idx + len(phrase)
                        while idx < len(answer) and answer[idx] in [',', '.', ':', ';', ' ']:
                            idx += 1
                        answer = answer[:phrase_idx].strip() + " " + answer[idx:].strip()
                        answer = answer.strip()
                        # Viết hoa chữ cái đầu nếu cần
                        if answer and answer[0].islower():
                            answer = answer[0].upper() + answer[1:]
                        break
        
        # Post-processing: Đảm bảo xuống dòng đúng cách
        # Đảm bảo có 2 dòng trống trước các tiêu đề markdown (nếu chưa có)
        answer = re.sub(r'([^\n])\n(##\s+)', r'\1\n\n\2', answer)
        # Đảm bảo có xuống dòng sau các tiêu đề markdown
        answer = re.sub(r'(##\s+[^\n]+)\n([^\n#])', r'\1\n\n\2', answer)
        
        # Sửa lỗi: Loại bỏ xuống dòng không mong muốn trong danh sách markdown (bullet points)
        # Pattern 1: List item có dấu chấm + 2 dòng trống + list item tiếp theo
        # → Chuyển thành: list item + dấu chấm + 1 dòng trống + list item tiếp theo
        answer = re.sub(r'(\n\s*[-*]\s+[^\n]+)\.\n\n(\s*[-*]\s+)', r'\1.\n\2', answer)
        
        # Pattern 2: List item có dấu chấm + 2 dòng trống + câu mới (không phải list item)
        # → Chuyển thành: list item + dấu chấm + space + câu mới
        answer = re.sub(r'(\n\s*[-*]\s+[^\n]+)\.\n\n([A-ZĐ])', r'\1. \2', answer)
        
        # Pattern 3: List item (không có dấu chấm) + 2 dòng trống + list item tiếp theo
        # → Chuyển thành: list item + 1 dòng trống + list item tiếp theo
        answer = re.sub(r'(\n\s*[-*]\s+[^\n]+)\n\n(\s*[-*]\s+)', r'\1\n\2', answer)
        
        # Pattern 4: List item + xuống dòng + text tiếp (trong cùng một item, có indent)
        # Ví dụ: "- text\n\n  tiếp tục" → "- text tiếp tục"
        answer = re.sub(r'(\n\s*[-*]\s+[^\n]+)\n\n(\s{2,}[^\n-*#])', r'\1 \2', answer)
        
        # Pattern 5: List item có dấu chấm/phẩy + xuống dòng + text tiếp (trong cùng item)
        # Ví dụ: "- text.\n\n  tiếp tục" → "- text. tiếp tục"
        answer = re.sub(r'(\n\s*[-*]\s+[^\n]+[.,;])\n\n(\s+[^\n-*#])', r'\1 \2', answer)
        
        # Pattern 6: List item + xuống dòng ngay (không có dấu câu) + text tiếp (trong cùng item)
        # Ví dụ: "- text\n\n  tiếp tục" → "- text tiếp tục"
        answer = re.sub(r'(\n\s*[-*]\s+[^\n]+[^.,;])\n\n(\s+[^\n-*#])', r'\1 \2', answer)
        
        # Pattern 7: List item bị xuống dòng ngay sau dấu gạch đầu dòng (lỗi format)
        # Ví dụ: "-\ntext" → "- text"
        answer = re.sub(r'(\n\s*[-*])\n(\s*[^\n])', r'\1 \2', answer)
        
        # Loại bỏ các dòng trống thừa (hơn 2 dòng trống liên tiếp)
        answer = re.sub(r'\n{3,}', r'\n\n', answer)
        
        # Loại bỏ khoảng trắng thừa ở đầu và cuối
        answer = answer.strip()
        
        # Chuẩn bị sources với citation
        sources = []
        for ctx in contexts_metadata:
            source_info = {
                "text": ctx.get("text", "")[:200] + "..." if len(ctx.get("text", "")) > 200 else ctx.get("text", ""),
                "citation": []
            }
            
            if ctx.get("article"):
                source_info["citation"].append(ctx["article"])
            if ctx.get("clause"):
                source_info["citation"].append(ctx["clause"])
            if ctx.get("chapter"):
                source_info["citation"].append(ctx["chapter"])
            
            source_info["score"] = ctx.get("final_score", 0.0)
            sources.append(source_info)
        
        if return_metadata:
            return {
                "answer": answer,
                "sources": sources,
                "confidence": confidence,
                "query_type": query_type,
                "context_count": len(contexts_metadata)
            }
        else:
            return answer
            
    except Exception as e:
        error_msg = f"❌ Lỗi khi gọi Gemini API: {str(e)}"
        if return_metadata:
            return {
                "answer": error_msg,
                "sources": [],
                "confidence": 0.0,
                "query_type": {},
                "error": str(e)
            }
        return error_msg