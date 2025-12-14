# Backend Source Code Documentation

Thư mục này chứa các module core của hệ thống RAG (Retrieval-Augmented Generation) cho Legal Chatbot.

## 📁 Tổng quan các file

| File | Công dụng | Công nghệ chính | Độ phức tạp |
|------|-----------|-----------------|-------------|
| `query.py` | Xử lý query và RAG pipeline | FAISS, BM25, Cross-Encoder, Gemini API | ⭐⭐⭐⭐⭐ |
| `embedding.py` | Tạo embeddings và FAISS index | Sentence Transformers, FAISS | ⭐⭐⭐⭐ |
| `preprocess.py` | Tiền xử lý text và markdown | Regex, Text processing | ⭐⭐⭐ |
| `read_pdf.py` | Đọc PDF với OCR | pytesseract, pdf2image, PIL | ⭐⭐⭐⭐ |
| `read_word.py` | Đọc Word documents | python-docx, docx2txt | ⭐⭐⭐ |
| `correction.py` | Sửa lỗi OCR | Regex, Pattern matching | ⭐⭐⭐ |
| `deploy.py` | Utilities cho deployment | - | ⭐⭐⭐ |

---

## 📄 Chi tiết từng file

### 1. `query.py` - RAG Query Processing Engine

**Công dụng:**
- Module chính xử lý query và thực hiện RAG pipeline
- Multi-stage retrieval với 6 giai đoạn tối ưu
- Tích hợp Gemini API để generate response

**Công nghệ sử dụng:**
- **FAISS**: Vector similarity search (Inner Product metric)
- **rank-bm25**: Fast BM25 keyword search với inverted index (O(1) lookup)
- **Sentence Transformers**: Bi-encoder và Cross-encoder cho semantic search
- **Google Gemini API**: LLM generation với retry logic
- **Reciprocal Rank Fusion (RRF)**: Kết hợp FAISS và BM25 ranks

**Phương thức hoạt động:**

1. **Stage 1: Hybrid Search (FAISS + BM25)**
   - FAISS semantic search: Tìm chunks tương tự về ngữ nghĩa
   - BM25 keyword search: Tìm chunks có từ khóa chính xác
   - RRF fusion: Kết hợp ranks từ 2 phương pháp
   - Output: ~150 candidates

2. **Stage 2: Cross-Encoder Re-ranking**
   - Re-rank candidates bằng cross-encoder (chính xác hơn)
   - Batch processing (16 items/batch) để tránh GPU memory spike
   - Fallback to bi-encoder với weight adjustment
   - Output: ~30 top candidates

3. **Stage 3: Metadata & Keyword Scoring**
   - Tính keyword score (từ khóa trong query)
   - Tính metadata score (điều khoản, chương được mention)
   - Adaptive weights dựa trên query type
   - Output: ~15 scored chunks

4. **Stage 4: Diversity Filtering**
   - Hard constraint: Skip nếu ≥2 chunks từ cùng điều khoản
   - Soft constraint: Skip nếu cosine similarity > 0.85
   - Output: ~10 diverse chunks

5. **Stage 5: Deduplication**
   - Nhóm theo article_number để giảm complexity
   - Cosine similarity của embeddings (chính xác hơn Jaccard)
   - Output: Deduplicated chunks

6. **Stage 6: Final Selection & Generation**
   - Chọn top K chunks cuối cùng
   - Tạo prompt với 3-layer architecture (System/Style/Task)
   - Gọi Gemini API với retry & exponential backoff
   - Streaming response về frontend

**Các hàm chính:**
- `load_rag_system()`: Load FAISS index và chunks
- `search_faiss()`: Main search function với multi-stage pipeline
- `ask_sth()`: Generate answer từ query
- `_re_rank_with_cross_encoder()`: Cross-encoder re-ranking với batching
- `_check_diversity_constraints()`: Hard/soft diversity filtering
- `_create_dynamic_prompt()`: Tạo optimized prompt (3-layer)

**Tối ưu hóa:**
- RRF thay vì normalize + weighted sum (stable ranking)
- Batch processing cho cross-encoder
- Cached embeddings theo chunk_id
- Retry logic với exponential backoff
- Prompt optimization (60% token reduction)

---

### 2. `embedding.py` - Embedding System & FAISS Index

**Công dụng:**
- Tạo embeddings từ text chunks
- Xây dựng và quản lý FAISS index
- Tối ưu chunking cho văn bản pháp luật

**Công nghệ sử dụng:**
- **Sentence Transformers**: Vietnamese Bi-Encoder model
- **FAISS**: Vector index (IndexFlatIP, IndexIVFFlat, IndexIVFPQ)
- **NumPy**: Array operations
- **Pickle**: Serialize metadata

**Phương thức hoạt động:**

1. **Chunking Strategy:**
   - Chunk size: 500 words (optimized cho legal documents)
   - Chunk overlap: 50 words (minimal để tránh duplication)
   - Tôn trọng cấu trúc pháp luật: Chương, Điều, Khoản, Điểm
   - Ưu tiên giữ nguyên "Điều" (không split)
   - Chỉ split khi vượt quá MAX_CHUNK_SIZE (1000 words)

2. **Embedding Creation:**
   - Model: Vietnamese Bi-Encoder (normalized embeddings)
   - Batch encoding để tối ưu performance
   - Normalize embeddings trước khi lưu vào FAISS

3. **FAISS Index:**
   - Metric: Inner Product (vì embeddings đã normalize)
   - Inner Product với normalized vectors = cosine similarity
   - Hỗ trợ IndexFlatIP, IndexIVFFlat, IndexIVFPQ

**Các class chính:**
- `LegalDocumentChunker`: Chunking tối ưu cho legal documents
- `EmbeddingSystem`: Quản lý embedding và FAISS index

**Các hàm chính:**
- `process_all_files()`: Xử lý tất cả text files
- `create_embeddings_for_files()`: Tạo embeddings và index
- `_chunk_section_content()`: Chunking với legal structure awareness

**Output:**
- `data/data_for_rag.index`: FAISS vector index
- `data/data_for_rag_meta.pkl`: Metadata (chunks, sources, embeddings)

---

### 3. `preprocess.py` - Text Preprocessing

**Công dụng:**
- Làm sạch markdown files
- Parse và restructure markdown
- Chuyển markdown → text files cho embedding

**Công nghệ sử dụng:**
- **Regex**: Pattern matching và text cleaning
- **Pathlib**: File path handling
- **OS**: File system operations

**Phương thức hoạt động:**

1. **Clean Markdown:**
   - Loại bỏ số trang, header/footer
   - Loại bỏ markdown artifacts (code blocks, images)
   - Chuẩn hóa khoảng trắng

2. **Parse Structure:**
   - Tách markdown thành các sections
   - Extract metadata (Chương, Điều, Khoản, Điểm)
   - Preserve legal document structure

3. **Restructure:**
   - Thêm markdown headers cho legal structures
   - Standardize formatting
   - Convert markdown → plain text

**Các hàm chính:**
- `clean_markdown()`: Làm sạch markdown
- `parse_markdown_structure()`: Parse cấu trúc
- `restructure_markdown()`: Restructure với legal headers
- `preprocess_file()`: Xử lý một file
- `preprocess_all_files()`: Xử lý tất cả files

**Input/Output:**
- Input: `documents/markdown/*.md`
- Output: `data/text/*.txt`

---

### 4. `read_pdf.py` - PDF OCR Processing

**Công dụng:**
- Đọc PDF files (scanned documents)
- OCR với pytesseract (tối ưu cho tiếng Việt)
- Chuyển đổi PDF → Markdown

**Công nghệ sử dụng:**
- **pytesseract**: OCR engine (Tesseract)
- **pdf2image**: PDF → Image conversion
- **PIL (Pillow)**: Image processing
- **NumPy**: Image array operations
- **scipy**: Advanced image preprocessing (optional)

**Phương thức hoạt động:**

1. **PDF to Images:**
   - Convert PDF pages thành images
   - DPI: 300 (high quality cho OCR)

2. **Image Preprocessing:**
   - Deskew (chỉnh độ nghiêng)
   - Enhance contrast và brightness
   - Noise reduction
   - Adaptive thresholding

3. **OCR Processing:**
   - Tesseract với Vietnamese language pack
   - Custom config cho tiếng Việt
   - Parallel processing (ThreadPoolExecutor)

4. **Post-processing:**
   - Optimize markdown output
   - Loại bỏ số trang, header/footer
   - Preserve legal structure

**Các hàm chính:**
- `read_pdf_to_markdown()`: Main function đọc PDF
- `_deskew_image()`: Chỉnh độ nghiêng ảnh
- `optimize_markdown_output()`: Tối ưu markdown output
- `process_legal_documents()`: Xử lý batch PDFs

**Input/Output:**
- Input: `documents/*.pdf`
- Output: `documents/markdown/*.md`

**Lưu ý:**
- Cần cài Tesseract OCR với Vietnamese language pack
- Cần Poppler cho pdf2image
- Chậm hơn read_word.py (vì OCR)

---

### 5. `read_word.py` - Word Document Processing

**Công dụng:**
- Đọc Word documents (.docx, .doc)
- Chuyển đổi Word → Markdown
- Preserve structure và formatting

**Công nghệ sử dụng:**
- **python-docx**: Đọc .docx files (chính xác, không OCR)
- **docx2txt**: Đọc .doc files (fallback)
- **textract**: Alternative cho .doc (optional)

**Phương thức hoạt động:**

1. **Structure Detection:**
   - Detect headings (heading 1-6, title, subtitle)
   - Detect lists (bullet, numbered)
   - Detect tables với merged cells
   - Preserve document order

2. **Text Extraction:**
   - Extract text với formatting (bold, italic)
   - Extract từ paragraphs, tables, lists
   - Handle merged cells trong tables

3. **Markdown Conversion:**
   - Convert headings → markdown headers
   - Convert lists → markdown lists
   - Convert tables → markdown tables
   - Preserve legal structure (Chương, Điều, Khoản, Điểm)

4. **Post-processing:**
   - Optimize markdown output
   - Loại bỏ số trang, header/footer
   - Standardize whitespace

**Các hàm chính:**
- `read_docx_to_markdown()`: Đọc .docx files
- `read_doc_to_markdown()`: Đọc .doc files
- `iter_block_items()`: Iterate blocks theo thứ tự
- `is_list_item()`: Detect list items
- `optimize_markdown_output()`: Tối ưu output

**Input/Output:**
- Input: `documents/*.docx`, `documents/*.doc`
- Output: `documents/markdown/*.md`

**Ưu điểm:**
- Không cần OCR (giữ nguyên chính tả)
- Nhanh hơn PDF OCR
- Chính xác hơn (không có lỗi OCR)

---

### 6. `correction.py` - OCR Error Correction

**Công dụng:**
- Sửa lỗi OCR từ PDF processing
- Pattern matching và correction
- Tối ưu cho tiếng Việt

**Công nghệ sử dụng:**
- **Regex**: Pattern matching
- **Dictionary lookup**: Common OCR errors
- **Context-aware correction**: Sửa dựa trên context

**Phương thức hoạt động:**

1. **Error Detection:**
   - Detect common OCR errors (ví dụ: "0" → "o", "1" → "l")
   - Pattern matching cho tiếng Việt
   - Context analysis

2. **Correction:**
   - Dictionary-based correction
   - Rule-based correction
   - Context-aware correction

**Các class chính:**
- `AdvancedOCRCorrector`: Main correction class

**Lưu ý:**
- Optional module (chỉ cần nếu dùng PDF OCR)
- Không cần thiết nếu chỉ dùng Word documents

---

### 7. `deploy.py` - Deployment Utilities

**Công dụng:**
- Utilities cho deployment
- Helper functions cho production

**Công nghệ sử dụng:**
- Standard Python libraries

**Lưu ý:**
- Module đơn giản, ít được sử dụng
- Có thể mở rộng trong tương lai

---

## 🔄 Data Pipeline Flow

```
1. Documents (Word/PDF)
   ↓
2. read_word.py / read_pdf.py
   ↓
   documents/markdown/*.md
   ↓
3. preprocess.py
   ↓
   data/text/*.txt
   ↓
4. embedding.py
   ↓
   data/data_for_rag.index
   data/data_for_rag_meta.pkl
   ↓
5. query.py (RAG System)
   ↓
   User Query → Retrieved Context → Gemini API → Response
```

---

## 🛠️ Dependencies

### Core Dependencies
- `sentence-transformers>=2.2.0`: Embedding models
- `faiss-cpu>=1.7.4`: Vector search
- `google-generativeai>=0.3.0`: Gemini API
- `flask>=3.0.0`: Web framework
- `rank-bm25>=0.2.2`: Fast BM25 search

### Document Processing
- `python-docx>=1.1.0`: Word document reading
- `docx2txt>=0.8`: .doc file support
- `pytesseract>=0.3.10`: PDF OCR
- `pdf2image>=1.16.3`: PDF to image

### Optional
- `pyvi>=0.1.1`: Vietnamese tokenizer
- `scipy>=1.10.0`: Advanced image processing

---

## 📊 Performance Characteristics

| Module | Time Complexity | Space Complexity | Bottleneck |
|--------|----------------|------------------|------------|
| `query.py` | O(K log K) | O(K) | Cross-encoder (GPU) |
| `embedding.py` | O(N × D) | O(N × D) | Embedding model |
| `read_pdf.py` | O(P × I) | O(I) | OCR processing |
| `read_word.py` | O(D) | O(D) | File I/O |
| `preprocess.py` | O(N) | O(N) | Regex operations |

**Ký hiệu:**
- N: Số chunks
- K: Số candidates
- D: Document size
- P: Số pages
- I: Image size

---

## 🔍 Key Design Decisions

1. **RRF thay vì Weighted Sum**: Stable ranking, không bias theo query
2. **Inner Product metric**: Chính xác với normalized embeddings
3. **Batch processing**: Tránh GPU memory spike
4. **Hard/Soft constraints**: Diversity filtering linh hoạt
5. **3-layer prompts**: Token optimization
6. **Chunking tối ưu**: Tôn trọng legal structure

---

## 📝 Notes

- Tất cả modules đều có error handling và logging
- Path handling tương thích Windows/Linux/Mac
- Unicode support cho tiếng Việt
- Memory-efficient với large datasets

