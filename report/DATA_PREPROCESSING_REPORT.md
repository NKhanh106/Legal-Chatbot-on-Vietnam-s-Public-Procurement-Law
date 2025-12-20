# BÁO CÁO QUY TRÌNH ĐỌC VÀ PREPROCESS DỮ LIỆU
## Legal Chatbot - Luật Đấu thầu Việt Nam

---

## 📋 MỤC LỤC

1. [Tổng Quan](#1-tổng-quan)
2. [Quy Trình Tổng Thể](#2-quy-trình-tổng-thể)
3. [Giai Đoạn 1: Đọc Tài Liệu](#3-giai-đoạn-1-đọc-tài-liệu)
4. [Giai Đoạn 2: Preprocessing](#4-giai-đoạn-2-preprocessing)
5. [Giai Đoạn 3: OCR Correction (Optional)](#5-giai-đoạn-3-ocr-correction-optional)
6. [Tối Ưu Hóa](#6-tối-ưu-hóa)
7. [Kết Luận](#7-kết-luận)

---

## 1. TỔNG QUAN

### 1.1. Mục Đích

Quy trình đọc và preprocess dữ liệu được thiết kế để:
- Chuyển đổi tài liệu Word/PDF sang định dạng text/markdown chuẩn
- Làm sạch và chuẩn hóa nội dung
- Tối ưu cho việc chunking và embedding
- Giữ nguyên cấu trúc và metadata của văn bản pháp luật

### 1.2. Pipeline Tổng Thể

```
Word/PDF Documents
    ↓
[read_word.py / read_pdf.py]
    ├─ Convert to Markdown
    ├─ Preserve structure
    └─ Extract tables
    ↓
[preprocess.py]
    ├─ Clean markdown
    ├─ Restructure
    └─ Convert to .txt
    ↓
[correction.py] (Optional)
    ├─ OCR correction (nếu cần)
    └─ Validation
    ↓
[embedding.py]
    ├─ Chunking
    └─ Embedding
```

### 1.3. Đặc Điểm Nổi Bật

- ✅ **Không dùng OCR** cho Word files (giữ nguyên chính tả)
- ✅ **Table-aware**: Phát hiện và preserve bảng biểu
- ✅ **Structure-aware**: Giữ nguyên cấu trúc pháp luật
- ✅ **Metadata extraction**: Tự động extract từ filename
- ✅ **OCR correction**: Sử dụng AI để sửa lỗi OCR (cho PDF scan)

---

## 2. QUY TRÌNH TỔNG THỂ

### 2.1. Workflow Diagram

```
┌─────────────────────┐
│ Word/PDF Documents  │
│ (documents/)         │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ read_word.py        │
│ read_pdf.py         │
│ - Convert to MD      │
│ - Preserve structure│
│ - Extract tables    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ documents/markdown/  │
│ (*.md files)        │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ preprocess.py        │
│ - Clean markdown     │
│ - Restructure        │
│ - Parse structure   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ data/text/          │
│ (*.txt files)       │
└──────────┬──────────┘
           │
           ▼ (Optional)
┌─────────────────────┐
│ correction.py       │
│ - OCR correction    │
│ - Validation        │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ embedding.py        │
│ - Chunking           │
│ - Embedding          │
└─────────────────────┘
```

### 2.2. Các Module Chính

1. **`read_word.py`**: Đọc Word documents (.doc, .docx)
2. **`read_pdf.py`**: Đọc PDF với OCR (cho PDF scan)
3. **`preprocess.py`**: Preprocessing markdown và text
4. **`correction.py`**: OCR correction với AI (optional)

---

## 3. GIAI ĐOẠN 1: ĐỌC TÀI LIỆU

### 3.1. Word Documents (read_word.py)

#### 3.1.1. Công Nghệ

- **Library**: `python-docx` (cho .docx)
- **Library**: `python-docx2txt` (cho .doc)
- **Không dùng OCR**: Vì file Word gốc, giữ nguyên chính tả

#### 3.1.2. Quy Trình

**Bước 1: Đọc Document**
```python
doc = Document(str(docx_path))
```

**Bước 2: Xử Lý Blocks Theo Thứ Tự**
- Sử dụng `iter_block_items()` để giữ nguyên thứ tự
- Xử lý xen kẽ: Paragraphs và Tables

**Bước 3: Detect và Convert**
- **Headings**: Detect từ style name (Heading 1-6)
- **Lists**: Detect từ paragraph format và style
- **Tables**: Convert sang markdown table format
- **Paragraphs**: Giữ nguyên với formatting (bold, italic)

**Bước 4: Post-Processing**
- Loại bỏ số trang và header/footer
- Chuẩn hóa markdown headers
- Thêm phân cấp cho cấu trúc pháp luật:
  - Chương, Phần, Mục → `##`
  - Điều → `###`
  - Khoản → `####`
  - Điểm → `#####`

#### 3.1.3. Table Handling

**Detection**:
- Detect markdown table pattern: `| ... |`
- Detect separator: `| --- |`

**Conversion**:
- Header row: `| Col1 | Col2 | Col3 |`
- Separator: `| --- | --- | --- |`
- Data rows: `| Data1 | Data2 | Data3 |`
- Escape pipe trong cell: `\|`

**Preservation**:
- Tables được giữ nguyên như một block
- Không bị cắt khi chunking

#### 3.1.4. Metadata Extraction

Từ filename (ví dụ: `05_2024_TT-BKHDT_607705.docx`):
- **Year**: `2024` (pattern: `_YYYY_`)
- **Document Type**: `Thông tư` (pattern: `TT-`, `ND-`, etc.)
- **Document Number**: `607705` (pattern: `_NNNNNN`)

### 3.2. PDF Documents (read_pdf.py)

#### 3.2.1. Công Nghệ

- **OCR Engine**: Tesseract OCR với Vietnamese language pack
- **Image Processing**: PIL, numpy, scipy (optional)
- **PDF to Image**: `pdf2image` (poppler)

#### 3.2.2. Quy Trình

**Bước 1: Convert PDF to Images**
```python
images = convert_from_path(pdf_path, dpi=400)
```

**Bước 2: Image Preprocessing** (cho ảnh chụp/scan)
- Grayscale conversion
- Noise reduction (median filter)
- Deskewing (chỉnh góc nghiêng)
- Contrast enhancement
- Brightness adjustment
- Binarization (Otsu threshold)

**Bước 3: OCR với Tesseract**
- Language: `vie_best` > `vie` > `vi`
- Config: `--oem 1 --psm 6 --dpi 400`
- Parallel processing: ThreadPoolExecutor (4 workers)

**Bước 4: Post-Processing**
- Tương tự như Word documents
- Loại bỏ số trang và artifacts
- Chuẩn hóa markdown

#### 3.2.3. Tối Ưu

- **DPI**: 400 (cân bằng tốc độ/chất lượng)
- **Parallel OCR**: 4 workers để tăng tốc
- **Language Detection**: Tự động chọn Vietnamese language pack tốt nhất

---

## 4. GIAI ĐOẠN 2: PREPROCESSING

### 4.1. Module: preprocess.py

#### 4.1.1. Chức Năng Chính

1. **`clean_markdown()`**: Làm sạch markdown
2. **`parse_markdown_structure()`**: Parse cấu trúc markdown
3. **`restructure_markdown()`**: Tái cấu trúc markdown
4. **`preprocess_file()`**: Xử lý một file
5. **`preprocess_all_files()`**: Xử lý tất cả files

#### 4.1.2. Clean Markdown

**Loại bỏ**:
- Số trang: `=+ Trang X =+`, `-X-`
- Code blocks: ` ```...``` `
- Images: `![...](...)`
- Khoảng trắng thừa: `\n{3,}` → `\n\n`

**Chuẩn hóa**:
- Spaces: `[ \t]+` → ` `
- Line endings: `\r\n` → `\n`

#### 4.1.3. Parse Markdown Structure

**Phát hiện**:
- **Headers**: `#`, `##`, `###`, etc.
- **Lists**: `-`, `*`, `+`, `1.`, `2.`, etc.
- **Legal Structure**: `Điều X`, `Khoản Y`, `Điểm Z`

**Output**:
```python
{
    "level": 2,
    "type": "legal_structure",
    "content": [...],
    "header": "Điều 23",
    "legal_type": "Điều",
    "legal_number": "23",
    "legal_title": "..."
}
```

#### 4.1.4. Restructure Markdown

**Logic**:
- Giữ nguyên list structure
- Tách legal structure thành paragraphs hợp lý
- Tách paragraph dài (>500 chars) thành các câu
- Đảm bảo paragraphs không quá dài

**Output**: Markdown đã được tái cấu trúc, phù hợp cho chunking

#### 4.1.5. Convert to Text

- Input: `.md` files từ `documents/markdown/`
- Output: `.txt` files trong `data/text/`
- Mục đích: Chuẩn hóa format cho embedding system

---

## 5. GIAI ĐOẠN 3: OCR CORRECTION (OPTIONAL)

### 5.1. Module: correction.py

#### 5.1.1. Mục Đích

Sửa lỗi OCR trong markdown files (chủ yếu cho PDF scan):
- Khôi phục dấu tiếng Việt
- Sửa lỗi chính tả
- Sửa lỗi ký tự bị miss

#### 5.1.2. Công Nghệ

- **AI Model**: Gemini (`gemini-2.5-flash-lite`)
- **Validation**: Kiểm tra tỷ lệ dấu tiếng Việt
- **Adaptive Prompting**: 2 vòng với prompts khác nhau

#### 5.1.3. Quy Trình

**Bước 1: Chunking**
- Gom nhiều đoạn thành chunks lớn (3000-8000 ký tự)
- Mục tiêu: 35 chunks/file để giảm số requests

**Bước 2: Correction với Validation Loop**
- **Vòng 1**: Prompt nhẹ nhàng (sửa lỗi OCR)
- **Validation**: Kiểm tra tỷ lệ dấu (threshold: 15%)
- **Vòng 2**: Prompt cực gắt (khôi phục dấu) nếu vòng 1 thất bại
- **Retry**: Tối đa 5 lần với adaptive threshold

**Bước 3: Post-Processing**
- Tối ưu format markdown
- Chuẩn hóa headers và spacing

#### 5.1.4. Rate Limiting

- **Delay**: 8 giây giữa các requests
- **Exponential Backoff**: 15s, 30s, 60s, 120s, 180s nếu rate limit
- **Chunk Size**: Lớn để giảm số requests (phù hợp với 1M TPM)

---

## 6. TỐI ƯU HÓA

### 6.1. Performance Optimizations

#### 6.1.1. Word Reading
- **Block Iteration**: Giữ nguyên thứ tự (paragraphs và tables xen kẽ)
- **Formatting Preservation**: Giữ nguyên bold, italic
- **Table Detection**: Phát hiện và convert chính xác

#### 6.1.2. PDF OCR
- **Parallel Processing**: 4 workers cho OCR
- **Image Optimization**: Preprocessing để tăng độ chính xác OCR
- **Language Detection**: Cache language để tránh detect lại

#### 6.1.3. Preprocessing
- **Structure Parsing**: Parse một lần, reuse nhiều lần
- **Regex Optimization**: Compile patterns một lần

#### 6.1.4. OCR Correction
- **Chunking**: Gom nhiều đoạn để giảm số requests
- **Context Passing**: Truyền context từ chunk trước để AI hiểu ngữ cảnh
- **Validation**: Chỉ retry khi cần thiết

### 6.2. Quality Optimizations

#### 6.2.1. Structure Preservation
- **Legal Hierarchy**: Giữ nguyên Chương → Điều → Khoản → Điểm
- **Markdown Headers**: Thêm headers phù hợp cho cấu trúc pháp luật
- **Table Preservation**: Không cắt bảng khi chunking

#### 6.2.2. Metadata Enrichment
- **Filename Parsing**: Tự động extract year, document_type, document_number
- **Structure Extraction**: Extract article, clause, point numbers
- **Table Metadata**: `has_table`, `table_data`

#### 6.2.3. Error Handling
- **Fallback Mechanisms**: Fallback nếu không có dependencies
- **Validation**: Validate text sau khi đọc
- **Error Messages**: Rõ ràng, dễ debug

---

## 7. KẾT LUẬN

### 7.1. Tổng Kết

Quy trình đọc và preprocess dữ liệu đã được thiết kế tốt:

- ✅ **Đầy đủ**: Hỗ trợ cả Word và PDF
- ✅ **Chất lượng**: Giữ nguyên cấu trúc và chính tả
- ✅ **Tối ưu**: Performance và quality optimizations
- ✅ **Robust**: Error handling và fallback mechanisms

### 7.2. Điểm Mạnh

1. **Không dùng OCR cho Word**: Giữ nguyên chính tả chính xác
2. **Table-aware**: Phát hiện và preserve bảng biểu
3. **Structure-aware**: Giữ nguyên cấu trúc pháp luật
4. **Metadata-rich**: Tự động extract metadata từ filename và content
5. **OCR Correction**: Sử dụng AI để sửa lỗi OCR (cho PDF scan)

### 7.3. Hướng Phát Triển

1. **Advanced Table Parsing**: Sử dụng layout-aware parsers (LayoutPDFReader, Unstructured)
2. **Incremental Updates**: Hỗ trợ update từng file mà không cần rebuild toàn bộ
3. **Validation**: Thêm validation về hiệu lực và phạm vi áp dụng
4. **Monitoring**: Metrics và logging chi tiết hơn

### 7.4. Khuyến Nghị Sử Dụng

- **Cho Word Documents**: 
  - Sử dụng `read_word.py` (không cần OCR correction)
  - Chất lượng tốt nhất, nhanh nhất
  
- **Cho PDF Documents**:
  - Sử dụng `read_pdf.py` (cần OCR)
  - Có thể cần `correction.py` để sửa lỗi OCR
  
- **Cho Production**:
  - Validate files trước khi process
  - Monitor chất lượng output
  - Cập nhật thường xuyên

---

**Báo cáo được tạo bởi**: AI Assistant  
**Ngày**: 2025  
**Phiên bản**: 1.0

