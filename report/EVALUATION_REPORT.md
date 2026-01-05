# BÁO CÁO ĐÁNH GIÁ HỆ THỐNG RAG (EVALUATION REPORT)

## 1. Tổng Quan về Hệ Thống Đánh Giá

Hệ thống đánh giá (`evaluate.py`) được thiết kế để đo lường hiệu năng tìm kiếm và sinh câu trả lời của mô hình RAG một cách khách quan và công bằng.

### 1.1. Bộ Dữ Liệu Kiểm Thử (Dataset)
- **Nguồn dữ liệu**: Các cặp câu hỏi và câu trả lời (Q&A) được sinh tự động bởi AI (LLM) dựa trên kho văn bản pháp luật gốc.
- **Đặc điểm**:
    - Câu hỏi đa dạng: Định nghĩa, quy trình, so sánh, yêu cầu, tình huống.
    - Ground Truth (Chân lý): File nguồn chính xác chứa thông tin trả lời.
- **Lưu ý**: Do dữ liệu được sinh bởi AI, độ chính xác của Ground Truth phụ thuộc vào khả năng của mô hình sinh. Tuy nhiên, đây là cách hiệu quả để tạo ra tập dữ liệu lớn phục vụ kiểm thử tự động.

## 2. Các Metrics Đánh Giá

Chúng tôi sử dụng các chỉ số tiêu chuẩn trong Information Retrieval để đánh giá:

### 2.1. Recall@K (R@K)
- **Định nghĩa**: Tỷ lệ số câu hỏi mà tài liệu đúng (Ground Truth) xuất hiện trong top K kết quả trả về.
- **Ý nghĩa**: Đo lường khả năng "bao phủ" của hệ thống. R@20 cao nghĩa là hệ thống hiếm khi bỏ sót tài liệu quan trọng.
- **Các mức K**: 1, 3, 5, 10, 20.

### 2.2. MAP@K (Mean Average Precision)
- **Định nghĩa**: Trung bình độ chính xác (Precision) tại các điểm cutoff, có tính đến thứ hạng của kết quả đúng.
- **Ý nghĩa**: Phản ánh chất lượng xếp hạng. Nếu tài liệu đúng nằm ở vị trí #1, điểm sẽ cao hơn nhiều so với vị trí #20.

### 2.3. MRR (Mean Reciprocal Rank)
- **Định nghĩa**: Trung bình nghịch đảo của thứ hạng tìm thấy đầu tiên (1/rank).
- **Ý nghĩa**: Đo lường khả năng đưa kết quả đúng lên vị trí đầu tiên (Top 1).

## 3. Quy Trình Đánh Giá (Evaluation Pipeline)

Quy trình đánh giá được thực hiện theo nguyên tắc "Fair Test" (Công bằng):

1.  **Chế độ Đánh giá (Evaluation Mode)**:
    - **Tắt Diversity Filter**: Để đo lường khả năng tìm kiếm thô (Raw Retrieval Power).
    - **Tắt Deduplication**: Tránh loại bỏ các bản sao hợp lệ có thể là Ground Truth.
    - **Return Metadata**: Buộc hệ thống trả về đầy đủ thông tin nguồn để so khớp.

2.  **Các Phương pháp So sánh (A/B Testing)**:

    | Phương pháp | Mô tả | Mục đích |
    | :--- | :--- | :--- |
    | **FAISS Only** | Chỉ sử dụng tìm kiếm Semantic Vector | Baseline để so sánh |
    | **Hybrid w/o Cross-Encoder** | FAISS + BM25 + RRF | Đánh giá hiệu quả của Cross-Encoder |
    | **Hybrid w/ Cross-Encoder** | Full Pipeline (FAISS + BM25 + RRF + Re-rank) | Đánh giá hiệu năng thực tế (Production) |

## 4. Kết Quả và Tối Ưu Hóa (Cập nhật gần nhất)

Sau các lần tinh chỉnh, hệ thống đã đạt được các cải tiến đáng kể:

### 4.1. Các Thay Đổi Cốt Lõi
- **RRF Score Boosting**: Tăng trọng số điểm RRF lên 25 lần để cân bằng với Keyword Score (tránh bị Keyword nhiễu lấn át).
- **Dynamic Limits**: Nới rộng giới hạn ứng viên (Stage 3) từ 30 lên 60 chunks để tránh cắt bỏ sớm các tài liệu tiềm năng.
- **Adaptive Cross-Encoder Weights**: Tự động điều chỉnh trọng số dựa trên độ tin cậy của Cross-Encoder và loại câu hỏi.

### 4.2. Ý Nghĩa Kết Quả
- **Recall Cải Thiện**: Việc tăng giới hạn Candidate giúp Recall@20 tăng lên, giảm thiểu trường hợp "không tìm thấy thông tin".
- **Ranking Chính Xác Hơn**: Boost RRF giúp các tài liệu có ý nghĩa ngữ nghĩa (Semantic) tốt được xếp hạng cao hơn, ngay cả khi ít từ khóa khớp chính xác.
- **Latency**: Việc tách rời đánh giá (có và không có Cross-Encoder) giúp xác định rõ chi phí thời gian (Latency) đổi lấy độ chính xác.

## 5. Hướng Dẫn Chạy Đánh Giá

Để chạy quá trình đánh giá:

```bash
python backend/src/evaluate.py
```

Kết quả sẽ được tự động lưu vào thư mục `evaluate/` bao gồm:
- `evaluation_results.csv`: Bảng số liệu chi tiết.
- `evaluation_chart.png`: Biểu đồ trực quan hóa so sánh Recall@20.
