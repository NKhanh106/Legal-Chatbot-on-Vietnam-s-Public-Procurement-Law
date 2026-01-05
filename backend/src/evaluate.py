"""
Module đánh giá hiệu năng hệ thống RAG (Evaluation Script) - REAL MODE.
Kết nối trực tiếp với backend/src/query.py để chạy đánh giá trên dữ liệu thực.

Usage:
    python backend/src/evaluate.py
"""

import os
import sys
import time
import re
import unicodedata
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Union

# Import Groq client để dùng LLM-as-a-Judge
from groq import Groq
from dotenv import load_dotenv

# --- CẤU HÌNH PATH & IMPORT ---
# 1. Xác định Project Root và thêm backend/src vào sys.path
current_file_path = os.path.abspath(__file__)
custom_backend_src = os.path.dirname(current_file_path) # e.g. .../backend/src
project_root = os.path.dirname(os.path.dirname(custom_backend_src)) # e.g. .../Project 3/...

if custom_backend_src not in sys.path:
    sys.path.insert(0, custom_backend_src) # Ưu tiên tìm trong current dir

# Load env
load_dotenv(os.path.join(project_root, ".env"))
load_dotenv(os.path.join(project_root, ".env.local"), override=True)

# 2. Import module query (REAL BACKEND)
try:
    import query
    print(f"✅ Đã import thành công module 'query' từ {query.__file__}")
except ImportError as e:
    print(f"❌ LỖI NGHIÊM TRỌNG: Không thể import module 'query'.")
    print(f"   Chi tiết lỗi: {e}")
    print(f"   Vui lòng kiểm tra lại PYTHONPATH hoặc vị trí file query.py.")
    query = None

# --- CẤU HÌNH DATASET ---
DEFAULT_DATASET_PATHS = [
    os.path.join(project_root, "bo_150_cau_hoi_RAG_Level4.csv"),
    os.path.join(project_root, "evaluate", "bo_150_cau_hoi_RAG_Level4.xlsx"),
    os.path.join(project_root, "bo_150_cau_hoi_RAG_Level4.xlsx"),
]

class RAGEvaluator:
    """
    Class đánh giá hiệu năng của hệ thống RAG (Real-time).
    """

    def __init__(self):
        self.results = []
        self.detailed_results = []
        
        # Khởi tạo Groq client cho LLM-as-a-Judge
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.llm_client = None
        
        if self.groq_api_key:
            try:
                self.llm_client = Groq(api_key=self.groq_api_key)
                print("✅ Đã khởi tạo LLM-as-a-Judge (Groq)")
            except Exception as e:
                print(f"⚠️ Không thể khởi tạo Groq client: {e}")
        else:
            print("⚠️ Không tìm thấy GROQ_API_KEY. Tính năng đánh giá Faithfulness sẽ bị bỏ qua (trả về 0).")

    def _normalize_text(self, text: Any) -> str:
        """Chuẩn hóa text (Unicode NFC, lowercase)."""
        if text is None:
            return ""
        if not isinstance(text, str):
            text = str(text)
        text = text.lower().strip()
        return unicodedata.normalize('NFC', text)

    def _extract_doc_code(self, text: str) -> Optional[str]:
        """Trích xuất mã số văn bản (Regex). Hỗ trợ cả định dạng file name."""
        # Pattern 1: Chuẩn legal (02/2024/TT-BKHĐT)
        pattern1 = r"(\d+/\d+/[a-zđ-]+)"
        match1 = re.search(pattern1, text)
        if match1: return match1.group(1)
        
        # Pattern 2: Filename format (02_2024_TT-BKHDT)
        pattern2 = r"(\d+_\d+_[a-z-]+)"
        match2 = re.search(pattern2, text)
    def _extract_source_from_chunk(self, chunk: Any) -> str:
        """Trích xuất tên nguồn từ kết quả Search."""
        if isinstance(chunk, dict):
            # Check ưu tiên
            if "source_document" in chunk: return str(chunk["source_document"])
            if "source" in chunk: return str(chunk["source"])
            
            # Check citation list (đặc thù của full pipeline format)
            if "citation" in chunk:
                citation = chunk["citation"]
                if isinstance(citation, list):
                    return ", ".join(str(c) for c in citation)
                return str(citation)
            
            # Check metadata lồng
            metadata = chunk.get("metadata", {})
            if isinstance(metadata, dict):
                if "source" in metadata: return str(metadata["source"])
                if "filename" in metadata: return str(metadata["filename"])
                if "source_document" in metadata: return str(metadata["source_document"])
                    
        # Check object
        if hasattr(chunk, "metadata"):
            meta = getattr(chunk, "metadata")
            if isinstance(meta, dict):
                return str(meta.get("source") or meta.get("filename") or meta.get("source_document") or "")
        
        if isinstance(chunk, str): return chunk
        return ""

    def _clean_code(self, text: str) -> str:
        """
        Chuẩn hóa mã văn bản pháp luật để so sánh nhất quán.
        Chuyển đổi các định dạng như "Số: 24/2024/NĐ-CP" hoặc "02_2024_TT-BKHDT_591580" thành định dạng chuẩn "XX/YYYY/CODE".
        """
        # 1. Chuyển chữ thường (đã làm ở _normalize_text nhưng làm lại cho chắc)
        text = text.lower()

        # 2. Xử lý tiếng Việt đặc thù (đ -> d) để khớp với tên file không dấu
        text = text.replace('đ', 'd')

        # 3. Chuẩn hóa dấu phân cách: chuyển _ thành /
        text = text.replace('_', '/')
        
        # 4. Thay thế regex chi tiết để loại bỏ các tiền tố pháp lý thông thường
        text = re.sub(r'^(số|nghị định|thông tư|luật|quyết định)\s*[:\.]?\s*', '', text, flags=re.IGNORECASE)
        
        # 5. Loại bỏ tất cả khoảng trắng
        text = text.replace(' ', '')
        
        # 6. Loại bỏ chuỗi ID số ngẫu nhiên ở cuối (ví dụ: /591580)
        text = re.sub(r'/\d+$', '', text)
        
        return text

    def _check_match(self, ground_truth: str, result_source: str) -> bool:
        """
        Thực hiện so khớp mờ thông minh giữa Ground Truth và Nguồn kết quả.
        """
        if not ground_truth or not result_source:
            return False

        gt_norm = self._normalize_text(ground_truth)
        res_norm = self._normalize_text(result_source)
        
        # 1. Kiểm tra khớp chính xác sau khi chuẩn hóa cơ bản
        if gt_norm == res_norm:
            return True
            
        # 2. So khớp mã văn bản bằng Regex (Ưu tiên cao)
        # Trích xuất và so sánh các mã pháp lý (ví dụ: 24/2024/nđ-cp)
        gt_code = self._extract_doc_code(gt_norm)
        res_code = self._extract_doc_code(res_norm)

        if gt_code and res_code:
            # Làm sạch và chuẩn hóa mã trước khi so sánh
            gt_code_clean = self._clean_code(gt_code)
            res_code_clean = self._clean_code(res_code)
            
            if gt_code_clean == res_code_clean: 
                return True
            if gt_code_clean in res_code_clean or res_code_clean in gt_code_clean: 
                return True
        
        # 3. Fallback chuỗi (Kiểm tra bao hàm tương đối)
        # Kiểm tra xem mã GT có xuất hiện ở bất kỳ đâu trong chuỗi kết quả đã chuẩn hóa không (và ngược lại)
        # Hữu ích cho các trường hợp Kết quả dài dòng như "Điều 1, Nghị định..." nhưng GT ngắn gọn.
        if gt_code:
             if self._clean_code(gt_code) in self._clean_code(res_norm):
                 return True
                 
        # 4. Kiểm tra bao hàm từ khóa (Fallback nếu trích xuất mã thất bại)
        # Nếu GT đủ dài (có thể là tiêu đề đầy đủ), kiểm tra xem nó có bao gồm chuỗi con hay không.
        if len(gt_norm) > 10 and len(res_norm) > 5:
            if res_norm in gt_norm: return True
            if gt_norm in res_norm: return True
            
        return False
        
    def _call_llm_judge_with_retry(self, prompt: str, max_retries=3) -> float:
        """Gọi LLM Judge với Retry."""
        if not self.llm_client:
            return 0.0
            
        for attempt in range(max_retries):
            try:
                chat_completion = self.llm_client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama-3.3-70b-versatile",
                    temperature=0.0,
                    max_tokens=10
                )
                result = chat_completion.choices[0].message.content.strip()
                match = re.search(r'\b(0|1)\b', result)
                if match:
                    return float(match.group(1))
                return 0.0
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"⚠️ LLM Judge Error: {e}")
                time.sleep((attempt + 1) * 2)
        return 0.0

    def _check_citation_accuracy(self, response: str, ground_truth_doc: str) -> bool:
        return self._check_match(ground_truth_doc, response)

    def _evaluate_faithfulness_llm(self, question: str, contexts: List[str], answer: str) -> float:
        if not answer: return 0.0
        context_text = "\n---\n".join(contexts[:3]) if contexts else "NO CONTEXT PROVIDED"
        
        prompt = f"""[System: You are an impartial AI Judge evaluating a RAG system.]
        TASK: Determine if the generated ANSWER relies ENTIRELY on the provided CONTEXTS.
        
        INPUT DATA:
        - Question: "{question}"
        - Contexts: 
        {context_text}
        
        - Generated Answer: 
        {answer}

        SCORING RULES:
        1. **Standard Case**: If the answer is supported by the Contexts -> Return 1.
        2. **Hallucination**: If the answer claims facts NOT present in Contexts -> Return 0.
        3. **Negative Test / Out-of-Domain**: If the Contexts are irrelevant or empty, AND the Answer correctly states "I cannot answer", "Information not found", or similar -> Return 1 (This is FAITHFUL behavior).
        4. **Contradiction**: If the Answer contradicts the Contexts -> Return 0.

        OUTPUT FORMAT: Return ONLY the digit '0' or '1'.
        """
        return self._call_llm_judge_with_retry(prompt)

    # --------------------------------------------------------------------------
    # REAL SEARCH FUNCTIONS (UNLOCKED)
    # --------------------------------------------------------------------------

    def search_faiss_only(self, query_text: str) -> Union[List[Dict], float]:
        """
        [REAL] Phương pháp 1: Chỉ tìm kiếm Vector (FAISS).
        Thực hiện query.search_faiss với quy trình đa tầng bị vô hiệu hóa (chỉ FAISS).
        """
        start_time = time.time()
        try:
            if not query:
                print("❌ Query module chưa load.")
                return [], 0.0
            
            # Truy xuất danh sách các dictionary chứa kết quả tìm kiếm
            # use_multi_stage=False vô hiệu hóa reranking, thực hiện tìm kiếm FAISS tiêu chuẩn
            # Truy xuất 20 kết quả hàng đầu để tính toán các chỉ số mở rộng (R@20, MAP@20)
            results = query.search_faiss(query_text, top_k=20, use_multi_stage=False)
            latency = (time.time() - start_time) * 1000
            return results, latency
        except Exception as e:
            print(f"❌ Error in search_faiss_only: {e}")
            return [], 0.0

    def search_hybrid(self, query_text: str) -> Union[List[Dict], float]:
        """
        [REAL] Phương pháp 2: Tìm kiếm Lai (Vector + BM25).
        Thực hiện query.search_faiss với quy trình đa tầng được kích hoạt.
        """
        start_time = time.time()
        try:
            if not query:
                print("❌ Query module chưa load.")
                return [], 0.0
            
            # Truy xuất kết quả sử dụng toàn bộ quy trình tìm kiếm lai (BM25 + FAISS)
            # use_multi_stage=True kích hoạt logic Tìm kiếm Lai nội bộ trong query.py
            # Truy xuất 20 kết quả hàng đầu để tính toán các chỉ số mở rộng (R@20, MAP@20)
            # TẮT DIVERSITY và DEDUPLICATION để đo lường Recall thuần túy của Reranker Ranking (trong evaluate.py)
            results = query.search_faiss(query_text, top_k=20, use_multi_stage=True, return_metadata=True)
            latency = (time.time() - start_time) * 1000
            return results, latency
        except Exception as e:
            print(f"❌ Error in search_hybrid: {e}")
            return [], 0.0

    def search_hybrid_no_cross(self, query_text: str) -> Union[List[Dict], float]:
        """
        [REAL] Phương pháp 2b: Tìm kiếm Lai (Vector + BM25) nhưng KHÔNG có Cross-Encoder re-ranking.
        Dùng để đánh giá hiệu quả của bước re-ranking.
        """
        start_time = time.time()
        try:
            if not query:
                print("❌ Query module chưa load.")
                return [], 0.0
            
            # enable_cross_encoder=False
            results = query.search_faiss(query_text, top_k=20, use_multi_stage=True, return_metadata=True, enable_cross_encoder=False)
            latency = (time.time() - start_time) * 1000
            return results, latency
        except Exception as e:
            print(f"❌ Error in search_hybrid_no_cross: {e}")
            return [], 0.0

    def search_full_pipeline_generated(self, query_text: str) -> Dict:
        """
        [REAL] Phương pháp 3: Full Pipeline (Tìm kiếm -> Xếp hạng lại -> Sinh câu trả lời).
        Sử dụng query.ask_sth.
        """
        start_time = time.time()
        try:
            if not query:
                print("❌ Query module chưa load.")
                return {'chunks': [], 'answer': "Error: Query module not loaded", 'latency': 0}

            # Gọi hàm thật
            result = query.ask_sth(query_text, return_metadata=True)
            
            # Xử lý kết quả trả về từ API
            chunks_raw = []
            answer_text = ""
            
            if isinstance(result, dict):
                # Format mới của ask_sth: chunks nằm trong key 'sources'
                chunks_raw = result.get('sources', [])
                if not chunks_raw:
                    # Fallback cũ
                    chunks_raw = result.get('sources_raw', [])
                
                answer_text = result.get('answer', "")
            else:
                answer_text = str(result)
            
            latency = (time.time() - start_time) * 1000
            return {
                'chunks': chunks_raw,
                'answer': answer_text,
                'latency': latency
            }
        except Exception as e:
            print(f"❌ Error in search_full_pipeline_generated: {e}")
            return {'chunks': [], 'answer': f"System Error: {e}", 'latency': 0}

    # --------------------------------------------------------------------------
    # EVALUATION LOGIC
    # --------------------------------------------------------------------------

    def evaluate_dataset(self, df: pd.DataFrame):
        if df.empty:
            print("❌ Dataset rỗng!")
            return

        methods = [
            ("FAISS Only (Retrieval)", self.search_faiss_only, False),
            ("Hybrid w/o Cross-Encoder", self.search_hybrid_no_cross, False),
            ("Hybrid w/ Cross-Encoder", self.search_hybrid, False)
            #("Full RAG (End-to-End)", self.search_full_pipeline_generated, True)
        ]

        self.detailed_results = []
        self.results = []
        
        os.makedirs("evaluate", exist_ok=True)
        checkpoint_file = os.path.join("evaluate", "evaluation_checkpoint.csv")
        
        print(f"🚀 Bắt đầu đánh giá trên {len(df)} câu hỏi (REAL MODE)...")

        for method_name, method_func, has_gen in methods:
            print(f"\n🧪 Đang chạy phương pháp: {method_name}")
            
            method_metrics = {
                "recall_1": 0, "recall_3": 0, "recall_5": 0, "recall_10": 0, "recall_20": 0,
                "mrr": 0, "map_20_list": [], "coverage_count": 0,
                "faithfulness": 0, "citation_acc": 0,
                "total_latency": 0, "count_gen": 0
            }

            temp_details = []

            for idx, row in tqdm(df.iterrows(), total=len(df), desc=method_name):
                q_id = row.get("ID", idx)
                question = str(row["Câu hỏi"])
                ground_truth = str(row["Nguồn tài liệu"]) 
                
                chunks = []
                answer = ""
                latency = 0
                
                # Gọi hàm search
                output = method_func(question)
                
                # Sleep nhỏ thôi vì chạy local không lo rate limit
                time.sleep(0.1)
                
                # Parse output
                if isinstance(output, tuple):
                    chunks, latency = output
                elif isinstance(output, dict):
                    chunks = output.get('chunks', [])
                    latency = output.get('latency', 0)
                    answer = output.get('answer', "")
                elif isinstance(output, list):
                     chunks = output
                     latency = 0

                method_metrics["total_latency"] += latency
                
                # --- Metrics Calc ---
                found_rank = -1
                top_1_result = "N/A"
                
                normalized_chunks = []
                for chunk in chunks:
                    source_name = self._extract_source_from_chunk(chunk)
                    text_content = ""
                    if isinstance(chunk, dict):
                        text_content = chunk.get("text", "") or chunk.get("content", "")
                    normalized_chunks.append({"source": source_name, "text": text_content})
                
                for rank, chunk_item in enumerate(normalized_chunks[:20], 1):
                    doc_source = chunk_item["source"]
                    if rank == 1: top_1_result = doc_source
                    
                    if self._check_match(ground_truth, doc_source):
                        found_rank = rank
                        break 
                
                # --- DEBUG LOGGING ---
                if found_rank == -1 and idx < 10:
                    print(f"\n[DEBUG Mismatch #{idx}]")
                    print(f"  Question: {question}")
                    print(f"  Ground Truth: '{ground_truth}'")
                    print(f"  Top 1 Retrieved: '{top_1_result}'")
                    print(f"  Top 5 Sources: {[c['source'] for c in normalized_chunks[:5]]}")
                    # In ra extracted code để kiểm tra regex
                    print(f"  GT Code: {self._extract_doc_code(self._normalize_text(ground_truth))}")
                    print(f"  Top 1 Code: {self._extract_doc_code(self._normalize_text(top_1_result))}")
                
                # AP Calculation for MAP@20 (Assuming 1 Ground Truth)
                ap_score = 0.0
                if found_rank != -1 and found_rank <= 20:
                    method_metrics["mrr"] += 1.0 / found_rank
                    if found_rank <= 1: method_metrics["recall_1"] += 1
                    if found_rank <= 3: method_metrics["recall_3"] += 1
                    if found_rank <= 5: method_metrics["recall_5"] += 1
                    if found_rank <= 10: method_metrics["recall_10"] += 1
                    if found_rank <= 20:
                        method_metrics["recall_20"] += 1
                        method_metrics["coverage_count"] += 1 # Coverage hit if found in top 20
                        # MAP@20 Calculation (Single Ground Truth)
                        # AP = Precision at rank k * rel(k)
                        # Với 1 GT tại vị trí rank: Precision = 1/rank
                        # AP = (1/rank * 1) / 1 = 1/rank
                        ap_score = 1.0 / found_rank
                
                method_metrics["map_20_list"].append(ap_score)
                
                faith_score = "N/A"
                citation_score = "N/A"
                
                if has_gen and answer:
                    method_metrics["count_gen"] += 1
                    
                    is_cited = self._check_citation_accuracy(answer, ground_truth)
                    if is_cited:
                        method_metrics["citation_acc"] += 1
                        citation_score = 1
                    else:
                        citation_score = 0
                        
                    context_texts = [c["text"] for c in normalized_chunks[:3]]
                    faith = self._evaluate_faithfulness_llm(question, context_texts, answer)
                    method_metrics["faithfulness"] += faith
                    faith_score = faith

                record = {
                    "Method": method_name,
                    "ID": q_id,
                    "Question": question,
                    "Ground_Truth": ground_truth,
                    "Top1_Result": top_1_result,
                    "Correct_Retrieval": found_rank != -1,
                    "Faithfulness": faith_score,
                    "Citation_Acc": citation_score,
                    "Latency_ms": round(latency, 2)
                }
                self.detailed_results.append(record)
                temp_details.append(record)
                
                if (idx + 1) % 5 == 0:
                     df_temp = pd.DataFrame(self.detailed_results)
                     df_temp.to_csv(checkpoint_file, index=False, encoding="utf-8-sig")

            n = len(df)
            n_gen = method_metrics["count_gen"] if method_metrics["count_gen"] > 0 else 1
            
            summary = {
                "Method": method_name,
                "R@1": method_metrics["recall_1"] / n,
                "R@3": method_metrics["recall_3"] / n,
                "R@5": method_metrics["recall_5"] / n,
                "R@10": method_metrics["recall_10"] / n,
                "R@20": method_metrics["recall_20"] / n,
                "MAP@20": np.mean(method_metrics["map_20_list"]) if method_metrics["map_20_list"] else 0.0,
                "MRR": method_metrics["mrr"] / n,
                "Coverage": method_metrics["coverage_count"] / n,
                "Avg_Latency_ms": method_metrics["total_latency"] / n,
                "Faithfulness": method_metrics["faithfulness"] / n_gen if has_gen else "N/A",
                "Citation_Accuracy": method_metrics["citation_acc"] / n_gen if has_gen else "N/A"
            }
            self.results.append(summary)

    def print_summary(self):
        if not self.results: return
        os.makedirs("evaluate", exist_ok=True)
        output_file = os.path.join("evaluate", "evaluate.txt")
        result_df = pd.DataFrame(self.results)
        display_df = result_df.copy()
        
        def fmt_pct(x): return f"{x:.2%}" if isinstance(x, (int, float)) else str(x)
        def fmt_float(x): return f"{x:.4f}" if isinstance(x, (int, float)) else str(x)

        for col in ["R@1", "R@3", "R@5", "R@10", "R@20", "Coverage", "Faithfulness", "Citation_Accuracy"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(fmt_pct)
        if "MRR" in display_df.columns: display_df["MRR"] = display_df["MRR"].apply(fmt_float)
        if "MAP@20" in display_df.columns: display_df["MAP@20"] = display_df["MAP@20"].apply(fmt_float)
        if "Avg_Latency_ms" in display_df.columns: 
            display_df["Avg_Latency_ms"] = display_df["Avg_Latency_ms"].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else str(x))
        
        output_str = "\n" + "="*60 + "\n"
        output_str += "📊 BẢNG TỔNG HỢP KẾT QUẢ ĐÁNH GIÁ (DATATHẬT)\n"
        output_str += "="*60 + "\n"
        output_str += display_df.to_string(index=False)
        output_str += "\n" + "="*60 + "\n"
        
        print(output_str)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(output_str)
            
    def visualize_results(self):
        if not self.results: return
        df = pd.DataFrame(self.results)
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")
        ax = sns.barplot(x="Method", y="R@20", data=df, palette="viridis")
        for container in ax.containers: ax.bar_label(container, fmt='%.2f', padding=3)
        plt.title("So sánh Recall@20 (REAL DATA)", fontsize=15)
        plt.ylabel("R@20 Score", fontsize=12)
        plt.xlabel("Phương pháp", fontsize=12)
        plt.ylim(0, 1.1) 
        output_img = os.path.join("evaluate", "evaluation_chart.png")
        plt.savefig(output_img)
        print(f"📈 Chart saved: {output_img}")

    def save_details(self):
        if not self.detailed_results: return
        os.makedirs("evaluate", exist_ok=True)
        output_file = os.path.join("evaluate", "evaluate.csv")
        df = pd.DataFrame(self.detailed_results)
        df.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"💾 Details saved: {output_file}")

def load_dataset() -> Optional[pd.DataFrame]:
    import warnings
    warnings.simplefilter(action='ignore', category=UserWarning) 
    for path in DEFAULT_DATASET_PATHS:
        if os.path.exists(path):
            print(f"📂 Dataset: {path}")
            if path.endswith(".csv"): return pd.read_csv(path)
            elif path.endswith(".xlsx"): return pd.read_excel(path, engine="openpyxl")
    print("❌ Dataset not found!")
    return None

def main():
    print("--- EVALUATION SCRIPT (REAL BACKEND) ---")
    df = load_dataset()
    if df is None: return
    required_cols = ["Câu hỏi", "Nguồn tài liệu"]
    if not all(col in df.columns for col in required_cols):
        print(f"❌ Missing columns: {required_cols}")
        return
    evaluator = RAGEvaluator()
    evaluator.evaluate_dataset(df)
    evaluator.print_summary()
    evaluator.visualize_results()
    evaluator.save_details()

if __name__ == "__main__":
    main()
