"""
Advanced OCR Correction Module - Sử dụng AI (Gemini) để sửa lỗi OCR trong markdown.
Tích hợp với config và workflow của project.
"""
import os
import sys
import time
import re
import logging
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# Import config
CONFIG_DIR = os.path.join(PROJECT_ROOT, "backend", "config")
sys.path.insert(0, CONFIG_DIR)

try:
    from config import GEMINI_API_KEY, GEMINI_MODEL_NAME_CORRECTION
except ImportError:
    # Fallback
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    # Sử dụng gemini-2.5-flash-lite: model nhẹ, nhanh, phù hợp cho OCR correction
    GEMINI_MODEL_NAME_CORRECTION = os.getenv("GEMINI_MODEL_NAME_CORRECTION", "gemini-2.5-flash-lite")

# Import Gemini
import google.generativeai as genai

# Project directories
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DOCUMENTS_DIR = os.path.join(PROJECT_ROOT, "documents")
MARKDOWN_DIR = os.path.join(DOCUMENTS_DIR, "markdown")  # Thư mục chứa markdown từ Word/PDF

# Cấu hình Logging
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "correction.log"), encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class AdvancedOCRCorrector:
    """
    Class để sửa lỗi OCR trong văn bản tiếng Việt sử dụng Gemini AI.
    Logic "Triệt để": Validation loop + Adaptive prompting để đảm bảo khôi phục dấu hoàn toàn.
    """
    
    def __init__(self, api_key: str = None, model_name: str = None):
        """
        Khởi tạo corrector.
        
        Args:
            api_key: Gemini API key (nếu None thì lấy từ config)
            model_name: Model name (nếu None thì dùng từ config hoặc default)
        """
        self.api_key = api_key or GEMINI_API_KEY
        # Sử dụng gemini-2.5-flash-lite: model nhẹ, nhanh, phù hợp cho OCR correction
        self.model_name = model_name or GEMINI_MODEL_NAME_CORRECTION or "gemini-2.5-flash-lite"
        
        if not self.api_key:
            raise ValueError(
                "⚠️ Vui lòng cung cấp GEMINI_API_KEY trong config/config.py "
                "hoặc set biến môi trường GEMINI_API_KEY!"
            )
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)
        logging.info(f"✅ Đã khởi tạo với model: {self.model_name}")
    
    def _is_valid_vietnamese(self, text: str, threshold: float = 0.15) -> bool:
        """
        Kiểm tra xem văn bản có đủ dấu tiếng Việt không.
        
        Args:
            text: Văn bản cần kiểm tra
            threshold: Ngưỡng tỷ lệ ký tự có dấu tối thiểu (mặc định 15%)
        
        Returns:
            True nếu văn bản có đủ dấu, False nếu không
        """
        if not text.strip():
            return True  # Văn bản rỗng coi như hợp lệ
        
        # Đếm các nguyên âm có dấu tiếng Việt
        vietnamese_diacritics = set('áàảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ')
        # Đếm cả chữ hoa
        vietnamese_diacritics.update('ÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴĐ')
        
        # Đếm ký tự có dấu
        diacritic_count = sum(1 for char in text if char in vietnamese_diacritics)
        
        # Đếm tổng số ký tự chữ cái (a-z, A-Z, và tiếng Việt)
        import re
        letter_count = len(re.findall(r'[a-zA-ZàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴĐ]', text))
        
        if letter_count == 0:
            return True  # Không có chữ cái, coi như hợp lệ
        
        # Tính tỷ lệ
        ratio = diacritic_count / letter_count
        
        # Log để debug
        if ratio < threshold:
            logging.debug(f"⚠️ Tỷ lệ dấu thấp: {ratio:.2%} (ngưỡng: {threshold:.2%})")
        
        return ratio >= threshold
        
    def _create_correction_prompt(self, text_chunk: str, context: str = None) -> str:
        """
        Tạo prompt chuyên sâu cho kỹ thuật Prompt Engineering (Vòng 1 - Nhẹ nhàng).
        
        Args:
            text_chunk: Đoạn văn bản cần sửa
            context: Context từ chunk trước (tùy chọn)
        
        Returns:
            Prompt đã được format
        """
        context_section = ""
        if context:
            context_section = f"\n[Context từ đoạn trước - để tham khảo ngữ cảnh]:\n{context}\n\n"
        
        return f"""Tôi cần bạn giúp sửa lỗi OCR trong văn bản tiếng Việt nhằm phục vụ cho dữ liệu luật cho dự án chatbot dùng RAG, tôi cần bạn sửa lại đoạn thông tin luật tôi cung cấp.
        Việc này không phải vi phạm bản quyền bởi vì tôi thu thập dữ liệu từ văn bản pháp luật chính thống nhưng sau khi đọc và xử lý bằng OCR thì bị sai chính tả, sai dấu, sai từ. Tôi cần sửa lại theo đúng văn bản gốc.
        Hãy tưởng tượng bạn là một chuyên gia biên tập văn bản pháp luật Việt Nam. Nhiệm vụ của bạn là khôi phục và sửa lỗi chính tả cho đoạn văn bản OCR bị lỗi dưới đây.

        YÊU CẦU TUYỆT ĐỐI:
1. Sửa lỗi chính tả (ví dụ: "lura chon" -> "lựa chọn", "t? churc" -> "tổ chức", "T? churc" -> "tổ chức").
2. **QUAN TRỌNG: Khôi phục dấu tiếng Việt bị thiếu dựa trên ngữ cảnh pháp lý. Mỗi từ tiếng Việt PHẢI có dấu đúng.**
3. Sửa lỗi ký tự bị miss (ví dụ: "churc" -> "chức", "thau" -> "thầu").
4. Giữ nguyên cấu trúc gốc (xuống dòng, số thứ tự 1. 2. 3., dấu chấm câu, markdown headers).
5. KHÔNG thêm lời dẫn, KHÔNG giải thích, chỉ trả về văn bản đã sửa.
6. Nếu là tên riêng hoặc địa danh, hãy viết hoa đúng chuẩn.
7. Giữ nguyên format markdown (headers với #, lists với -, etc.).
{context_section}
        Văn bản gốc bị lỗi:
        ---
        {text_chunk}
        ---

Văn bản đã sửa (chỉ trả về text, không có lời giải thích):"""
    
    def _create_diacritic_restoration_prompt(self, text_chunk: str, context: str = None) -> str:
        """
        Tạo prompt CỰC GẮT chuyên về khôi phục dấu (Vòng 2 - Khi vòng 1 thất bại).
        
        Args:
            text_chunk: Đoạn văn bản cần sửa
            context: Context từ chunk trước (tùy chọn)
        
        Returns:
            Prompt đã được format với yêu cầu cực gắt về dấu
        """
        context_section = ""
        if context:
            context_section = f"\n[Context từ đoạn trước - để tham khảo ngữ cảnh]:\n{context}\n\n"
        
        return f"""NHIỆM VỤ: PHỤC HỒI DẤU TIẾNG VIỆT CHO VĂN BẢN PHÁP LUẬT

Bạn là chuyên gia ngôn ngữ tiếng Việt. Văn bản dưới đây đã bị mất dấu hoàn toàn do lỗi OCR.
Việc này không phải vi phạm bản quyền bởi vì tôi thu thập dữ liệu từ văn bản pháp luật chính thống nhưng sau khi đọc và xử lý bằng OCR thì bị sai chính tả, sai dấu, sai từ. Tôi cần sửa lại theo đúng văn bản gốc.
Nhiệm vụ của bạn là THÊM DẤU cho TẤT CẢ các từ tiếng Việt dựa trên ngữ cảnh pháp lý.

YÊU CẦU TUYỆT ĐỐI - KHÔNG ĐƯỢC BỎ SÓT:
1. **TUYỆT ĐỐI KHÔNG** để từ tiếng Việt nào không có dấu (ví dụ: "Cung hoa xa hoi" -> "Cộng hòa xã hội", "dau thau" -> "đấu thầu").
2. **TUYỆT ĐỐI KHÔNG** thay đổi cấu trúc câu, chỉ thêm dấu.
3. **TUYỆT ĐỐI KHÔNG** thay đổi số, ký tự đặc biệt, markdown format.
4. Mỗi từ tiếng Việt PHẢI có dấu đúng chuẩn (á, à, ả, ã, ạ, ă, â, ê, ô, ơ, ư, đ...).
5. Dựa vào ngữ cảnh pháp lý để suy luận dấu đúng (ví dụ: "dau thau" trong ngữ cảnh pháp luật -> "đấu thầu").
6. Giữ nguyên hoàn toàn: xuống dòng, số thứ tự, dấu chấm câu, markdown headers (#, -, *).
7. KHÔNG thêm lời dẫn, KHÔNG giải thích, chỉ trả về văn bản đã thêm dấu.
{context_section}
Văn bản KHÔNG DẤU cần phục hồi:
---
{text_chunk}
---

Văn bản ĐÃ CÓ DẤU (chỉ trả về text, không có lời giải thích):"""

    def _extract_text_from_response(self, response, original_text: str) -> Optional[str]:
        """
        Trích xuất text từ response, xử lý các trường hợp RECITATION (finish_reason = 4).
        
        Args:
            response: Response object từ Gemini API
            original_text: Văn bản gốc (để fallback nếu cần)
        
        Returns:
            Text đã được trích xuất và làm sạch, hoặc None nếu không thể trích xuất
        """
        if not hasattr(response, 'candidates') or not response.candidates:
            return None
        
        finish_reason = response.candidates[0].finish_reason
        
        # Nếu có text, dùng text đó (dù finish_reason = 4)
        if hasattr(response, 'text') and response.text:
            corrected = response.text.strip()
            # Loại bỏ các marker như "---" nếu có
            if "---" in corrected:
                corrected = corrected.split("---")[-1].strip()
            return corrected
        
        # Nếu không có text và là RECITATION, trả về None để retry
        if finish_reason == 4:
            logging.debug("   ⚠️ RECITATION detected nhưng không có text, sẽ retry")
            return None
        
        # Các trường hợp khác không có text
        return None

    def correct_text_segment(self, text: str, retries: int = 5, context: str = None) -> str:
        """
        Phiên bản sửa lỗi OCR với Validation Loop và Adaptive Prompting (Logic "Triệt để").
        
        Flow:
        1. Vòng 1: Dùng prompt nhẹ nhàng (_create_correction_prompt)
        2. Kiểm tra validation (_is_valid_vietnamese)
        3. Nếu thất bại: Vòng 2 dùng prompt cực gắt (_create_diacritic_restoration_prompt)
        4. Lặp lại cho đến khi đạt yêu cầu hoặc hết retries
        
        Args:
            text: Văn bản cần sửa
            retries: Số lần thử tối đa
            context: Context từ chunk trước (để tham khảo ngữ cảnh)
        """
        if not text.strip():
            return text
        
        # Cấu hình an toàn để giảm thiểu việc bị chặn
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        response = None
        use_aggressive_prompt = False  # Flag để chuyển sang prompt cực gắt
        
        for attempt in range(retries):
            try:
                # Adaptive Prompting: Attempt 0 dùng prompt nhẹ (trừ khi đã set flag), các attempt sau dùng prompt cực gắt
                if attempt == 0 and not use_aggressive_prompt:
                    prompt = self._create_correction_prompt(text, context)
                    logging.debug(f"   [Attempt {attempt + 1}] Sử dụng prompt nhẹ nhàng (sửa lỗi OCR)")
                else:
                    prompt = self._create_diacritic_restoration_prompt(text, context)
                    logging.debug(f"   [Attempt {attempt + 1}] Sử dụng prompt CỰC GẮT (khôi phục dấu)")
                
                response = self.model.generate_content(
                    prompt, 
                    safety_settings=safety_settings
                )
                
                # --- XỬ LÝ RECITATION (Finish Reason 4) ---
                corrected = self._extract_text_from_response(response, text)
                
                # Nếu không thể trích xuất text, xử lý retry
                if corrected is None:
                    # RECITATION không có text hoặc lỗi khác, retry nếu còn cơ hội
                    if attempt < retries - 1:
                        use_aggressive_prompt = True
                        time.sleep(2)
                        continue
                    # Hết retries, trả về gốc
                    logging.warning(f"   ⚠️ Không thể trích xuất text sau {retries} lần thử")
                    return text
                
                # --- VALIDATION LOOP: Kiểm tra xem văn bản có đủ dấu không ---
                # Adaptive threshold: giảm nhẹ threshold sau mỗi attempt để linh hoạt hơn
                adaptive_threshold = 0.15 * (1 - attempt * 0.03)  # Giảm 3% mỗi attempt, tối thiểu 0.10
                adaptive_threshold = max(0.10, adaptive_threshold)
                
                if self._is_valid_vietnamese(corrected, threshold=adaptive_threshold):
                    # ✅ Đạt yêu cầu, trả về kết quả
                    logging.debug(f"   ✅ Validation passed: Văn bản có đủ dấu (threshold: {adaptive_threshold:.2%})")
                    return corrected
                else:
                    # ❌ Không đủ dấu, chuyển sang prompt cực gắt và retry
                    logging.debug(f"   ⚠️ Validation failed: Văn bản thiếu dấu (threshold: {adaptive_threshold:.2%}), chuyển sang prompt cực gắt...")
                    use_aggressive_prompt = True
                    if attempt < retries - 1:
                        time.sleep(2)  # Delay trước khi retry
                        continue
                    else:
                        # Hết retries, trả về kết quả cuối cùng (dù chưa đạt yêu cầu)
                        logging.warning(f"   ⚠️ Hết retries, trả về kết quả chưa hoàn hảo")
                        return corrected
                    
            except Exception as e:
                err_msg = str(e)
                
                # Nếu gặp lỗi "Invalid operation... finish_reason is 4" trong Exception
                if "finish_reason" in err_msg and "4" in err_msg:
                    if response is not None:
                        try:
                            corrected = self._extract_text_from_response(response, text)
                            if corrected is not None:
                                # Kiểm tra validation
                                if self._is_valid_vietnamese(corrected):
                                    return corrected
                                # Nếu không đạt, chuyển sang prompt cực gắt
                                use_aggressive_prompt = True
                                if attempt < retries - 1:
                                    time.sleep(2)
                                    continue
                                return corrected
                        except (AttributeError, IndexError, KeyError) as extract_error:
                            logging.debug(f"   Lỗi khi trích xuất text từ exception handler: {extract_error}")
                            pass
                    return text
                
                # Nếu lỗi Rate Limit (429) thì mới retry với exponential backoff mạnh hơn
                if "429" in err_msg or "quota" in err_msg.lower() or "rate limit" in err_msg.lower():
                    # Exponential backoff: 15s, 30s, 60s, 120s, 180s (max)
                    wait_time = min(15 * (2 ** attempt), 180)
                    logging.warning(f"⏳ Quá tải (429). Đợi {wait_time}s trước khi thử lại (attempt {attempt + 1}/{retries})...")
                    time.sleep(wait_time)
                    # Tiếp tục retry nếu chưa hết số lần thử
                    if attempt < retries - 1:
                        continue
                else:
                    logging.error(f"Lỗi API khác: {err_msg}")
                    if attempt < retries - 1:
                        time.sleep(3)  # Delay nhẹ cho lỗi khác
                        continue
        
        # Nếu đến đây nghĩa là đã hết tất cả retries
        if response is not None:
            try:
                # Thử lấy text từ response cuối cùng nếu có
                corrected = self._extract_text_from_response(response, text)
                if corrected is not None:
                    logging.warning("⚠️  Sử dụng response cuối cùng (có thể chưa hoàn hảo)")
                    return corrected
            except (AttributeError, IndexError, KeyError) as extract_error:
                logging.debug(f"   Lỗi khi trích xuất text từ response cuối: {extract_error}")
                pass
        
        logging.error("❌ Không thể sửa đoạn này sau nhiều lần thử. Giữ nguyên gốc.")
        return text

    def process_file(self, input_path: str, output_path: str = None, 
                    target_chunks: int = 35, delay: float = 8.0):
        """
        Xử lý một file markdown để sửa lỗi OCR.
        Tối ưu: Gom nhiều đoạn văn lại thành chunks lớn để giảm số requests.
        
        Args:
            input_path: Đường dẫn file input
            output_path: Đường dẫn file output (nếu None thì tự động tạo)
            target_chunks: Số chunks mục tiêu (mặc định 35 để có chunks lớn hơn, giảm số request)
            delay: Delay giữa các request (giây, mặc định 8.0 để tránh rate limit)
        """
        path = Path(input_path)
        if not path.exists():
            logging.error(f"❌ File {input_path} không tồn tại.")
            return False

        # Tự động tạo output path nếu không có
        if output_path is None:
            output_path = str(path.parent / f"{path.stem}_corrected{path.suffix}")

        logging.info(f"📖 Đang đọc file: {input_path}")
        with open(path, "r", encoding="utf-8") as f:
            raw_content = f.read()

        file_size = len(raw_content)
        logging.info(f"📊 Kích thước file: {file_size:,} ký tự")
        
        # Tính toán chunk_size dựa trên target_chunks
        # Gemini-2.5-flash-live có giới hạn TPM: 1 triệu tokens/phút (rất cao!)
        # Không giới hạn RPM và RPD, nên có thể dùng chunks lớn hơn để giảm số request
        # Chunks lớn (3000-8000 ký tự) giúp AI có nhiều context hơn và giảm số request
        
        # Xử lý file nhỏ: nếu file quá nhỏ so với target_chunks, giảm số chunks mục tiêu
        if file_size < target_chunks * 500:
            # File nhỏ, điều chỉnh target_chunks để tránh chunks quá lớn
            adjusted_target_chunks = max(1, file_size // 500)
            estimated_chunk_size = max(500, min(file_size // adjusted_target_chunks, 8000))
        else:
            estimated_chunk_size = max(2000, file_size // target_chunks)
            estimated_chunk_size = min(estimated_chunk_size, 8000)  # Giới hạn tối đa 8000 ký tự (phù hợp với 1M TPM)
        
        logging.info(f"✂️  Đang gom văn bản thành chunks ~{estimated_chunk_size:,} ký tự/chunk...")
        
        # Tách thành các đoạn văn (paragraphs) dựa trên xuống dòng kép
        paragraphs = raw_content.split("\n\n")
        logging.info(f"   Tìm thấy {len(paragraphs)} đoạn văn gốc")
        
        # Gom các đoạn văn lại thành chunks lớn
        chunks = []
        current_chunk = []
        current_length = 0
        
        for para in paragraphs:
            para_length = len(para) + 2  # +2 cho "\n\n"
            
            # Xử lý paragraph quá lớn: nếu paragraph đơn lẻ lớn hơn chunk_size, cần chia nhỏ
            if para_length > estimated_chunk_size:
                # Chia paragraph lớn thành các câu hoặc dòng
                # Ưu tiên chia theo câu (dấu chấm, chấm hỏi, chấm than)
                sentences = re.split(r'([.!?]\s+)', para)
                # Gộp lại các câu với dấu câu của chúng
                sentence_parts = []
                for i in range(0, len(sentences) - 1, 2):
                    if i + 1 < len(sentences):
                        sentence_parts.append(sentences[i] + sentences[i + 1])
                    else:
                        sentence_parts.append(sentences[i])
                
                # Nếu vẫn quá lớn, chia theo dòng
                if not sentence_parts:
                    sentence_parts = para.split('\n')
                
                # Thêm các phần đã chia vào chunks
                for part in sentence_parts:
                    part_length = len(part) + 2
                    if current_length + part_length > estimated_chunk_size and current_chunk:
                        chunks.append("\n\n".join(current_chunk))
                        current_chunk = [part]
                        current_length = part_length
                    else:
                        current_chunk.append(part)
                        current_length += part_length
            elif current_length + para_length > estimated_chunk_size and current_chunk:
                # Nếu thêm đoạn này vượt quá chunk_size và đã có nội dung, lưu chunk hiện tại
                chunks.append("\n\n".join(current_chunk))
                current_chunk = [para]
                current_length = para_length
            else:
                current_chunk.append(para)
                current_length += para_length
        
        # Thêm chunk cuối cùng
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))
        
        logging.info(f"✅ Đã gom thành {len(chunks)} chunks (mục tiêu: {target_chunks})")
        if chunks:
            avg_size = sum(len(c) for c in chunks) // len(chunks)
            logging.info(f"   Trung bình: {avg_size:,} ký tự/chunk")
        else:
            logging.warning("   ⚠️ Không có chunks nào được tạo!")
        logging.info(f"🔄 Bắt đầu xử lý {len(chunks)} chunks...")
        
        corrected_chunks = []
        
        # Khởi tạo flag để theo dõi rate limit
        self._last_was_429 = False
        
        # Sử dụng tqdm để hiện thanh tiến trình
        previous_chunk_context = None  # Lưu context từ chunk trước để truyền vào
        for i, chunk in enumerate(tqdm(chunks, desc="Sửa lỗi OCR"), 1):
            if not chunk.strip():
                corrected_chunks.append("")
                previous_chunk_context = None
                continue
                
            logging.debug(f"   Xử lý chunk {i}/{len(chunks)} ({len(chunk):,} ký tự)")
            
            # Truyền context từ chunk trước để AI hiểu ngữ cảnh
            # Ưu tiên lấy context từ ranh giới pháp lý (Điều/Khoản) thay vì chỉ dựa vào dấu chấm
            context = None
            if previous_chunk_context:
                # Tìm ranh giới pháp lý (Điều, Khoản) trong context
                # Pattern: "Điều X." hoặc "Khoản X." hoặc "#### X."
                # Bao gồm cả chữ "đ" trong tiếng Việt
                legal_boundary_pattern = r'(?:###\s+)?Điều\s+\d+[a-zđ]?\.|(?:####\s+)?\d+\.\s+[A-ZÀ-ỹ]'
                legal_matches = list(re.finditer(legal_boundary_pattern, previous_chunk_context, re.IGNORECASE))
                
                if legal_matches:
                    # Lấy từ ranh giới pháp lý cuối cùng đến hết
                    last_match = legal_matches[-1]
                    context = previous_chunk_context[last_match.start():]
                    # Giới hạn độ dài context (tối đa 500 ký tự)
                    if len(context) > 500:
                        context = context[-500:]
                else:
                    # Fallback: Lấy 1-2 câu cuối dựa trên dấu chấm câu
                    sentences = previous_chunk_context.split('.')
                    if len(sentences) >= 2:
                        context = '. '.join(sentences[-2:]) + '.'
                    else:
                        context = previous_chunk_context[-200:]  # Lấy 200 ký tự cuối
            
            # Thử correct với retry và theo dõi rate limit
            # correct_text_segment sẽ tự xử lý rate limit và retry, nên không cần try-except ở đây
            corrected_text = self.correct_text_segment(chunk, context=context)
            
            # Kiểm tra nếu văn bản không được sửa (có thể do rate limit)
            # Nếu chunk giữ nguyên sau nhiều retry, có thể là do rate limit
            if corrected_text == chunk and len(chunk.strip()) > 100:
                # Chunk lớn mà không được sửa, có khả năng gặp rate limit
                # Tăng delay cho request tiếp theo để phòng ngừa
                self._last_was_429 = True
                logging.warning(f"   ⚠️  Chunk {i} không được sửa (có thể do rate limit), sẽ tăng delay cho chunk tiếp theo")
            
            corrected_chunks.append(corrected_text)
            
            # Lưu context cho chunk tiếp theo
            # Ưu tiên lấy từ ranh giới pháp lý cuối cùng (Điều/Khoản) để có ngữ cảnh có ý nghĩa hơn
            # Bao gồm cả chữ "đ" trong tiếng Việt
            legal_boundary_pattern = r'(?:###\s+)?Điều\s+\d+[a-zđ]?\.|(?:####\s+)?\d+\.\s+[A-ZÀ-ỹ]'
            legal_matches = list(re.finditer(legal_boundary_pattern, corrected_text, re.IGNORECASE))
            
            if legal_matches:
                # Lấy từ ranh giới pháp lý cuối cùng đến hết chunk
                last_match = legal_matches[-1]
                previous_chunk_context = corrected_text[last_match.start():]
                # Giới hạn độ dài (tối đa 500 ký tự)
                if len(previous_chunk_context) > 500:
                    previous_chunk_context = previous_chunk_context[-500:]
            else:
                # Fallback: Lấy 1-2 câu cuối dựa trên dấu chấm câu
                sentences = corrected_text.split('.')
                if len(sentences) >= 2:
                    previous_chunk_context = '. '.join(sentences[-2:]) + '.'
                else:
                    previous_chunk_context = corrected_text[-200:] if len(corrected_text) > 200 else corrected_text
            
            # Rate limiting (tránh spam API) - tăng delay để tránh rate limit
            if i < len(chunks):  # Không delay sau request cuối
                time.sleep(delay)
                # Nếu chunk vừa xử lý có lỗi 429, tăng delay cho request tiếp theo
                if hasattr(self, '_last_was_429') and self._last_was_429:
                    additional_delay = 5.0  # Thêm 5 giây nếu vừa gặp 429
                    logging.info(f"   ⏸️  Thêm delay {additional_delay}s do vừa gặp rate limit...")
                    time.sleep(additional_delay)
                    self._last_was_429 = False

        # Ghép lại
        final_content = "\n\n".join(corrected_chunks)
        
        # Tinh chỉnh format markdown (quan trọng cho embedding)
        logging.info("🧹 Đang tinh chỉnh format markdown...")
        original_size = len(final_content)
        final_content = self._optimize_markdown_format(final_content)
        optimized_size = len(final_content)
        size_reduction = original_size - optimized_size
        logging.info(f"   📊 Đã tối ưu: giảm {size_reduction:,} ký tự")

        # Lưu file
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8", newline='\n', buffering=8192) as f:
            f.write(final_content)
            
        logging.info(f"✅ Xử lý hoàn tất! Kết quả lưu tại: {output_path}")
        return True
    
    def _optimize_markdown_format(self, markdown_content: str) -> str:
        """
        Tối ưu và tinh chỉnh format markdown sau khi correction.
        Tương tự như hàm optimize_markdown_output trong read_pdf.py.
        
        Args:
            markdown_content: Nội dung markdown cần tối ưu
            
        Returns:
            Markdown đã được tối ưu
        """
        if not markdown_content:
            return ""
        
        # Loại bỏ số trang và header/footer - cải thiện để bắt nhiều pattern hơn
        # Pattern 1: "Trang X" hoặc "Page X" với các ký tự phân cách
        markdown_content = re.sub(r"=+\s*Trang\s*\d+\s*=+", "", markdown_content, flags=re.IGNORECASE)
        markdown_content = re.sub(r"-\s*\d+\s*-", "", markdown_content)
        markdown_content = re.sub(r"^\s*Page\s+\d+\s*$", "", markdown_content, flags=re.MULTILINE | re.IGNORECASE)
        
        # Pattern 2: Số trang đứng một mình trên dòng (chỉ số, có thể có ký tự đặc biệt)
        markdown_content = re.sub(r"^\s*[^\w]*\d+[^\w]*\s*$", "", markdown_content, flags=re.MULTILINE)
        
        # Pattern 3: Số trang ở đầu dòng với ký tự đặc biệt
        markdown_content = re.sub(r"^[^\w]*\d+[^\w]*\s+", "", markdown_content, flags=re.MULTILINE)
        
        # Pattern 4: Số trang ở cuối dòng với ký tự đặc biệt
        markdown_content = re.sub(r"\s+[^\w]*\d+[^\w]*$", "", markdown_content, flags=re.MULTILINE)
        
        # Pattern 5: Dòng chỉ chứa số và ký tự đặc biệt (header/footer)
        markdown_content = re.sub(r"^[^\w\s]*\d+[^\w\s]*$", "", markdown_content, flags=re.MULTILINE)
        
        # Loại bỏ markdown artifacts
        markdown_content = re.sub(r"```[\s\S]*?```", "", markdown_content)
        markdown_content = re.sub(r"!\[.*?\]\(.*?\)", "", markdown_content)
        
        # Chuẩn hóa khoảng trắng và line endings
        markdown_content = re.sub(r"\r\n", "\n", markdown_content)
        markdown_content = re.sub(r"\r", "\n", markdown_content)
        markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)
        markdown_content = re.sub(r"[ \t]+", " ", markdown_content)
        markdown_content = re.sub(r" +\n", "\n", markdown_content)
        
        # Chuẩn hóa headers markdown - đảm bảo có đúng một khoảng trắng sau #
        markdown_content = re.sub(r"^(#{1,6})([^\s#])", r"\1 \2", markdown_content, flags=re.MULTILINE)
        
        # Thêm phân cấp markdown cho cấu trúc pháp luật (quan trọng cho embedding)
        # Xử lý từng dòng để thêm headers phù hợp
        lines = markdown_content.split('\n')
        processed_lines = []
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            prev_line_empty = i == 0 or not lines[i-1].strip()
            next_line_empty = i == len(lines) - 1 or not lines[i+1].strip()
            
            # Chương, Phần, Mục → ## (level 2)
            if re.match(r"^(Chương|Phần|Mục)\s+([IVX\d]+|[A-Z])\s*[\.:]?\s*", line_stripped, re.IGNORECASE):
                line_stripped = re.sub(
                    r"^(Chương|Phần|Mục)\s+([IVX\d]+|[A-Z])\s*[\.:]?\s*(.+)?$",
                    r"## \1 \2\3",
                    line_stripped,
                    flags=re.IGNORECASE
                )
                # Đảm bảo có dòng trống trước (trừ khi ở đầu file)
                if not prev_line_empty:
                    processed_lines.append('')
                processed_lines.append(line_stripped)
                # Đảm bảo có dòng trống sau (trừ khi ở cuối file)
                if not next_line_empty:
                    processed_lines.append('')
                continue
            
            # Điều → ### (level 3) - chỉ khi chưa có header
            # Bao gồm cả chữ "đ" trong tiếng Việt (ví dụ: "Điều 1đ")
            if re.match(r"^Điều\s+\d+[a-zđ]?\s*[\.:]", line_stripped, re.IGNORECASE) and not line_stripped.startswith('#'):
                line_stripped = re.sub(
                    r"^Điều\s+(\d+[a-zđ]?)\s*[\.:]\s*(.+)?$",
                    r"### Điều \1. \2",
                    line_stripped,
                    flags=re.IGNORECASE
                )
                # Đảm bảo có dòng trống trước (trừ khi ở đầu file)
                if not prev_line_empty:
                    processed_lines.append('')
                processed_lines.append(line_stripped)
                # Đảm bảo có dòng trống sau (trừ khi ở cuối file)
                if not next_line_empty:
                    processed_lines.append('')
                continue
            
            # Khoản (số) → #### (level 4) - chỉ khi là khoản của điều (số đơn giản)
            if re.match(r"^(\d+)\.\s+[A-ZÀ-ỹ]", line_stripped) and len(line_stripped.split('.')[0].strip()) <= 2:
                line_stripped = re.sub(
                    r"^(\d+)\.\s+(.+)$",
                    r"#### \1. \2",
                    line_stripped
                )
                # Đảm bảo có dòng trống trước (trừ khi ở đầu file)
                if not prev_line_empty:
                    processed_lines.append('')
                processed_lines.append(line_stripped)
                # Đảm bảo có dòng trống sau (trừ khi ở cuối file)
                if not next_line_empty:
                    processed_lines.append('')
                continue
            
            # Điểm (a, b, c, d, đ, e, g...) → ##### (level 5)
            # Bao gồm cả chữ đ của tiếng Việt
            if re.match(r"^[a-zđ]\)\s+", line_stripped, re.IGNORECASE):
                line_stripped = re.sub(
                    r"^([a-zđ])\)\s+(.+)$",
                    r"##### \1) \2",
                    line_stripped,
                    flags=re.IGNORECASE
                )
                # Đảm bảo có dòng trống trước (trừ khi ở đầu file)
                if not prev_line_empty:
                    processed_lines.append('')
                processed_lines.append(line_stripped)
                # Đảm bảo có dòng trống sau (trừ khi ở cuối file)
                if not next_line_empty:
                    processed_lines.append('')
                continue
            
            # Giữ nguyên các dòng khác
            processed_lines.append(line)
        
        markdown_content = '\n'.join(processed_lines)
        
        # Chuẩn hóa khoảng trắng quanh Headers: đảm bảo có dòng trống trước và sau Header
        # Thay thế \n#### thành \n\n#### và ####\n thành ####\n\n
        markdown_content = re.sub(r'\n(#{1,6}\s)', r'\n\n\1', markdown_content)  # Thêm dòng trống trước header
        markdown_content = re.sub(r'(#{1,6}[^\n]*)\n(?!\n|#)', r'\1\n\n', markdown_content)  # Thêm dòng trống sau header (nếu chưa có)
        
        # Chuẩn hóa lại format sau khi thêm headers
        # Bao gồm cả chữ "đ" trong tiếng Việt
        markdown_content = re.sub(
            r"^(###\s+Điều\s+\d+[a-zđ]?)\s*\.\s*\.\s*",
            r"\1. ",
            markdown_content,
            flags=re.MULTILINE
        )
        
        # Loại bỏ ký tự control không hợp lệ
        markdown_content = re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]", "", markdown_content)
        
        # Chuẩn hóa dấu ngoặc kép tiếng Việt
        markdown_content = markdown_content.replace('"', '"').replace('"', '"')
        markdown_content = markdown_content.replace("'", "'").replace("'", "'")
        
        # Loại bỏ dòng trống ở đầu và cuối
        lines = markdown_content.split('\n')
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        
        markdown_content = '\n'.join(lines)
        
        if markdown_content and not markdown_content.endswith('\n'):
            markdown_content += '\n'
        
        return markdown_content.strip()

def process_all_markdown_files(documents_dir: str = DOCUMENTS_DIR, 
                               output_dir: str = None,
                               suffix: str = "_corrected"):
    """
    Xử lý tất cả file markdown trong thư mục documents.
    
    Args:
        documents_dir: Thư mục chứa file markdown
        output_dir: Thư mục output (nếu None thì ghi đè file gốc)
        suffix: Suffix để thêm vào tên file output
    """
    docs_path = Path(documents_dir)
    if not docs_path.exists():
        logging.error(f"❌ Thư mục {documents_dir} không tồn tại.")
        return
    
    # Tìm tất cả file .md
    md_files = list(docs_path.glob("*.md"))
    
    # Loại bỏ file đã được sửa (có suffix)
    md_files = [f for f in md_files if not f.stem.endswith(suffix)]
    
    if not md_files:
        logging.info(f"⚠️  Không tìm thấy file .md nào trong {markdown_dir}")
        return
    
    logging.info(f"🔍 Tìm thấy {len(md_files)} file markdown để xử lý\n")
    logging.info("=" * 60)
    
    # Khởi tạo corrector
    try:
        corrector = AdvancedOCRCorrector()
    except ValueError as e:
        logging.error(str(e))
        return
    
    processed_count = 0
    for md_file in md_files:
        try:
            if output_dir:
                output_path = Path(output_dir) / f"{md_file.stem}{suffix}{md_file.suffix}"
            else:
                output_path = md_file.parent / f"{md_file.stem}{suffix}{md_file.suffix}"
            
            logging.info(f"\n📄 Xử lý: {md_file.name}")
            success = corrector.process_file(str(md_file), str(output_path))
            if success:
                processed_count += 1
        except Exception as e:
            logging.error(f"❌ Lỗi khi xử lý {md_file}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    logging.info("\n" + "=" * 60)
    logging.info(f"✅ Hoàn tất! Đã xử lý {processed_count}/{len(md_files)} file")

# --- MAIN ---
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Sửa lỗi OCR trong file markdown sử dụng Gemini AI")
    parser.add_argument("--input", "-i", type=str, help="Đường dẫn file input (nếu không có thì xử lý tất cả .md trong documents/)")
    parser.add_argument("--output", "-o", type=str, help="Đường dẫn file output")
    parser.add_argument("--all", "-a", action="store_true", help="Xử lý tất cả file .md trong documents/")
    parser.add_argument("--chunks", "-c", type=int, default=35, help="Số chunks mục tiêu (mặc định 35 để có chunks lớn hơn, giảm số request)")
    parser.add_argument("--delay", "-d", type=float, default=8.0, help="Delay giữa các request (giây, mặc định 8.0 để tránh rate limit)")
    
    args = parser.parse_args()
    
    if args.all or not args.input:
        # Xử lý tất cả files
        process_all_markdown_files()
    else:
        # Xử lý file cụ thể
        try:
            corrector = AdvancedOCRCorrector()
            corrector.process_file(args.input, args.output, target_chunks=args.chunks, delay=args.delay)
        except ValueError as e:
            logging.error(str(e))
            sys.exit(1)
