"""
Module đọc PDF sử dụng pytesseract OCR với tối ưu tối đa cho tiếng Việt.
Tự động chọn cấu hình tốt nhất để có độ chính xác cao nhất.
"""

import os
import sys
import re
from pathlib import Path
from typing import Optional, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

# Import tqdm cho progress bar (optional)
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DOCUMENTS_DIR = os.path.join(PROJECT_ROOT, "documents")
MARKDOWN_DIR = os.path.join(DOCUMENTS_DIR, "markdown")  # Thư mục lưu markdown từ PDF

# Tạo thư mục markdown nếu chưa có
os.makedirs(MARKDOWN_DIR, exist_ok=True)

# Import pytesseract và pdf2image (bắt buộc)
try:
    import pytesseract
    from pdf2image import convert_from_path
    from PIL import Image, ImageEnhance, ImageOps, ImageFilter
    import numpy as np
except ImportError as e:
    logger.error(f"❌ Lỗi import: {e}")
    logger.error("Vui lòng cài đặt: pip install pytesseract pdf2image pillow numpy")
    sys.exit(1)

# Optional imports cho advanced preprocessing (ảnh chụp/scan)
try:
    from scipy import ndimage
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    logger.debug("⚠️  scipy không có - một số tính năng preprocessing sẽ bị giảm")

try:
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.debug("⚠️  sklearn không có - deskewing sẽ dùng phương pháp đơn giản hơn")

# Cache cho language detection (tránh gọi subprocess nhiều lần)
_cached_language = None

def _deskew_image(image: Image.Image, max_angle: float = 5.0) -> Image.Image:
    """
    Chỉnh góc nghiêng (deskew) của ảnh - quan trọng cho ảnh chụp.
    
    Args:
        image: PIL Image (grayscale)
        max_angle: Góc tối đa để detect (độ)
    
    Returns:
        PIL Image đã được chỉnh góc
    """
    try:
        img_array = np.array(image)
        
        if not HAS_SCIPY:
            # Nếu không có scipy, bỏ qua deskewing (quá phức tạp)
            return image
        
        # Edge detection: tìm các cạnh của chữ
        # Sử dụng Sobel filter để detect edges
        sobel_x = ndimage.sobel(img_array, axis=1)
        sobel_y = ndimage.sobel(img_array, axis=0)
        edges = np.hypot(sobel_x, sobel_y)
        edges = (edges > np.percentile(edges, 90)).astype(np.uint8) * 255
        
        # Sample một số điểm edge để tính góc
        edge_points = np.argwhere(edges > 0)
        if len(edge_points) < 100:
            # Không đủ edge points, không deskew
            return image
        
        # Tính góc bằng cách fit line qua các edge points
        if HAS_SKLEARN:
            # Sử dụng PCA để tìm hướng chính (chính xác nhất)
            pca = PCA(n_components=1)
            sample_size = min(1000, len(edge_points))
            pca.fit(edge_points[:sample_size])
            angle_rad = np.arctan2(pca.components_[0][1], pca.components_[0][0])
            angle_deg = np.degrees(angle_rad)
        else:
            # Phương pháp đơn giản hơn: tính góc từ các dòng text
            h, w = edges.shape
            angles = []
            for y in range(0, h, 20):  # Sample mỗi 20 pixels để nhanh hơn
                row = edges[y, :]
                if np.sum(row > 0) > 10:  # Có đủ edge points
                    indices = np.where(row > 0)[0]
                    if len(indices) > 1:
                        # Tính góc từ độ dài và vị trí
                        dx = indices[-1] - indices[0]
                        if dx > w * 0.1:  # Chỉ tính nếu line đủ dài
                            # Góc nhỏ, ước tính từ sự thay đổi
                            angles.append(0)  # Giả định horizontal, có thể cải thiện
        
            if not angles:
                return image
            angle_deg = np.mean(angles) if angles else 0
        
        # Giới hạn góc trong khoảng -max_angle đến max_angle
        if abs(angle_deg) > max_angle:
            return image  # Góc quá lớn, không deskew
        
        # Rotate ảnh để chỉnh góc
        rotated = image.rotate(-angle_deg, expand=False, fillcolor=255)
        return rotated
        
    except Exception:
        # Nếu có lỗi, trả về ảnh gốc
        return image


@lru_cache(maxsize=1)
def check_tesseract_vietnamese_support():
    """Kiểm tra Tesseract có hỗ trợ tiếng Việt và trả về ngôn ngữ tốt nhất (cached)."""
    global _cached_language
    if _cached_language is not None:
        return _cached_language
    
    try:
        import subprocess
        result = subprocess.run(
            ['tesseract', '--list-langs'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            langs = result.stdout.lower()
            # Ưu tiên vie_best > vie > vi
            if 'vie_best' in langs:
                logger.info("✅ Tìm thấy vie_best - sử dụng để có độ chính xác cao nhất")
                _cached_language = 'vie_best'
                return 'vie_best'
            elif 'vie' in langs:
                logger.info("✅ Tìm thấy vie - sử dụng cho OCR tiếng Việt")
                _cached_language = 'vie'
                return 'vie'
            elif 'vi' in langs:
                logger.warning("⚠️  Chỉ tìm thấy vi (không phải vie)")
                _cached_language = 'vi'
                return 'vi'
            else:
                logger.error("❌ Tesseract không có gói ngôn ngữ tiếng Việt!")
                logger.error("   Tải vie.traineddata từ: https://github.com/tesseract-ocr/tessdata")
                _cached_language = None
                return None
        else:
            logger.error("❌ Không thể kiểm tra Tesseract")
            _cached_language = None
            return None
    except Exception as e:
        logger.error(f"❌ Lỗi khi kiểm tra Tesseract: {e}")
        logger.error("   Đảm bảo Tesseract đã được cài đặt và có trong PATH")
        _cached_language = None
        return None


def optimize_markdown_output(markdown_content: str) -> str:
    """Tối ưu markdown output sau khi OCR."""
    if not markdown_content:
        return ""
    
    # Loại bỏ số trang và header/footer - cải thiện để bắt nhiều pattern hơn
    # Pattern 1: "Trang X" hoặc "Page X" với các ký tự phân cách
    markdown_content = re.sub(r"=+\s*Trang\s*\d+\s*=+", "", markdown_content, flags=re.IGNORECASE)
    markdown_content = re.sub(r"-\s*\d+\s*-", "", markdown_content)
    markdown_content = re.sub(r"^\s*Page\s+\d+\s*$", "", markdown_content, flags=re.MULTILINE | re.IGNORECASE)
    
    # Pattern 2: Số trang đứng một mình trên dòng (chỉ số, có thể có ký tự đặc biệt)
    # Ví dụ: "72", "72 m", "ˆ 7", "74"
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
    
    # Chuẩn hóa headers markdown
    markdown_content = re.sub(r"^(#{1,6})([^\s#])", r"\1 \2", markdown_content, flags=re.MULTILINE)
    
    # Thêm phân cấp markdown cho cấu trúc pháp luật (quan trọng cho embedding)
    # Xử lý từng dòng để thêm headers phù hợp
    lines = markdown_content.split('\n')
    processed_lines = []
    
    for line in lines:
        line_stripped = line.strip()
        
        # Chương, Phần, Mục → ## (level 2)
        if re.match(r"^(Chương|Phần|Mục)\s+([IVX\d]+|[A-Z])\s*[\.:]?\s*", line_stripped, re.IGNORECASE):
            line_stripped = re.sub(
                r"^(Chương|Phần|Mục)\s+([IVX\d]+|[A-Z])\s*[\.:]?\s*(.+)?$",
                r"## \1 \2\3",
                line_stripped,
                flags=re.IGNORECASE
            )
            processed_lines.append(line_stripped)
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
            processed_lines.append(line_stripped)
            continue
        
        # Khoản (số) → #### (level 4) - chỉ khi là khoản của điều (số đơn giản)
        if re.match(r"^(\d+)\.\s+[A-ZÀ-ỹ]", line_stripped) and len(line_stripped.split('.')[0].strip()) <= 2:
            line_stripped = re.sub(
                r"^(\d+)\.\s+(.+)$",
                r"#### \1. \2",
                line_stripped
            )
            processed_lines.append(line_stripped)
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
            processed_lines.append(line_stripped)
            continue
        
        # Giữ nguyên các dòng khác
        processed_lines.append(line)
    
    markdown_content = '\n'.join(processed_lines)
    
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


def read_pdf_to_markdown(
    pdf_path: str,
    output_path: Optional[str] = None,
    dpi: int = 400,
    language: Optional[str] = None
) -> str:
    """
    Đọc PDF và chuyển đổi sang markdown với độ chính xác cao nhất.
    
    Args:
        pdf_path: Đường dẫn đến file PDF
        output_path: Đường dẫn file output (nếu None, tự động tạo)
        dpi: Độ phân giải (400 = cân bằng tốt, 600 = chính xác nhất nhưng chậm)
        language: Mã ngôn ngữ (nếu None, tự động chọn tốt nhất)
    
    Returns:
        Đường dẫn đến file markdown đã tạo
    """
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"❌ Không tìm thấy file: {pdf_path}")
    
    if not pdf_path.suffix.lower() == '.pdf':
        raise ValueError(f"❌ File không phải PDF: {pdf_path}")
    
    # Tạo output path - lưu vào documents/markdown/
    if output_path is None:
        output_path = Path(MARKDOWN_DIR) / f"{pdf_path.stem}.md"
    else:
        output_path = Path(output_path)
    
    logger.info(f"📄 Đang xử lý: {pdf_path.name}")
    logger.info(f"   Kích thước: {pdf_path.stat().st_size / (1024*1024):.2f} MB")
    
    # Tự động chọn ngôn ngữ tốt nhất
    if language is None:
        language = check_tesseract_vietnamese_support()
        if language is None:
            raise RuntimeError("❌ Không thể tìm thấy ngôn ngữ tiếng Việt cho Tesseract")
    
    if output_path.exists():
        logger.info(f"   ⚠️  File markdown đã tồn tại, sẽ ghi đè: {output_path}")
    
    try:
        # Trích xuất ảnh từ PDF (PDF scan đã là ảnh, chỉ cần extract từng trang)
        # DPI chỉ ảnh hưởng chất lượng render khi extract, không tạo ảnh mới
        logger.info(f"   📤 Đang trích xuất trang ảnh từ PDF (render DPI={dpi})...")
        
        # Tối ưu: sử dụng thread_count dựa trên số cores
        thread_count = min(4, os.cpu_count() or 1)
        
        images = convert_from_path(
            str(pdf_path),
            dpi=dpi,
            fmt='png',
            thread_count=thread_count
        )
        
        logger.info(f"   📄 Đã trích xuất {len(images)} trang ảnh")
        
        # Cấu hình Tesseract tối ưu nhất cho tiếng Việt và ảnh chụp/scan
        # --oem 1: LSTM engine (tốt nhất cho tiếng Việt)
        # --psm 6: Uniform block of text (tốt cho văn bản pháp luật)
        # Thêm --dpi để Tesseract biết độ phân giải (quan trọng cho ảnh chụp)
        tesseract_config = f'--oem 1 --psm 6 --dpi {dpi}'
        
        logger.info(f"   🔄 Đang thực hiện OCR với pytesseract (lang={language}, DPI={dpi})...")
        logger.info(f"   ⚙️  Config: {tesseract_config} (LSTM engine - tối ưu nhất)")
        
        # Hàm xử lý một trang (để parallelize)
        def process_page(page_data: Tuple[int, Image.Image]) -> Tuple[int, str]:
            """Xử lý một trang PDF (ảnh chụp/scan) và trả về text với preprocessing tối ưu."""
            page_num, image = page_data
            
            # ========== TIỀN XỬ LÝ ẢNH CHO ẢNH CHỤP/SCAN ==========
            
            # 1. Grayscale conversion
            gray_image = image.convert('L')
            
            # 2. Noise reduction (quan trọng cho ảnh chụp có nhiễu)
            if HAS_SCIPY:
                # Median filter để loại bỏ nhiễu (tốt cho ảnh chụp)
                img_array = np.array(gray_image)
                img_array = ndimage.median_filter(img_array, size=3)
                gray_image = Image.fromarray(img_array, mode='L')
            else:
                # Fallback: dùng PIL filter đơn giản
                try:
                    gray_image = gray_image.filter(ImageFilter.MedianFilter(size=3))
                except:
                    pass  # Nếu không có filter, bỏ qua
            
            # 3. Deskewing (chỉnh góc nghiêng) - QUAN TRỌNG cho ảnh chụp
            try:
                gray_image = _deskew_image(gray_image)
            except Exception as e:
                # Nếu deskewing fail, tiếp tục với ảnh gốc
                pass
            
            # 4. Contrast enhancement (tăng độ tương phản)
            enhancer = ImageEnhance.Contrast(gray_image)
            gray_image = enhancer.enhance(1.3)  # Tăng 30% (tăng từ 20% cho ảnh chụp)
            
            # 5. Brightness adjustment (điều chỉnh độ sáng)
            enhancer = ImageEnhance.Brightness(gray_image)
            # Tự động điều chỉnh brightness dựa trên mean
            try:
                img_array = np.array(gray_image)
                mean_brightness = np.mean(img_array)
                # Nếu quá tối (< 100), tăng brightness; nếu quá sáng (> 200), giảm
                if mean_brightness < 100:
                    gray_image = enhancer.enhance(1.2)  # Tăng sáng
                elif mean_brightness > 200:
                    gray_image = enhancer.enhance(0.9)  # Giảm sáng
            except:
                pass
            
            # 6. Binarization: Chuyển sang ảnh nhị phân (đen/trắng) - CẢI THIỆN cho ảnh chụp
            try:
                img_array = np.array(gray_image)
                
                # Sử dụng Otsu threshold thực sự (tốt hơn cho ảnh chụp)
                # Otsu threshold: tìm threshold tối ưu để tách foreground/background
                hist, bins = np.histogram(img_array.flatten(), 256, [0, 256])
                # Tính Otsu threshold
                total = img_array.size
                sum_total = np.sum(np.arange(256) * hist)
                sum_bg = 0
                w_bg = 0
                max_var = 0
                threshold_value = 128
                
                for i in range(256):
                    w_bg += hist[i]
                    if w_bg == 0:
                        continue
                    w_fg = total - w_bg
                    if w_fg == 0:
                        break
                    sum_bg += i * hist[i]
                    mean_bg = sum_bg / w_bg
                    mean_fg = (sum_total - sum_bg) / w_fg
                    var_between = w_bg * w_fg * (mean_bg - mean_fg) ** 2
                    if var_between > max_var:
                        max_var = var_between
                        threshold_value = i
                
                # Chuyển sang nhị phân: > threshold = trắng (255), <= threshold = đen (0)
                binary_array = np.where(img_array < threshold_value, 0, 255).astype(np.uint8)
                
                # Morphological operations để loại bỏ nhiễu nhỏ (quan trọng cho ảnh chụp)
                if HAS_SCIPY:
                    # Opening: erosion + dilation để loại bỏ nhiễu nhỏ
                    binary_array = ndimage.binary_opening(binary_array > 0, structure=np.ones((2, 2))).astype(np.uint8) * 255
                
                gray_image = Image.fromarray(binary_array, mode='L')
            except (ImportError, AttributeError, Exception):
                # Fallback: dùng PIL ImageOps nếu không có numpy hoặc lỗi
                # Autocontrast + threshold đơn giản
                gray_image = ImageOps.autocontrast(gray_image, cutoff=2)
                # Threshold đơn giản với PIL
                threshold = 128  # Giá trị threshold mặc định
                gray_image = gray_image.point(lambda x: 0 if x < threshold else 255, mode='1').convert('L')
            
            # OCR với cấu hình tối ưu
            text = pytesseract.image_to_string(
                gray_image,
                lang=language,
                config=tesseract_config
            )
            
            return (page_num, text)
        
        # Parallel OCR processing (sử dụng ThreadPoolExecutor)
        # Tesseract có thể chạy parallel với thread-safe
        max_workers = min(4, len(images), os.cpu_count() or 1)  # Tối đa 4 workers hoặc số cores
        logger.info(f"   ⚡ Sử dụng {max_workers} workers để OCR song song...")
        
        all_text_dict = {}  # Dùng dict để giữ thứ tự
        
        # Sử dụng tqdm nếu có, nếu không thì dùng logging
        if HAS_TQDM:
            pbar = tqdm(total=len(images), desc="   OCR pages", unit="page", leave=False)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit tất cả tasks
            future_to_page = {
                executor.submit(process_page, (i, image)): i 
                for i, image in enumerate(images, 1)
            }
            
            # Collect results với progress
            completed = 0
            for future in as_completed(future_to_page):
                try:
                    page_num, text = future.result()
                    all_text_dict[page_num] = text
                    completed += 1
                    if HAS_TQDM:
                        pbar.update(1)
                    elif completed % 10 == 0 or completed == len(images):
                        logger.info(f"   📄 Đã OCR {completed}/{len(images)} trang...")
                except Exception as e:
                    page_idx = future_to_page[future]
                    logger.error(f"   ❌ Lỗi khi OCR trang {page_idx}: {e}")
                    all_text_dict[page_idx] = ""  # Thêm text rỗng nếu lỗi
                    if HAS_TQDM:
                        pbar.update(1)
        
        if HAS_TQDM:
            pbar.close()
        
        # Sắp xếp lại theo thứ tự trang
        all_text = [all_text_dict[i] for i in sorted(all_text_dict.keys())]
        
        # Ghép tất cả text lại
        markdown_content = '\n\n'.join(all_text)
        
        # Tối ưu markdown
        logger.info("   🧹 Đang tối ưu markdown...")
        original_size = len(markdown_content)
        markdown_content = optimize_markdown_output(markdown_content)
        optimized_size = len(markdown_content)
        
        # Lưu file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8', newline='\n', buffering=8192) as f:
            f.write(markdown_content)
        
        # Thống kê
        char_count = len(markdown_content)
        line_count = markdown_content.count('\n')
        word_count = len(re.findall(r'\b\w+\b', markdown_content))
        size_reduction = original_size - optimized_size
        
        logger.info(f"   ✅ Hoàn tất!")
        logger.info(f"   📊 Số ký tự: {char_count:,} (giảm {size_reduction:,} ký tự)")
        logger.info(f"   📊 Số dòng: {line_count:,}")
        logger.info(f"   📊 Số từ: {word_count:,}")
        logger.info(f"   💾 Đã lưu: {output_path}")
        
        return str(output_path)
        
    except KeyboardInterrupt:
        logger.warning(f"   ⚠️  Đã dừng xử lý {pdf_path.name} bởi người dùng")
        raise
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"   ❌ Lỗi input: {e}")
        raise
    except Exception as e:
        logger.error(f"   ❌ Lỗi khi xử lý {pdf_path.name}: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        raise


def process_legal_documents(
    documents_dir: str = DOCUMENTS_DIR,
    pdf_files: Optional[list] = None,
    dpi: int = 400
):
    """
    Xử lý file PDF văn bản luật đấu thầu với độ chính xác cao nhất.
    
    Args:
        documents_dir: Thư mục chứa documents
        pdf_files: Danh sách file PDF cụ thể (nếu None, đọc tất cả file PDF trong thư mục)
        dpi: Độ phân giải (400 = cân bằng, 600 = chính xác nhất)
    """
    documents_dir = Path(documents_dir)
    
    if not documents_dir.exists():
        raise FileNotFoundError(f"❌ Không tìm thấy thư mục: {documents_dir}")
    
    # Tìm các file PDF
    pdf_paths = []
    
    if pdf_files is None:
        # Đọc tất cả file PDF trong thư mục
        pdf_paths = list(documents_dir.glob("*.pdf"))
        if not pdf_paths:
            logger.error(f"❌ Không tìm thấy file PDF nào trong thư mục: {documents_dir}")
            return
        logger.info(f"📁 Tìm thấy {len(pdf_paths)} file PDF trong thư mục")
    else:
        # Chỉ đọc các file được chỉ định
        for pdf_file in pdf_files:
            pdf_path = documents_dir / pdf_file
            if pdf_path.exists():
                pdf_paths.append(pdf_path)
            else:
                logger.warning(f"⚠️  Không tìm thấy: {pdf_file}")
    
    if not pdf_paths:
        logger.error("❌ Không tìm thấy file PDF nào để xử lý!")
        return
    
    logger.info(f"🚀 Bắt đầu xử lý {len(pdf_paths)} file PDF với độ chính xác cao nhất...")
    logger.info("=" * 60)
    
    # Xử lý từng file
    results = []
    for i, pdf_path in enumerate(pdf_paths, 1):
        logger.info(f"\n[{i}/{len(pdf_paths)}] {pdf_path.name}")
        logger.info("-" * 60)
        
        try:
            output_path = read_pdf_to_markdown(
                pdf_path=pdf_path,
                dpi=dpi
            )
            results.append({
                'input': str(pdf_path),
                'output': output_path,
                'status': 'success'
            })
        except Exception as e:
            logger.error(f"❌ Lỗi: {e}")
            results.append({
                'input': str(pdf_path),
                'output': None,
                'status': 'error',
                'error': str(e)
            })
    
    # Tổng kết
    logger.info("\n" + "=" * 60)
    logger.info("📊 TỔNG KẾT")
    logger.info("=" * 60)
    
    success_count = sum(1 for r in results if r['status'] == 'success')
    error_count = len(results) - success_count
    
    logger.info(f"✅ Thành công: {success_count}/{len(results)}")
    if error_count > 0:
        logger.info(f"❌ Lỗi: {error_count}/{len(results)}")
    
    for result in results:
        if result['status'] == 'success':
            logger.info(f"   ✅ {Path(result['input']).name} → {Path(result['output']).name}")
        else:
            logger.info(f"   ❌ {Path(result['input']).name} - {result.get('error', 'Unknown error')}")


def main():
    """Hàm main để chạy script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Đọc PDF văn bản luật đấu thầu với OCR tiếng Việt chính xác nhất'
    )
    parser.add_argument(
        '--documents-dir',
        type=str,
        default=DOCUMENTS_DIR,
        help='Thư mục chứa documents (mặc định: documents/)'
    )
    parser.add_argument(
        '--files',
        nargs='+',
        help='Danh sách file PDF cụ thể cần xử lý (nếu không chỉ định, sẽ đọc tất cả file PDF trong thư mục)'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=400,
        choices=[300, 400, 600],
        help='Độ phân giải (400=cân bằng, 600=chính xác nhất nhưng chậm, mặc định: 400)'
    )
    
    args = parser.parse_args()
    
    try:
        process_legal_documents(
            documents_dir=args.documents_dir,
            pdf_files=args.files,
            dpi=args.dpi
        )
    except KeyboardInterrupt:
        logger.info("\n⚠️  Đã dừng bởi người dùng")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Lỗi: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
