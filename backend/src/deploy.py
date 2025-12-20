"""
Script để test query từ terminal.
Nhận text input, tìm kiếm top k chunks và log vào folder log.
"""
import sys
import os
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "backend", "src"))

# Import query module (từ cùng thư mục)
import query

def create_log_folder():
    """Tạo folder log nếu chưa có."""
    log_dir = Path(PROJECT_ROOT) / "log"
    log_dir.mkdir(exist_ok=True)
    return log_dir

def log_chunks_to_file(query_text: str, chunks: list, log_dir: Path):
    """Log các chunks vào file."""
    # Tạo tên file với timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Sanitize query text để làm tên file (chỉ lấy 50 ký tự đầu, loại bỏ ký tự đặc biệt)
    safe_query = "".join(c for c in query_text[:50] if c.isalnum() or c in (' ', '-', '_')).strip()
    safe_query = safe_query.replace(' ', '_')
    filename = f"query_{timestamp}_{safe_query}.txt"
    log_file = log_dir / filename
    
    # Ghi vào file
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"QUERY: {query_text}\n")
        f.write(f"TIMESTAMP: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"TOTAL CHUNKS: {len(chunks)}\n")
        f.write("=" * 80 + "\n\n")
        
        for idx, chunk in enumerate(chunks, 1):
            f.write(f"\n{'='*80}\n")
            f.write(f"CHUNK #{idx}\n")
            f.write(f"{'='*80}\n")
            
            # Text
            chunk_text = chunk.get("text", "")
            f.write(f"\nTEXT:\n{chunk_text}\n")
            
            # Scores
            f.write(f"\nSCORES:\n")
            if chunk.get("final_score") is not None:
                f.write(f"  - Final Score: {chunk.get('final_score', 0.0):.4f}\n")
            if chunk.get("cross_score") is not None:
                f.write(f"  - Cross-Encoder Score: {chunk.get('cross_score', 0.0):.4f}\n")
            if chunk.get("hybrid_score") is not None:
                f.write(f"  - Hybrid Score (RRF): {chunk.get('hybrid_score', 0.0):.4f}\n")
            if chunk.get("keyword_score") is not None:
                f.write(f"  - Keyword Score: {chunk.get('keyword_score', 0.0):.4f}\n")
            if chunk.get("metadata_score") is not None:
                f.write(f"  - Metadata Score: {chunk.get('metadata_score', 0.0):.4f}\n")
            
            # Metadata
            f.write(f"\nMETADATA:\n")
            if chunk.get("article"):
                f.write(f"  - Article: {chunk['article']}\n")
            if chunk.get("article_number"):
                f.write(f"  - Article Number: {chunk['article_number']}\n")
            if chunk.get("clause"):
                f.write(f"  - Clause: {chunk['clause']}\n")
            if chunk.get("point"):
                f.write(f"  - Point: {chunk['point']}\n")
            if chunk.get("chapter"):
                f.write(f"  - Chapter: {chunk['chapter']}\n")
            if chunk.get("source_file"):
                f.write(f"  - Source File: {chunk['source_file']}\n")
            if chunk.get("chunk_idx") is not None:
                f.write(f"  - Chunk Index: {chunk['chunk_idx']}\n")
            
            # Word count
            word_count = chunk.get("word_count", len(chunk_text.split()))
            f.write(f"  - Word Count: {word_count}\n")
            
            f.write(f"\n")
    
    return log_file

def main():
    """Main function để chạy query testing."""
    print("=" * 80)
    print("📘 QUERY TEST - RETRIEVAL ONLY")
    print("=" * 80)
    print("\nHướng dẫn:")
    print("  - Nhập query text và nhấn Enter để tìm kiếm top k chunks")
    print("  - Kết quả sẽ được log vào folder 'log/'")
    print("  - Nhập 'exit' hoặc 'quit' để thoát")
    print("=" * 80)
    
    # Tạo folder log
    log_dir = create_log_folder()
    print(f"\n📁 Log folder: {log_dir}\n")
    
    while True:
        try:
            # Nhận input từ user
            query_text = input("\n💬 Nhập query text: ").strip()
            
            if not query_text:
                continue
            
            # Xử lý lệnh thoát
            if query_text.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Tạm biệt!")
                break
            
            # Tìm kiếm chunks với metadata
            print(f"\n🔍 Đang tìm kiếm chunks cho query: '{query_text}'...")
            chunks = query.search_faiss(
                query_text,
                top_k=3,  # Top 3 chunks
                use_multi_stage=True,
                return_metadata=True
            )
            
            if not chunks:
                print("⚠️  Không tìm thấy chunks nào!")
                continue
            
            # Log vào file
            log_file = log_chunks_to_file(query_text, chunks, log_dir)
            
            # Thông báo thành công
            print(f"\n✅ Đã log {len(chunks)} chunks vào file:")
            print(f"   📄 {log_file}")
            print(f"   📊 Top {len(chunks)} chunks đã được lưu")
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Đã dừng (Ctrl+C)")
            print("👋 Tạm biệt!")
            break
        except Exception as e:
            print(f"\n❌ Lỗi: {str(e)}")
            import traceback
            traceback.print_exc()
            print("\n💡 Thử lại với query khác hoặc nhập 'exit' để thoát")

if __name__ == "__main__":
    main()

