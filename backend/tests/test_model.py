"""
Test script cho RAG system với Gemini API.
Test cả Gemini API trực tiếp và RAG system hoàn chỉnh.
"""
import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "backend", "src"))

# Import query module
from src import query
import google.generativeai as genai

def test_gemini_direct():
    """Test Gemini API trực tiếp (không qua RAG)."""
    print("=" * 60)
    print("🧪 TEST 1: Gemini API trực tiếp")
    print("=" * 60)
    
    try:
        # Import config
        CONFIG_DIR = os.path.join(PROJECT_ROOT, "backend", "config")
        sys.path.insert(0, CONFIG_DIR)
        try:
            from config import GEMINI_API_KEY, GEMINI_MODEL_NAME
        except ImportError:
            GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
            if not GEMINI_API_KEY:
                print("⚠️  GEMINI_API_KEY not found in environment variables")
                return False
            GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-pro")
        
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        
        test_prompt = "Luật đấu thầu là gì?"
        response = model.generate_content(test_prompt)
        
        print("✅ Gemini API hoạt động!")
        print(f"Response: {response.text[:200]}...")
        return True
    except Exception as e:
        print(f"❌ Lỗi khi test Gemini API: {e}")
        return False

def test_rag_system():
    """Test RAG system hoàn chỉnh."""
    print("\n" + "=" * 60)
    print("🧪 TEST 2: RAG System")
    print("=" * 60)
    
    try:
        # Test search
        test_query = "Điều kiện tham gia đấu thầu"
        results = query.search_faiss(test_query, top_k=3)
        
        if results:
            print(f"✅ Search hoạt động! Tìm thấy {len(results)} kết quả")
            print(f"Kết quả đầu tiên: {results[0][:200]}...")
        else:
            print("⚠️  Không tìm thấy kết quả (có thể chưa có index)")
            return False
        
        # Test ask_sth
        answer = query.ask_sth(test_query, return_metadata=False)
        if answer:
            print(f"✅ Ask_sth hoạt động!")
            print(f"Answer: {answer[:200]}...")
            return True
        else:
            print("❌ Ask_sth không trả về kết quả")
            return False
            
    except Exception as e:
        print(f"❌ Lỗi khi test RAG system: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Bắt đầu test RAG system...\n")
    
    # Test 1: Gemini API
    test1_result = test_gemini_direct()
    
    # Test 2: RAG System
    test2_result = test_rag_system()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 KẾT QUẢ TEST")
    print("=" * 60)
    print(f"Gemini API: {'✅ PASS' if test1_result else '❌ FAIL'}")
    print(f"RAG System: {'✅ PASS' if test2_result else '❌ FAIL'}")
    
    if test1_result and test2_result:
        print("\n✅ Tất cả tests đều PASS!")
    else:
        print("\n⚠️  Một số tests FAIL. Vui lòng kiểm tra lại.")

