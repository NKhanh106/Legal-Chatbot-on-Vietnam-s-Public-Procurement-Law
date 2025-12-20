"""
Backend API Server cho Legal Chatbot.
Tích hợp RAG system với React frontend.
"""
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import os
import sys
import json
import time
from typing import Generator

# Add backend/src to path
BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(BACKEND_ROOT)
sys.path.insert(0, BACKEND_ROOT)
sys.path.insert(0, os.path.join(BACKEND_ROOT, "src"))

# Import RAG system
try:
    from src import query
except ImportError:
    # Fallback import
    import query

app = Flask(__name__)
# Enable CORS for React frontend với cấu hình chi tiết
CORS(app, 
     origins=["http://localhost:5173", "http://127.0.0.1:5173"],
     allow_headers=["Content-Type", "Authorization"],
     methods=["GET", "POST", "OPTIONS"],
     supports_credentials=True)

# Configuration
API_PORT = int(os.getenv("API_PORT", 5000))
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "Legal Chatbot API",
        "rag_enabled": True
    })

@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Chat endpoint với RAG.
    
    Request body:
    {
        "message": "câu hỏi của người dùng",
        "stream": false  # optional, default false
    }
    
    Response:
    {
        "response": "câu trả lời từ RAG system",
        "sources": ["chunk1", "chunk2", ...]  # optional, nếu cần
    }
    """
    try:
        data = request.get_json()
        user_message = data.get("message", "").strip()
        conversation_history = data.get("history", [])  # Lấy conversation history
        stream = data.get("stream", False)
        
        if not user_message:
            return jsonify({"error": "Message is required"}), 400
        
        # Sử dụng RAG system để tạo câu trả lời
        if stream:
            # Streaming response
            def generate():
                # Gọi RAG system với metadata và conversation history
                # TỐI ƯU: Không cần tìm kiếm context riêng vì ask_sth đã tự tìm kiếm
                result = query.ask_sth(
                    user_message, 
                    return_metadata=True, 
                    use_advanced=True,
                    conversation_history=conversation_history
                )
                answer = result.get("answer", "")
                
                # Simulate streaming bằng cách chia nhỏ response
                # Chia thành các chunk nhỏ để frontend có thể hiển thị từng phần
                # QUAN TRỌNG: Giữ nguyên xuống dòng (\n) và markdown formatting
                import re
                # Chia text thành các chunk, giữ nguyên \n và markdown
                # Chia theo từ nhưng giữ nguyên whitespace (bao gồm \n)
                words = re.split(r'(\s+)', answer)  # Split nhưng giữ lại whitespace
                chunk_size = 10  # Số từ mỗi chunk
                for i in range(0, len(words), chunk_size):
                    chunk_parts = words[i:i+chunk_size]
                    chunk = ''.join(chunk_parts)  # Join lại, giữ nguyên whitespace
                    if chunk:  # Chỉ gửi nếu chunk không rỗng
                        yield f"data: {json.dumps({'chunk': chunk})}\n\n"
                        time.sleep(0.03)  # Small delay for streaming effect
                
                # Gửi metadata cuối cùng (sources và confidence)
                yield f"data: {json.dumps({'sources': result.get('sources', []), 'confidence': result.get('confidence', 0.0)})}\n\n"
                yield "data: [DONE]\n\n"
            
            return Response(
                stream_with_context(generate()),
                mimetype="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                }
            )
        else:
            # Non-streaming response với metadata đầy đủ
            result = query.ask_sth(
                user_message, 
                return_metadata=True, 
                use_advanced=True,
                conversation_history=conversation_history
            )
            
            return jsonify({
                "response": result.get("answer", ""),
                "sources": result.get("sources", []),
                "confidence": result.get("confidence", 0.0),
                "query_type": result.get("query_type", {}),
                "context_count": result.get("context_count", 0)
            })
    
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": "Đã có lỗi xảy ra khi xử lý câu hỏi",
            "details": str(e) if DEBUG else None
        }), 500

@app.route("/api/search", methods=["POST"])
def search():
    """
    Semantic search endpoint.
    
    Request body:
    {
        "query": "từ khóa tìm kiếm",
        "top_k": 3  # optional, default 3
    }
    
    Response:
    {
        "results": ["chunk1", "chunk2", ...],
        "count": 3
    }
    """
    try:
        data = request.get_json()
        search_query = data.get("query", "").strip()
        top_k = data.get("top_k", 3)
        
        if not search_query:
            return jsonify({"error": "Query is required"}), 400
        
        results = query.search_faiss(search_query, top_k=top_k)
        
        return jsonify({
            "results": results,
            "count": len(results)
        })
    
    except Exception as e:
        print(f"Error in search endpoint: {str(e)}")
        return jsonify({
            "error": "Đã có lỗi xảy ra khi tìm kiếm",
            "details": str(e) if DEBUG else None
        }), 500

@app.route("/api/status", methods=["GET"])
def status():
    """Get system status."""
    try:
        # Kiểm tra xem RAG system có sẵn sàng không
        test_query = "test"
        _ = query.search_faiss(test_query, top_k=1)
        
        return jsonify({
            "status": "ready",
            "rag_system": "enabled",
            "index_loaded": True
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "rag_system": "disabled",
            "error": str(e) if DEBUG else "RAG system not available"
        }), 500

if __name__ == "__main__":
    print("🚀 Starting Legal Chatbot API Server...")
    print(f"📡 Server will run on http://localhost:{API_PORT}")
    print(f"🔗 RAG System: Enabled")
    print(f"📚 Using FAISS index from: {os.path.join(PROJECT_ROOT, 'data')}")
    print("-" * 60)
    
    app.run(
        host="0.0.0.0",
        port=API_PORT,
        debug=DEBUG,
        threaded=True
    )

