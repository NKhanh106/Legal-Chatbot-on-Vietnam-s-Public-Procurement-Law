/**
 * API Service - Kết nối với backend RAG system
 * Thay thế cho geminiService.ts để sử dụng RAG thay vì gọi Gemini trực tiếp
 */
import { Message } from "../types";

// Get API URL from environment or use default
// Vite exposes env variables via import.meta.env
// Trong development, sử dụng relative path để dùng Vite proxy
// Trong production, sử dụng full URL từ env hoặc default
// @ts-ignore - Vite environment variables
const getApiBaseUrl = () => {
  // @ts-ignore
  if (import.meta.env?.VITE_API_URL) {
    // @ts-ignore
    return import.meta.env.VITE_API_URL;
  }
  // @ts-ignore
  if (import.meta.env?.DEV) {
    // Development mode: sử dụng relative path để dùng Vite proxy
    return "";
  }
  // Production mode: sử dụng default URL
  return "http://localhost:5000";
};

const API_BASE_URL = getApiBaseUrl();

/**
 * Gửi message đến backend API với RAG
 */
export const streamChatResponse = async (
  history: Message[],
  newMessage: string,
  onChunk: (text: string, sources?: any[], confidence?: number) => void
): Promise<string> => {
  try {
    // Format conversation history để gửi đến backend
    // Chỉ gửi các messages trước message hiện tại (không bao gồm newMessage)
    const conversationHistory = history.map(msg => ({
      role: msg.role,
      content: msg.content
    }));
    
    // Gửi request đến backend API
    const response = await fetch(`${API_BASE_URL}/api/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        message: newMessage,
        history: conversationHistory, // Gửi conversation history
        stream: true, // Request streaming response
      }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }

    // Đọc streaming response
    const reader = response.body?.getReader();
    const decoder = new TextDecoder();
    let fullText = "";
    let buffer = "";

    if (!reader) {
      throw new Error("Response body is not readable");
    }

    while (true) {
      const { done, value } = await reader.read();
      
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || ""; // Giữ lại phần chưa hoàn chỉnh

      for (const line of lines) {
        if (line.startsWith("data: ")) {
          const data = line.slice(6);
          
          if (data === "[DONE]") {
            return fullText;
          }

          try {
            const parsed = JSON.parse(data);
            if (parsed.chunk) {
              // Accumulate chunks (not replace)
              fullText += parsed.chunk;
              onChunk(fullText);
            } else if (parsed.sources || parsed.confidence !== undefined) {
              // Send final metadata (sources and confidence)
              onChunk(fullText, parsed.sources, parsed.confidence);
            } else if (parsed.metadata) {
              // Alternative metadata format
              onChunk(fullText, parsed.metadata.sources, parsed.metadata.confidence);
            }
          } catch (e) {
            // Ignore parse errors
          }
        }
      }
    }

    return fullText;
  } catch (error) {
    console.error("Error calling API:", error);
    throw error;
  }
};

/**
 * Non-streaming chat response (fallback)
 */
export const chatResponse = async (
  message: string
): Promise<{ response: string; sources?: string[] }> => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        message: message,
        stream: false,
      }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Error calling API:", error);
    throw error;
  }
};

/**
 * Semantic search trong RAG system
 */
export const searchDocuments = async (
  query: string,
  topK: number = 3
): Promise<string[]> => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/search`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        query: query,
        top_k: topK,
      }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return data.results || [];
  } catch (error) {
    console.error("Error searching documents:", error);
    throw error;
  }
};

/**
 * Kiểm tra trạng thái API server
 */
export const checkApiStatus = async (): Promise<boolean> => {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    return response.ok;
  } catch (error) {
    console.error("API server is not available:", error);
    return false;
  }
};

/**
 * Generate title cho chat session (vẫn dùng Gemini trực tiếp vì đơn giản)
 * Có thể migrate sang backend sau nếu cần
 */
export const generateTitle = async (firstMessage: string): Promise<string> => {
  // Fallback: sử dụng Gemini trực tiếp cho title generation
  // Hoặc có thể gọi backend API nếu có endpoint
  try {
    // Tạm thời dùng logic đơn giản
    const words = firstMessage.split(" ").slice(0, 5);
    return words.join(" ") || "Đoạn chat mới";
  } catch (e) {
    return "Đoạn chat mới";
  }
};

