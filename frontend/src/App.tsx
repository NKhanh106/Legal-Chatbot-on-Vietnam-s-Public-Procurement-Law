import React, { useState, useEffect, useCallback } from 'react';
import { Sidebar } from './components/Sidebar';
import { ChatArea } from './components/ChatArea';
import { InputBar } from './components/InputBar';
import { ChatSession, Message, LoadingState } from './types';
// Sử dụng API service với RAG thay vì gọi Gemini trực tiếp
import { streamChatResponse, generateTitle } from './services/apiService';
import { PanelLeft, SquarePen } from 'lucide-react';

// Helper for unique ID since we can't import uuid package in this strict env without adding it to package.json which isn't available here
const generateId = () => Date.now().toString(36) + Math.random().toString(36).substr(2);

const App: React.FC = () => {
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [loadingState, setLoadingState] = useState<LoadingState>('idle');
  const [isSidebarOpen, setSidebarOpen] = useState(true);

  // Load from local storage on mount (optional mock)
  useEffect(() => {
    const saved = localStorage.getItem('bid_law_chats');
    if (saved) {
      setSessions(JSON.parse(saved));
    }
  }, []);

  // Save to local storage
  useEffect(() => {
    localStorage.setItem('bid_law_chats', JSON.stringify(sessions));
  }, [sessions]);

  const getCurrentSession = () => sessions.find(s => s.id === currentSessionId);

  const createNewChat = useCallback(() => {
    const newSession: ChatSession = {
      id: generateId(),
      title: 'Đoạn chat mới',
      messages: [],
      createdAt: Date.now(),
    };
    setSessions(prev => [...prev, newSession]);
    setCurrentSessionId(newSession.id);
    return newSession.id;
  }, []);

  const updateSessionMessages = (sessionId: string, newMessages: Message[]) => {
    setSessions(prev => prev.map(s => {
      if (s.id === sessionId) {
        return { ...s, messages: newMessages };
      }
      return s;
    }));
  };

  const updateSessionTitle = (sessionId: string, title: string) => {
    setSessions(prev => prev.map(s => {
        if (s.id === sessionId) return { ...s, title };
        return s;
    }));
  };

  const handleSend = async (text: string) => {
    let activeSessionId = currentSessionId;
    let isNewChat = false;

    if (!activeSessionId) {
      activeSessionId = createNewChat();
      isNewChat = true;
    }

    // Double check existence (edge case)
    const currentSession = sessions.find(s => s.id === activeSessionId) || { messages: [] };
    
    // Add User Message
    const userMsg: Message = {
      id: generateId(),
      role: 'user',
      content: text,
      timestamp: Date.now(),
    };

    const updatedMessages = [...currentSession.messages, userMsg];
    updateSessionMessages(activeSessionId, updatedMessages);
    
    // Optimistic Title Generation for new chats
    if (isNewChat || currentSession.messages.length === 0) {
        generateTitle(text).then(title => {
            if(activeSessionId) updateSessionTitle(activeSessionId, title);
        });
    }

    setLoadingState('streaming');

    // Create placeholder for model message
    const modelMsgId = generateId();
    const modelMsgPlaceholder: Message = {
      id: modelMsgId,
      role: 'model',
      content: '', // Will start empty
      timestamp: Date.now() + 1,
    };
    
    // We don't add the empty message to state immediately to avoid an empty bubble flash, 
    // or we can add it and rely on streaming to fill it. 
    // Let's add it so the UI scrolls down.
    updateSessionMessages(activeSessionId, [...updatedMessages, modelMsgPlaceholder]);

    try {
      await streamChatResponse(updatedMessages, text, (accumulatedText) => {
        setSessions(prev => prev.map(s => {
          if (s.id === activeSessionId) {
            const msgs = s.messages.map(m => 
              m.id === modelMsgId ? { ...m, content: accumulatedText } : m
            );
            return { ...s, messages: msgs };
          }
          return s;
        }));
      });
      setLoadingState('idle');
    } catch (error) {
      console.error(error);
      setLoadingState('error');
      // Add error message to chat
      const errorMsg: Message = {
        id: generateId(),
        role: 'model',
        content: "Xin lỗi, đã có lỗi xảy ra khi kết nối với máy chủ. Vui lòng thử lại sau.",
        timestamp: Date.now(),
      };
      // Remove the stalled placeholder and add error
      // Note: In a real app we'd want to keep partial response if any
       setSessions(prev => prev.map(s => {
          if (s.id === activeSessionId) {
            const msgs = s.messages.filter(m => m.id !== modelMsgId);
            return { ...s, messages: [...msgs, errorMsg] };
          }
          return s;
        }));
    }
  };

  const activeSession = getCurrentSession();

  return (
    <div className="flex h-screen w-full bg-[#212121] text-gray-100 font-sans overflow-hidden">
      {/* Sidebar - Hidden on mobile unless toggled */}
      <div className={`${isSidebarOpen ? 'fixed inset-0 z-50 md:static md:inset-auto md:z-auto' : 'hidden'} md:block`}>
          <div className="absolute inset-0 bg-black/50 md:hidden" onClick={() => setSidebarOpen(false)}></div>
          <Sidebar 
            sessions={sessions}
            currentSessionId={currentSessionId}
            onSelectSession={(id) => {
                setCurrentSessionId(id);
                setSidebarOpen(false); // Close on mobile selection
            }}
            onNewChat={() => {
                createNewChat();
                setSidebarOpen(false);
            }}
            isOpen={isSidebarOpen}
            toggleSidebar={() => setSidebarOpen(!isSidebarOpen)}
          />
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col h-full relative min-w-0">
        
        {/* Mobile Header / Top Bar */}
        <div className="sticky top-0 z-10 flex items-center justify-between p-3 md:p-4 text-gray-300 bg-[#212121]">
            <div className="flex items-center gap-2">
                <button 
                    onClick={() => setSidebarOpen(!isSidebarOpen)} 
                    className="p-2 hover:bg-[#2f2f2f] rounded-lg transition-colors text-gray-400 hover:text-white"
                    title={isSidebarOpen ? "Đóng sidebar" : "Mở sidebar"}
                >
                    <PanelLeft className="w-6 h-6" />
                </button>
                <div className="font-semibold text-lg text-gray-200 cursor-pointer flex items-center gap-1 hover:bg-[#2f2f2f] px-3 py-1 rounded-lg">
                    Luật Đấu Thầu
                    <span className="text-xs bg-yellow-600/20 text-yellow-500 px-1.5 py-0.5 rounded ml-2 border border-yellow-600/30">Beta</span>
                </div>
            </div>
            
            <div className="flex items-center gap-2">
                <button 
                  onClick={createNewChat} 
                  className="md:hidden p-2 hover:bg-[#2f2f2f] rounded-lg text-gray-400"
                >
                    <SquarePen className="w-6 h-6" />
                </button>
            </div>
        </div>

        {/* Chat Area */}
        <ChatArea 
          messages={activeSession?.messages || []}
          loadingState={loadingState}
          onSend={handleSend}
        />

        {/* Input Area */}
        <InputBar 
          onSend={handleSend}
          isLoading={loadingState !== 'idle'}
        />
      </div>
    </div>
  );
};

export default App;

