import React from 'react';
import { Plus, MoreHorizontal } from 'lucide-react';
import { ChatSession } from '../types';

interface SidebarProps {
  sessions: ChatSession[];
  currentSessionId: string | null;
  onSelectSession: (id: string) => void;
  onNewChat: () => void;
  isOpen: boolean;
  toggleSidebar: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  sessions,
  currentSessionId,
  onSelectSession,
  onNewChat,
  isOpen,
  toggleSidebar
}) => {
  return (
    <div 
      className={`${
        isOpen ? 'w-[260px]' : 'w-0'
      } bg-[#171717] flex flex-col transition-all duration-300 ease-in-out overflow-hidden border-r border-white/5 relative z-20 h-full flex-shrink-0`}
    >
      {/* Header / New Chat */}
      <div className="p-3">
        <button
          onClick={onNewChat}
          className="flex items-center gap-3 w-full px-3 py-2 text-sm text-gray-100 rounded-lg hover:bg-[#212121] transition-colors border border-white/10 hover:border-white/10 group"
        >
          <div className="p-1">
             <div className="h-6 w-6 bg-white text-black rounded-full flex items-center justify-center">
                 <img src="/logo.png" className="w-4 h-4" alt="logo" />
             </div>
          </div>
          <span className="flex-1 text-left font-medium">Đoạn chat mới</span>
          <Plus className="w-5 h-5 text-gray-400 opacity-0 group-hover:opacity-100 transition-opacity" />
        </button>
      </div>

      {/* History List */}
      <div className="flex-1 overflow-y-auto px-2 py-2 scrollbar-thin scrollbar-thumb-gray-700">
        <div className="text-xs font-semibold text-gray-500 px-3 py-2">Gần đây</div>
        <div className="space-y-1">
          {sessions.slice().reverse().map((session) => (
            <button
              key={session.id}
              onClick={() => onSelectSession(session.id)}
              className={`flex items-center gap-3 w-full px-3 py-2 text-sm rounded-lg transition-colors group relative ${
                currentSessionId === session.id
                  ? 'bg-[#212121] text-white'
                  : 'text-gray-300 hover:bg-[#212121]'
              }`}
            >
              <span className="truncate flex-1 text-left">{session.title}</span>
              {currentSessionId === session.id && (
                 <MoreHorizontal className="w-4 h-4 text-gray-400" />
              )}
            </button>
          ))}
          {sessions.length === 0 && (
             <div className="px-3 py-2 text-sm text-gray-600 italic">Chưa có lịch sử chat</div>
          )}
        </div>
      </div>
    </div>
  );
};

