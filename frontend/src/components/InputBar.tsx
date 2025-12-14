import React, { useState, useRef, useEffect } from 'react';
import { ArrowUp, Paperclip, Mic, Globe } from 'lucide-react';

interface InputBarProps {
  onSend: (text: string) => void;
  isLoading: boolean;
}

export const InputBar: React.FC<InputBarProps> = ({ onSend, isLoading }) => {
  const [input, setInput] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleSend = () => {
    if (input.trim() && !isLoading) {
      onSend(input);
      setInput('');
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto';
      }
    }
  };

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`;
    }
  }, [input]);

  return (
    <div className="w-full max-w-[1536px] mx-auto px-4 pb-4 md:pb-6 relative">
      <div className="relative flex items-end bg-[#2f2f2f] rounded-3xl px-4 py-3 shadow-lg w-2/3 mx-auto">
        <button className="p-2 text-gray-400 hover:text-white transition-colors rounded-full hover:bg-white/10 mr-2 flex-shrink-0">
          <Paperclip className="w-5 h-5" />
        </button>
        
        <textarea
          ref={textareaRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Hỏi bất kỳ điều gì về đấu thầu..."
          className="flex-1 bg-transparent border-none text-white placeholder-gray-400 resize-none outline-none max-h-[200px] py-2"
          rows={1}
          disabled={isLoading}
        />

        <div className="flex items-center gap-2 ml-2 flex-shrink-0 pb-1">
            {/* Mock Mic button - visuals only */}
            {!input && (
                <button className="p-2 bg-transparent text-gray-400 rounded-full hover:bg-white/10 transition-colors">
                    <Mic className="w-5 h-5" />
                </button>
            )}
            
            <button
                onClick={handleSend}
                disabled={!input.trim() || isLoading}
                className={`p-2 rounded-full transition-colors ${
                input.trim() 
                    ? 'bg-white text-black hover:bg-gray-200' 
                    : 'bg-[#676767] text-gray-900 cursor-not-allowed opacity-50'
                }`}
            >
                <ArrowUp className="w-5 h-5" />
            </button>
        </div>
      </div>
      <div className="text-center mt-2 text-xs text-gray-500">
        Luật Đấu Thầu AI có thể mắc lỗi. Hãy kiểm tra thông tin quan trọng.
      </div>
    </div>
  );
};

