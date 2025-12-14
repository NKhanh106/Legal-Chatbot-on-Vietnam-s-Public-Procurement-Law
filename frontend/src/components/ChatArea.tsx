import React, { useRef, useEffect } from 'react';
import { Message, LoadingState } from '../types';
import { WelcomeScreen } from './WelcomeScreen';
import { User, Copy, ThumbsUp, ThumbsDown } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface ChatAreaProps {
  messages: Message[];
  loadingState: LoadingState;
  onSend: (text: string) => void;
}

export const ChatArea: React.FC<ChatAreaProps> = ({ messages, loadingState, onSend }) => {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loadingState]);

  if (messages.length === 0) {
    return <WelcomeScreen onSuggestionClick={onSend} />;
  }

  return (
    <div className="flex-1 overflow-y-auto scrollbar-thin scrollbar-thumb-gray-700 w-full">
      <div className="max-w-3xl mx-auto px-4 py-6 md:py-10">
        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`flex gap-4 mb-6 md:mb-8 group ${
              msg.role === 'user' ? 'justify-end' : 'justify-start'
            }`}
          >
             {/* Model Avatar */}
            {msg.role === 'model' && (
              <div className="w-8 h-8 rounded-full bg-white flex items-center justify-center flex-shrink-0 self-center border border-white/10">
                 <img src="/logo.png" className="w-5 h-5" alt="Logo" />
              </div>
            )}

            {/* Message Content */}
            <div className={`relative max-w-[85%] md:max-w-[90%] ${msg.role === 'user' ? 'bg-[#2f2f2f] rounded-3xl px-5 py-2.5' : ''}`}>
               {msg.role === 'user' ? (
                   <p className="whitespace-pre-wrap text-white leading-7">{msg.content}</p>
               ) : (
                   <div className="text-gray-100 leading-7">
                        {/* Hiển thị loading indicator nếu message rỗng và đang streaming */}
                        {!msg.content && loadingState === 'streaming' ? (
                            <div className="flex items-center gap-1 h-8">
                                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                            </div>
                        ) : msg.content ? (
                            <>
                                {/* Render markdown với formatting */}
                                <div className="font-light text-[0.95rem] md:text-[1rem] markdown-content">
                                    <ReactMarkdown
                                        remarkPlugins={[remarkGfm]}
                                        components={{
                                            // Paragraph: Giữ nguyên xuống dòng và khoảng cách
                                            p: ({node, ...props}) => <p className="mb-4 last:mb-0 whitespace-pre-wrap" style={{ whiteSpace: 'pre-wrap' }} {...props} />,
                                            // Strong/Bold: Tô đậm với màu trắng
                                            strong: ({node, ...props}) => <strong className="text-white font-semibold" style={{ fontWeight: 600 }} {...props} />,
                                            // Lists
                                            ul: ({node, ...props}) => <ul className="list-disc list-inside mb-4 space-y-2 ml-4" {...props} />,
                                            ol: ({node, ...props}) => <ol className="list-decimal list-inside mb-4 space-y-2 ml-4" {...props} />,
                                            li: ({node, ...props}) => <li className="ml-2" {...props} />,
                                            // Headings
                                            h2: ({node, ...props}) => <h2 className="text-xl font-semibold mt-6 mb-3 text-gray-100 first:mt-0" {...props} />,
                                            h3: ({node, ...props}) => <h3 className="text-lg font-semibold mt-5 mb-2 text-gray-100" {...props} />,
                                            // Code
                                            code: ({node, inline, ...props}: any) => 
                                                inline ? (
                                                    <code className="bg-[#2f2f2f] px-1.5 py-0.5 rounded text-sm text-gray-200 font-mono" {...props} />
                                                ) : (
                                                    <code className="block bg-[#2f2f2f] p-3 rounded text-sm text-gray-200 font-mono my-3 overflow-x-auto" {...props} />
                                                ),
                                            // Blockquote
                                            blockquote: ({node, ...props}) => <blockquote className="border-l-4 border-gray-500 pl-4 italic my-4 text-gray-300" {...props} />,
                                            // Line breaks
                                            br: ({node, ...props}) => <br className="block my-2" {...props} />,
                                        }}
                                    >
                                    {msg.content}
                                    </ReactMarkdown>
                                </div>
                                
                                {/* Action buttons for model response */}
                                <div className="flex items-center gap-2 mt-2 opacity-0 group-hover:opacity-100 transition-opacity">
                                    <button className="p-1 text-gray-500 hover:text-gray-300 rounded"><Copy className="w-4 h-4" /></button>
                                    <button className="p-1 text-gray-500 hover:text-gray-300 rounded"><ThumbsUp className="w-4 h-4" /></button>
                                    <button className="p-1 text-gray-500 hover:text-gray-300 rounded"><ThumbsDown className="w-4 h-4" /></button>
                                </div>
                            </>
                        ) : null}
                   </div>
               )}
            </div>

            {/* User Avatar (Only needed if we want to show it on the right) */}
            {/* The user avatar in standard chat UI usually isn't shown next to the bubble, or is implied. 
                But let's stick to the request image style which doesn't explicitly show avatars for user bubbles usually. */}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>
    </div>
  );
};

