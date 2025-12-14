import React from 'react';
import { Gavel, BookOpen, Scale, FileText } from 'lucide-react';

interface WelcomeScreenProps {
  onSuggestionClick: (text: string) => void;
}

export const WelcomeScreen: React.FC<WelcomeScreenProps> = ({ onSuggestionClick }) => {
  const suggestions = [
    {
      icon: <Gavel className="w-5 h-5 text-orange-400" />,
      text: "Quy trình đấu thầu rộng rãi qua mạng",
      sub: "như thế nào?"
    },
    {
      icon: <FileText className="w-5 h-5 text-blue-400" />,
      text: "Các trường hợp chỉ định thầu",
      sub: "theo Điều 23 Luật Đấu thầu"
    },
    {
      icon: <Scale className="w-5 h-5 text-purple-400" />,
      text: "Phân biệt E-HSMT và E-HSDT",
      sub: "trong đấu thầu qua mạng"
    },
    {
      icon: <BookOpen className="w-5 h-5 text-green-400" />,
      text: "Soạn thảo kế hoạch lựa chọn nhà thầu",
      sub: "cho dự án đầu tư công"
    }
  ];

  return (
    <div className="flex flex-col items-center justify-center h-full max-w-3xl mx-auto px-4">
      <div className="w-12 h-12 bg-white rounded-full flex items-center justify-center mb-6 shadow-lg shadow-white/10">
          <img src="/logo.png" className="w-8 h-8" alt="Logo" />
      </div>
      <h2 className="text-2xl font-semibold text-white mb-8">Hôm nay bạn muốn tìm hiểu gì về Luật Đấu Thầu?</h2>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 w-full">
        {suggestions.map((item, idx) => (
          <button
            key={idx}
            onClick={() => onSuggestionClick(item.text + " " + item.sub)}
            className="flex flex-col items-start p-4 border border-white/10 rounded-xl hover:bg-[#2f2f2f] transition-colors text-left group"
          >
            <div className="mb-2 group-hover:scale-110 transition-transform duration-200">
                {item.icon}
            </div>
            <span className="text-sm font-medium text-gray-200">{item.text}</span>
            <span className="text-sm text-gray-500">{item.sub}</span>
          </button>
        ))}
      </div>
    </div>
  );
};

