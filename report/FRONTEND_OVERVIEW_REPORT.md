# BÁO CÁO TỔNG QUAN FRONTEND
## Legal Chatbot - Luật Đấu thầu Việt Nam

---

## 📋 MỤC LỤC

1. [Tổng Quan](#1-tổng-quan)
2. [Công Nghệ và Stack](#2-công-nghệ-và-stack)
3. [Kiến Trúc Ứng Dụng](#3-kiến-trúc-ứng-dụng)
4. [Các Component Chính](#4-các-component-chính)
5. [Tính Năng](#5-tính-năng)
6. [Tích Hợp Backend](#6-tích-hợp-backend)
7. [Kết Luận](#7-kết-luận)

---

## 1. TỔNG QUAN

### 1.1. Mục Đích

Frontend được xây dựng để cung cấp giao diện người dùng cho hệ thống chatbot tư vấn Luật Đấu thầu Việt Nam, tích hợp với backend RAG system.

### 1.2. Đặc Điểm

- ✅ **Modern UI**: Dark theme, responsive design
- ✅ **Real-time Streaming**: Hiển thị response từng phần
- ✅ **Conversation History**: Lưu và quản lý lịch sử chat
- ✅ **Markdown Support**: Render markdown với formatting đẹp
- ✅ **Mobile Responsive**: Hoạt động tốt trên mobile và desktop

### 1.3. Tech Stack

- **Framework**: React 19.2.0
- **Language**: TypeScript 5.8.2
- **Build Tool**: Vite 6.2.0
- **UI Libraries**: 
  - `lucide-react`: Icons
  - `react-markdown`: Markdown rendering
  - `remark-gfm`: GitHub Flavored Markdown support

---

## 2. CÔNG NGHỆ VÀ STACK

### 2.1. Core Technologies

#### 2.1.1. React 19.2.0
- **Mục đích**: UI framework
- **Features**: 
  - Hooks (useState, useEffect, useCallback, useRef)
  - Component-based architecture
  - State management

#### 2.1.2. TypeScript 5.8.2
- **Mục đích**: Type safety và developer experience
- **Benefits**:
  - Type checking tại compile time
  - IntelliSense và autocomplete
  - Refactoring an toàn

#### 2.1.3. Vite 6.2.0
- **Mục đích**: Build tool và dev server
- **Features**:
  - Fast HMR (Hot Module Replacement)
  - Optimized builds
  - ES modules support

### 2.2. UI Libraries

#### 2.2.1. react-markdown 9.1.0
- **Mục đích**: Render markdown content từ AI response
- **Plugins**:
  - `remark-gfm`: GitHub Flavored Markdown (tables, strikethrough, etc.)
  - `remark-breaks`: Line breaks support

#### 2.2.2. lucide-react 0.555.0
- **Mục đích**: Icon library
- **Icons sử dụng**: 
  - `PanelLeft`: Sidebar toggle
  - `SquarePen`: New chat
  - `ArrowUp`: Send button
  - `Copy`, `ThumbsUp`, `ThumbsDown`: Message actions
  - `User`: User avatar

### 2.3. Styling

- **Approach**: Tailwind CSS (inline classes)
- **Theme**: Dark theme (`bg-[#212121]`, `bg-[#171717]`)
- **Responsive**: Mobile-first với breakpoints (`md:`)

---

## 3. KIẾN TRÚC ỨNG DỤNG

### 3.1. Component Structure

```
App.tsx (Root)
├─ Sidebar
│  ├─ New Chat Button
│  └─ Chat History List
├─ ChatArea
│  ├─ WelcomeScreen (khi chưa có messages)
│  └─ Message List
│     ├─ User Messages
│     └─ Model Messages (với Markdown)
│        └─ Action Buttons (Copy, Like, Dislike)
└─ InputBar
   ├─ Textarea
   └─ Send Button
```

### 3.2. State Management

**Local State (React Hooks)**:
- `sessions`: Danh sách chat sessions
- `currentSessionId`: Session hiện tại
- `loadingState`: Trạng thái loading ('idle', 'streaming', 'error')
- `isSidebarOpen`: Trạng thái sidebar (mobile)

**Persistence**:
- `localStorage`: Lưu sessions để persist giữa các lần reload

### 3.3. Data Flow

```
User Input
    ↓
InputBar → handleSend()
    ↓
App.tsx → streamChatResponse()
    ↓
apiService.ts → POST /api/chat
    ↓
Backend API → RAG System
    ↓
Streaming Response
    ↓
onChunk callback → Update UI
    ↓
ChatArea → Render Markdown
```

---

## 4. CÁC COMPONENT CHÍNH

### 4.1. App.tsx (Root Component)

**Trách nhiệm**:
- Quản lý sessions và state
- Xử lý send message
- Tích hợp với API service

**Key Features**:
- **Session Management**: Create, select, update sessions
- **Title Generation**: Tự động generate title từ message đầu tiên
- **Streaming**: Xử lý streaming response từ backend
- **Error Handling**: Xử lý lỗi và hiển thị thông báo

**State**:
```typescript
sessions: ChatSession[]
currentSessionId: string | null
loadingState: 'idle' | 'streaming' | 'error'
isSidebarOpen: boolean
```

### 4.2. Sidebar.tsx

**Trách nhiệm**:
- Hiển thị danh sách chat sessions
- Tạo chat mới
- Chọn session hiện tại

**Features**:
- **Chat History**: Hiển thị các sessions gần đây
- **New Chat Button**: Tạo session mới
- **Mobile Responsive**: Overlay trên mobile, sidebar trên desktop
- **Active State**: Highlight session đang active

**Props**:
```typescript
sessions: ChatSession[]
currentSessionId: string | null
onSelectSession: (id: string) => void
onNewChat: () => void
isOpen: boolean
toggleSidebar: () => void
```

### 4.3. ChatArea.tsx

**Trách nhiệm**:
- Hiển thị messages
- Render markdown với formatting
- Xử lý user interactions (copy, like, dislike)

**Features**:
- **Welcome Screen**: Hiển thị khi chưa có messages
- **Message Rendering**: 
  - User messages: Plain text
  - Model messages: Markdown với formatting
- **Markdown Components**:
  - Paragraphs với spacing
  - Headings (h2, h3)
  - Lists (ul, ol, li)
  - Code blocks (inline và block)
  - Bold, italic
  - Line breaks
- **Action Buttons**:
  - Copy to clipboard
  - Like/Dislike feedback
- **Loading Indicator**: Hiển thị khi đang streaming

**Props**:
```typescript
messages: Message[]
loadingState: LoadingState
onSend: (text: string) => void
```

### 4.4. InputBar.tsx

**Trách nhiệm**:
- Nhận input từ user
- Gửi message đến backend

**Features**:
- **Auto-resize Textarea**: Tự động điều chỉnh chiều cao
- **Keyboard Shortcuts**: Enter để send, Shift+Enter để xuống dòng
- **Send Button**: Disabled khi input rỗng hoặc đang loading
- **Placeholder**: "Hỏi bất kỳ điều gì về đấu thầu..."
- **Disclaimer**: "Luật Đấu Thầu AI có thể mắc lỗi..."

**Props**:
```typescript
onSend: (text: string) => void
isLoading: boolean
```

### 4.5. WelcomeScreen.tsx

**Trách nhiệm**:
- Hiển thị màn hình chào mừng
- Gợi ý các câu hỏi mẫu

**Features**:
- **Welcome Message**: Giới thiệu về chatbot
- **Sample Questions**: Các câu hỏi mẫu để user click
- **Visual Design**: Centered, clean layout

### 4.6. apiService.ts

**Trách nhiệm**:
- Kết nối với backend API
- Xử lý streaming responses
- Error handling

**Functions**:
- `streamChatResponse()`: Gửi message và nhận streaming response
- `chatResponse()`: Non-streaming fallback
- `searchDocuments()`: Semantic search (chưa sử dụng)
- `checkApiStatus()`: Health check
- `generateTitle()`: Generate title cho chat (simple implementation)

**API Endpoints**:
- `POST /api/chat`: Chat endpoint với streaming
- `POST /api/search`: Semantic search (optional)
- `GET /health`: Health check

---

## 5. TÍNH NĂNG

### 5.1. Core Features

#### 5.1.1. Chat Interface
- ✅ **Real-time Streaming**: Hiển thị response từng phần
- ✅ **Markdown Rendering**: Format đẹp với headings, lists, code
- ✅ **Message History**: Lưu và hiển thị lịch sử chat
- ✅ **Session Management**: Nhiều sessions, switch giữa các sessions

#### 5.1.2. User Interactions
- ✅ **Copy to Clipboard**: Copy message content
- ✅ **Like/Dislike**: Feedback cho responses
- ✅ **Auto-scroll**: Tự động scroll đến message mới nhất
- ✅ **Keyboard Shortcuts**: Enter để send, Shift+Enter để xuống dòng

#### 5.1.3. UI/UX
- ✅ **Dark Theme**: Modern dark theme
- ✅ **Responsive Design**: Mobile và desktop
- ✅ **Loading States**: Visual feedback khi đang xử lý
- ✅ **Error Handling**: Thông báo lỗi rõ ràng

### 5.2. Advanced Features

#### 5.2.1. Streaming Response
- **Implementation**: Server-Sent Events (SSE)
- **Format**: `data: {chunk: "..."}\n\n`
- **Accumulation**: Accumulate chunks để hiển thị full response
- **Metadata**: Nhận sources và confidence score ở cuối

#### 5.2.2. Markdown Rendering
- **Custom Components**: 
  - Paragraphs với spacing
  - Headings với font size phù hợp
  - Lists với proper indentation
  - Code blocks với syntax highlighting (có thể mở rộng)
- **Preserve Formatting**: Giữ nguyên xuống dòng và spacing

#### 5.2.3. Session Persistence
- **Storage**: localStorage
- **Key**: `bid_law_chats`
- **Format**: JSON array of ChatSession objects
- **Auto-save**: Tự động lưu sau mỗi thay đổi

---

## 6. TÍCH HỢP BACKEND

### 6.1. API Integration

#### 6.1.1. Chat Endpoint

**Request**:
```typescript
POST /api/chat
{
  "message": "đấu thầu là gì",
  "history": [
    {"role": "user", "content": "..."},
    {"role": "model", "content": "..."}
  ],
  "stream": true
}
```

**Response** (Streaming):
```
data: {"chunk": "..."}
data: {"chunk": "..."}
data: {"sources": [...], "confidence": 0.85}
data: [DONE]
```

#### 6.1.2. Error Handling

- **Network Errors**: Hiển thị thông báo lỗi
- **API Errors**: Parse error message từ response
- **Timeout**: Có thể thêm timeout handling

#### 6.1.3. Configuration

- **API URL**: 
  - Development: Relative path (Vite proxy)
  - Production: `VITE_API_URL` env variable hoặc default `http://localhost:5000`

### 6.2. Data Flow

```
User types message
    ↓
InputBar → handleSend()
    ↓
App.tsx → streamChatResponse(history, message, onChunk)
    ↓
apiService.ts → fetch('/api/chat', {stream: true})
    ↓
Backend → RAG System → Gemini API
    ↓
Streaming Response → onChunk callback
    ↓
App.tsx → Update sessions state
    ↓
ChatArea → Re-render với new message
```

---

## 7. KẾT LUẬN

### 7.1. Tổng Kết

Frontend đã được xây dựng tốt với:

- ✅ **Modern Stack**: React 19, TypeScript, Vite
- ✅ **Good UX**: Dark theme, responsive, intuitive
- ✅ **Feature-rich**: Streaming, markdown, session management
- ✅ **Well-integrated**: Tích hợp tốt với backend RAG system

### 7.2. Điểm Mạnh

1. **User Experience**: 
   - Dark theme hiện đại
   - Responsive trên mọi thiết bị
   - Streaming response mượt mà
   - Markdown rendering đẹp

2. **Code Quality**:
   - TypeScript cho type safety
   - Component-based architecture
   - Clean separation of concerns

3. **Features**:
   - Session management
   - Conversation history
   - User feedback (like/dislike)
   - Copy to clipboard

### 7.3. Hướng Phát Triển

1. **Citation Display**: Hiển thị sources và citations trong UI
2. **Confidence Indicator**: Hiển thị confidence score
3. **Export Chat**: Export conversation thành PDF/Markdown
4. **Search**: Tích hợp semantic search trong UI
5. **Settings**: User settings (theme, language, etc.)
6. **Analytics**: Track user interactions và feedback

### 7.4. Khuyến Nghị

- **Cho Production**:
  - Thêm error boundaries
  - Thêm loading skeletons
  - Optimize bundle size
  - Add PWA support
  
- **Cho Development**:
  - Thêm unit tests
  - Thêm E2E tests
  - Improve accessibility
  - Add i18n support

---

**Báo cáo được tạo bởi**: AI Assistant  
**Ngày**: 2025  
**Phiên bản**: 1.0

