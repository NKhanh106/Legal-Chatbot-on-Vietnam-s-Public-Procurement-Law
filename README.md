# Legal Chatbot on Vietnam's Public Procurement Law

<div align="center">
<img width="1200" height="475" alt="GHBanner" src="https://github.com/user-attachments/assets/0aa67016-6eaf-458a-adb2-6e31a0763ed6" />
</div>

A ChatGPT-styled AI assistant specialized in Vietnamese Bid Law (Luật đấu thầu), powered by RAG (Retrieval-Augmented Generation) and Google Gemini API.

## ✨ Features

- **🔍 RAG System**: Advanced retrieval-augmented generation for accurate legal document search
- **🇻🇳 Vietnamese Language Support**: Optimized for Vietnamese legal documents with semantic search
- **📄 Document Processing**: 
  - PDF OCR and text extraction using pytesseract (optimized for Vietnamese)
  - Word document (.docx/.doc) processing with structure preservation
  - Markdown output with legal document structure (Chương, Điều, Khoản, Điểm)
- **💬 Modern UI**: React-based chat interface with TypeScript and real-time streaming
- **⚡ Streaming Responses**: Real-time response streaming from backend API
- **🎯 Semantic Search**: FAISS-based vector search with BM25 keyword search and cross-encoder re-ranking

## 🏗️ Architecture

### System Flow

```
User Query (Frontend)
    ↓
React Frontend (TypeScript)
    ↓ HTTP POST /api/chat
Flask Backend API
    ↓
RAG System (query.py)
    ├─→ FAISS Vector Search (semantic)
    ├─→ BM25 Keyword Search (exact match)
    └─→ Cross-Encoder Re-ranking
    ↓
Context Retrieval
    ↓
Gemini API (LLM Generation)
    ↓
Streaming Response
    ↓
Frontend Display
```

### Components

1. **Frontend** (`frontend/`): React + TypeScript + Vite
   - Chat interface with streaming support
   - Session management with localStorage
   - Responsive design with Tailwind CSS

2. **Backend API** (`backend/api/`): Flask REST API
   - `/api/chat` - Main chat endpoint with streaming
   - `/api/search` - Semantic search endpoint
   - `/health` - Health check

3. **RAG System** (`backend/src/`):
   - `query.py` - Query processing and RAG pipeline
   - `embedding.py` - FAISS index creation and management (optimized chunking for legal documents)
   - `preprocess.py` - Text preprocessing and chunking
   - `read_pdf.py` - PDF OCR and extraction (optimized for Vietnamese)
   - `read_word.py` - Word document processing (.docx/.doc) with structure preservation
   - `correction.py` - OCR error correction (optional)

## 📁 Project Structure

```
.
├── backend/                  # Python Backend
│   ├── api/
│   │   └── server.py         # Flask API server
│   ├── src/                  # Core RAG modules
│   │   ├── query.py          # RAG query processing
│   │   ├── embedding.py      # FAISS embedding system
│   │   ├── preprocess.py     # Text preprocessing
│   │   ├── read_pdf.py       # PDF OCR processing
│   │   ├── correction.py    # OCR correction (optional)
│   │   └── deploy.py         # Deployment utilities
│   ├── config/
│   │   ├── config.py.example # Config template
│   │   └── config.py         # Actual config (gitignored)
│   ├── tests/
│   │   └── test_model.py     # Test scripts
│   ├── requirements.txt      # Python dependencies
│   └── environment.yml       # Conda environment
│
├── frontend/                 # React Frontend
│   ├── src/
│   │   ├── components/       # React components
│   │   │   ├── ChatArea.tsx
│   │   │   ├── InputBar.tsx
│   │   │   ├── Sidebar.tsx
│   │   │   └── WelcomeScreen.tsx
│   │   ├── services/
│   │   │   └── apiService.ts # API client
│   │   ├── App.tsx           # Main app
│   │   ├── index.tsx         # Entry point
│   │   └── types.ts          # TypeScript types
│   ├── public/
│   │   ├── index.html
│   │   └── logo.png
│   ├── package.json
│   ├── tsconfig.json
│   └── vite.config.ts
│
├── scripts/                  # Utility scripts
│   ├── start_server.bat      # Windows startup
│   └── start_server.sh       # Linux/Mac startup
│
├── documents/                # Source documents
│   ├── *.pdf                # PDF documents
│   ├── *.docx, *.doc        # Word documents
│   └── markdown/            # Processed markdown files
├── data/                     # Generated data (gitignored)
│   ├── *.index              # FAISS indices
│   ├── *_meta.pkl           # Metadata files
│   └── text/                # Preprocessed text files
│
├── README.md                 # This file
├── .gitignore               # Git ignore rules
└── metadata.json            # Extension metadata
```

## 🔧 Prerequisites

- **Node.js** 18+ (for frontend)
- **Python** 3.11 (for backend)
- **Conda** (recommended) or **venv** for Python environment
- **Tesseract OCR** (for PDF processing - optional if only using Word documents)
- **Poppler** (for PDF to image conversion - optional)
- **python-docx** (for Word document processing)
- **Google Gemini API Key** (get from [Google AI Studio](https://makersuite.google.com/app/apikey))

## 📦 Installation

### 1. Clone Repository

```bash
git clone <repository-url>
cd Legal-Chatbot-on-Vietnam-s-Public-Procurement-Law
```

### 2. Backend Setup

#### Option A: Using Conda (Recommended)

```bash
# Create environment from file
conda env create -f backend/environment.yml

# Activate environment
conda activate legal-chatbot
```

#### Option B: Using venv

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\Activate.ps1

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt
```

### 3. Frontend Setup

```bash
cd frontend
npm install
```

### 4. Configuration

#### Backend Configuration

1. Copy config template:
```bash
cp backend/config/config.py.example backend/config/config.py
```

2. Edit `backend/config/config.py` and add your Gemini API key:
```python
# Load from .env.local (recommended)
# Or set directly (not recommended for production)
GEMINI_API_KEY = "your_api_key_here"
GEMINI_MODEL_NAME = "gemini-2.5-pro"
```

**OR** create `.env.local` in project root:
```env
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL_NAME=gemini-2.5-pro
```

#### Frontend Configuration

Create `.env.local` in project root (optional):
```env
VITE_API_URL=http://localhost:5000
```

**Note**: `.env.local` and `backend/config/config.py` are in `.gitignore` to keep your API keys secure.

## 🚀 Running the Application

### Start Backend Server

```bash
# Windows
scripts\start_server.bat

# Linux/Mac
bash scripts/start_server.sh

# Or manually
python backend/api/server.py
```

The API server will run on `http://localhost:5000`

### Start Frontend

```bash
cd frontend
npm run dev
```

The frontend will be available at `http://localhost:5173`

## 💻 Development

### Building RAG Index

Before using the chatbot, you need to build the RAG index from your documents:

```bash
# Step 1: Process Word documents (.docx/.doc) → Markdown
python backend/src/read_word.py
# Output: documents/markdown/*.md

# Step 2: (Optional) Extract text from PDFs (OCR)
python backend/src/read_pdf.py
# Output: documents/markdown/*.md (from PDFs)

# Step 3: (Optional) Correct OCR errors
python backend/src/correction.py --all

# Step 4: Preprocess and chunk documents
python backend/src/preprocess.py
# Input: documents/markdown/*.md
# Output: data/text/*.txt

# Step 5: Create embeddings and FAISS index
python backend/src/embedding.py
# Output: data/data_for_rag.index, data/data_for_rag_meta.pkl
```

**Note**: The system now supports both Word documents (recommended for original legal texts) and PDFs (for scanned documents).

The index files will be saved in `data/` directory:
- `data/faiss_index.index` - FAISS vector index
- `data/chunks_meta.pkl` - Metadata (chunks, sources, etc.)

### Running Tests

```bash
python -m pytest backend/tests/
```

## 🔐 Security

### API Key Protection

- ✅ API keys are stored in `backend/config/config.py` (gitignored)
- ✅ Environment variables (`.env.local`) are gitignored
- ✅ No API keys in source code
- ✅ Frontend does NOT directly call Gemini API (all requests go through backend)

### Files Ignored by Git

- `backend/config/config.py` - Contains API keys
- `.env.local` - Environment variables
- `data/*.index`, `data/*_meta.pkl`, `data/*.txt` - Generated files
- `node_modules/`, `venv/`, `__pycache__/` - Dependencies
- `logs/`, `*.log` - Log files

## 📚 How It Works

### RAG Pipeline

1. **Document Processing**:
   - Word/PDF → Markdown (preserving legal structure: Chương, Điều, Khoản, Điểm) → Preprocessing → Chunks
   - Optimized chunking: Respects legal document boundaries, keeps Điều/Khoản/Điểm intact

2. **Embedding**:
   - Chunks → Vietnamese Bi-Encoder → FAISS Index
   - Chunk size: 500 words (optimized for legal documents)
   - Overlap: 50 words (minimal to avoid duplication)

3. **Query Processing**:
   - User Query → Embedding → FAISS Search (semantic) → BM25 Search (keyword) → Cross-Encoder Re-ranking

4. **Response Generation**:
   - Retrieved Context + Query → Gemini API → Streaming Response
   - Post-processing: Fixes markdown formatting, removes unwanted line breaks

### Search Strategy

The system uses a hybrid search approach:

1. **Semantic Search** (FAISS): Finds semantically similar chunks
2. **Keyword Search** (BM25): Finds exact keyword matches
3. **Re-ranking** (Cross-Encoder): Ranks results by relevance

## 🛠️ Technology Stack

### Backend
- **Flask** - Web framework
- **FAISS** - Vector similarity search
- **Sentence Transformers** - Embedding models (Vietnamese Bi-Encoder)
- **Google Gemini API** - LLM for generation
- **pytesseract** - PDF OCR (for scanned documents)
- **python-docx** - Word document processing (for original documents)
- **BM25** - Keyword search
- **Cross-Encoder** - Re-ranking

### Frontend
- **React** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool
- **Tailwind CSS** - Styling

## 📝 License

[Add your license here]

## 🤝 Contributing

[Add contribution guidelines here]

## 📧 Contact

[Add contact information here]
