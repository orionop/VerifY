# VerifY - Title Verification System

VerifY is a comprehensive title verification system designed for the Press Registrar General of India (PRGI) to validate new publication titles against approximately 160,000 existing titles in their database. The system employs a multi-dimensional verification approach to ensure uniqueness and compliance with regulatory guidelines.

## 📋 Introduction

The VerifY system helps prevent duplicate or confusingly similar publication titles by analyzing submissions across multiple dimensions:

- **Lexical Similarity**: Detects spelling variations and calculates string-based match scores
- **Phonetic Similarity**: Identifies sound-alike titles that may have different spellings
- **Semantic Similarity**: Captures meaning-based similarities using advanced language models
- **Content Rule Checking**: Enforces policy guidelines by detecting disallowed words and title combinations
- **Cross-language Detection**: Identifies similar meanings across different Indian languages

## 🚀 Getting Started

### Prerequisites
- Node.js 16+
- Python 3.9+
- Docker and Docker Compose (for full stack deployment)

### Installation

Clone the repository:
```bash
git clone https://github.com/orionop/VerifY.git
cd VerifY
```

### Frontend Setup
```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

### Backend Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
cd backend
pip install -r requirements.txt

# Start development server
uvicorn app.main:app --reload
```

### Full Stack Deployment with Docker
```bash
# Start all services
docker-compose up -d

# Access the application at http://localhost:8080
```

## 📖 System Architecture

The system employs a layered architecture:

1. **Frontend Layer**: Svelte + Vite application with a clean, minimalist UI
2. **API Layer**: FastAPI backend handling verification requests
3. **Verification Engine**: Multi-dimensional similarity analysis pipeline
4. **Storage Layer**: Elasticsearch and FAISS for efficient title indexing and retrieval


