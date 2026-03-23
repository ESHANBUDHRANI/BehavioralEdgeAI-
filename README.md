# Explainable Risk Attribution with Agent-Based Reasoning
## Groq Version — 100% Free, No Payment Required

---

## What this version uses
- **Chat LLM:** Groq API — llama-3.3-70b-versatile (free)
- **Embeddings:** HuggingFace sentence-transformers/all-MiniLM-L6-v2 (free, local)
- **Database:** SQLite (free, local)
- **Vector store:** ChromaDB (free, local)
- **Market data:** yfinance (free)

---

## Step-by-Step Setup

### Step 1 — Get a free Groq API key
1. Go to https://console.groq.com
2. Sign up with Google or email (no credit card needed)
3. Go to https://console.groq.com/keys
4. Click "Create API Key"
5. Copy the key — it starts with "gsk_"

### Step 2 — Add your key to the .env file
Open the .env file and replace the placeholder:
```
GROQ_API_KEY=gsk_your-actual-key-here
```

### Step 3 — Install Python 3.10+
Download from https://python.org

### Step 4 — Install Node.js 18+
Download from https://nodejs.org

### Step 5 — Install Python packages
```bash
pip3 install -r backend/requirements.txt
```

### Step 6 — Initialize the database
```bash
python3 backend/database/init_db.py
```

### Step 7 — Start the backend
```bash
uvicorn backend.main:app --reload --port 8000
```

### Step 8 — Start the frontend (new terminal)
```bash
cd frontend
npm install
npm run dev
```

### Step 9 — Open the app
Go to http://localhost:5173

---

## Files changed from original
- backend/chat/llm_agent.py — ChatOllama → ChatGroq (llama-3.3-70b-versatile)
- backend/chat/graph.py — OllamaEmbeddings → HuggingFaceEmbeddings (local)
- backend/chat/index_builder.py — OllamaEmbeddings → HuggingFaceEmbeddings (local)
- backend/requirements.txt — added groq, langchain-groq, langchain-huggingface, sentence-transformers
- .env — GROQ_API_KEY instead of OPENAI_API_KEY
