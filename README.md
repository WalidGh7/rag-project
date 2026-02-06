# Thesis RAG Chat

A RAG (Retrieval-Augmented Generation) app that lets you chat with a PDF thesis. It ingests the document into a vector database and uses semantic search + an LLM to answer questions grounded in the actual content.

## Tech Stack

- **FastAPI** — API server
- **LangChain** — RAG pipeline (LCEL)
- **ChromaDB** — vector store
- **OpenAI** — embeddings (`text-embedding-3-small`) + chat (`gpt-4o-mini`)
- **pymupdf4llm** — PDF to Markdown conversion
- **PostgreSQL** — query logging
- **Docker Compose** — containerized deployment

## How It Works

1. **Ingestion**: PDF is converted to Markdown, split by document headers (chapters/sections), then sub-split into ~1500-char chunks. Each chunk is embedded and stored in ChromaDB with structural metadata.

2. **Query**: User question is embedded, top-k similar chunks are retrieved, and the LLM generates an answer citing specific chapters and pages.

## Project Structure

```
app/
  main.py          # FastAPI entry point
  api.py           # /query, /health, /history endpoints
  schemas.py       # Pydantic models
  static/          # Chat UI (HTML + CSS + JS)
rag/
  ingest.py        # PDF chunking (hybrid structural + recursive)
  vectorstore.py   # ChromaDB setup
  rag_chain.py     # Retrieval + LLM chain
  prompts.py       # Prompt template
db/
  database.py      # SQLAlchemy engine
  models.py        # QueryLog model
```

## Setup

### Docker (recommended)

```bash
# 1. Create .env with your OpenAI key
cp .env.example .env

# 2. Place your thesis PDF
cp your_thesis.pdf thesis/master_thesis.pdf

# 3. Start services
docker compose up --build

# 4. Ingest the PDF (once)
docker compose exec api python -c "from rag.ingest import ingest_pdf; print(ingest_pdf('thesis/master_thesis.pdf'))"

# 5. Open http://localhost:8000
```

### Local

```bash
pip install -r requirements.txt
# Set up PostgreSQL and update DATABASE_URL in .env
uvicorn app.main:app --reload
```

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Chat UI |
| `/query` | POST | `{"question": "..."}` → `{"answer": "...", "sources": [...]}` |
| `/health` | GET | Health check |
| `/history` | GET | Last 20 queries |
