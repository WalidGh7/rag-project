from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.schemas import QueryRequest, QueryResponse
from rag.rag_chain import build_rag, settings as rag_settings
from db.database import get_db
from db.models import QueryLog

router = APIRouter()

@router.get("/health")
def health():
    return {"status": "ok"}

@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, db: Session = Depends(get_db)):
    answer, sources, _ = build_rag(req.question)

    log = QueryLog(
        question=req.question,
        answer=answer,
        sources={"items": sources},
        model=rag_settings.CHAT_MODEL,
        top_k=rag_settings.TOP_K,
    )
    db.add(log)
    db.commit()
    db.refresh(log)

    return QueryResponse(answer=answer, sources=sources)

@router.get("/history")
def history(limit: int = 20, db: Session = Depends(get_db)):
    rows = (
        db.query(QueryLog)
        .order_by(QueryLog.id.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id": r.id,
            "question": r.question,
            "answer": r.answer,
            "sources": r.sources,
            "model": r.model,
            "top_k": r.top_k,
            "created_at": r.created_at,
        }
        for r in rows
    ]

