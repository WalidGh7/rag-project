from pydantic_settings import BaseSettings, SettingsConfigDict
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from rag.vectorstore import get_vectorstore, get_embeddings
from rag.prompts import RAG_PROMPT

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    OPENAI_API_KEY: str
    CHAT_MODEL: str = "gpt-4o-mini"
    TOP_K: int = 6

settings = Settings()

def _format_source(metadata: dict) -> str:
    """Build a readable citation from chunk metadata."""
    parts = []
    if metadata.get("chapter"):
        parts.append(metadata["chapter"])
    if metadata.get("section"):
        parts.append(metadata["section"])
    if metadata.get("subsection"):
        parts.append(metadata["subsection"])
    label = " > ".join(parts) if parts else metadata.get("source", "unknown")
    page = metadata.get("page")
    if page is not None:
        label += f", p.{page}"
    return label

def build_rag(question: str):
    embeddings = get_embeddings()
    vs = get_vectorstore(embeddings)
    retriever = vs.as_retriever(search_kwargs={"k": settings.TOP_K})

    docs = retriever.invoke(question)

    # Build context with structural citations
    context = "\n\n".join(
        f"[{_format_source(d.metadata)}]\n{d.page_content}"
        for d in docs
    )

    llm = ChatOpenAI(
        model=settings.CHAT_MODEL,
        api_key=settings.OPENAI_API_KEY,
        temperature=0.2,
    )

    chain = RAG_PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"question": question, "context": context})

    # Return unique sources with structural info
    sources = []
    seen = set()
    for d in docs:
        key = (
            d.metadata.get("source"),
            d.metadata.get("page"),
            d.metadata.get("chapter"),
            d.metadata.get("section"),
        )
        if key not in seen:
            seen.add(key)
            sources.append({
                "source": d.metadata.get("source"),
                "page": d.metadata.get("page"),
                "chapter": d.metadata.get("chapter"),
                "section": d.metadata.get("section"),
            })

    return answer, sources
