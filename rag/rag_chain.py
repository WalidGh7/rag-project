from pydantic_settings import BaseSettings, SettingsConfigDict
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from rag.vectorstore import get_vectorstore, get_embeddings
from rag.prompts import RAG_PROMPT

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    OPENAI_API_KEY: str  # <-- required
    CHAT_MODEL: str = "gpt-4o-mini"
    TOP_K: int = 4

settings = Settings()

def build_rag(question: str):
    embeddings = get_embeddings()
    vs = get_vectorstore(embeddings)
    retriever = vs.as_retriever(search_kwargs={"k": settings.TOP_K})

    docs = retriever.invoke(question)
    context = "\n\n".join(
        f"[{d.metadata.get('source')}:{d.metadata.get('page')}] {d.page_content}"
        for d in docs
    )

    llm = ChatOpenAI(
        model=settings.CHAT_MODEL,
        api_key=settings.OPENAI_API_KEY,  # <-- explicit
    )

    chain = RAG_PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"question": question, "context": context})

    sources = [{"source": d.metadata.get("source"), "page": d.metadata.get("page")} for d in docs]
    return answer, sources


