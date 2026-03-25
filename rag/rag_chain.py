from pydantic_settings import BaseSettings, SettingsConfigDict
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder

from rag.vectorstore import get_vectorstore, get_embeddings
from rag.prompts import RAG_PROMPT


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    OPENAI_API_KEY: str
    CHAT_MODEL: str = "gpt-4o-mini"
    TOP_K: int = 6
    CANDIDATE_K: int = 20

settings = Settings()

# Load the reranker model once at startup (not on every request)
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def _reciprocal_rank_fusion(results_lists: list[list[Document]], k: int = 60) -> list[Document]:
    """
    Merges multiple ranked lists of chunks into one using RRF.
    Each chunk gets a score of 1/(rank + k) from each list.
    Higher total score = more relevant overall.
    k=60 is the standard default that works well in practice.
    """
    scores: dict[str, float] = {}
    docs_map: dict[str, Document] = {}

    for results in results_lists:
        for rank, doc in enumerate(results):
            key = doc.page_content[:200] 
            if key not in scores:
                scores[key] = 0.0
                docs_map[key] = doc
            scores[key] += 1.0 / (rank + k)

    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [docs_map[key] for key in sorted_keys]

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

    # Step 1: Get all stored chunks from ChromaDB for BM25 
    # BM25 needs the actual text of every chunk to build its keyword index.
    raw = vs.get()
    all_docs = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(raw["documents"], raw["metadatas"])
    ]

    # Step 2: Run two retrievers independently 
    # Semantic: finds chunks by meaning (vector similarity)
    semantic_results = vs.similarity_search(question, k=settings.CANDIDATE_K)

    # BM25: finds chunks by exact keyword matching
    bm25_retriever = BM25Retriever.from_documents(all_docs, k=settings.CANDIDATE_K)
    bm25_results = bm25_retriever.invoke(question)

    # Step 3: Merge both result lists using RRF 
    # Each chunk gets a score from its rank in each list.
    # Chunks that rank high in both lists float to the top.
    candidates = _reciprocal_rank_fusion([semantic_results, bm25_results])

    # Step 4: Rerank with a local cross-encoder model 
    # The cross-encoder scores each (question, chunk) pair directly,
    # which is more accurate than vector similarity alone.
    pairs = [(question, doc.page_content) for doc in candidates]
    scores = reranker.predict(pairs)

    # Sort by score descending and keep only TOP_K best chunks
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    top_docs = [doc for _, doc in ranked[: settings.TOP_K]]

    # Step 5: Build context and call the LLM 
    context = "\n\n".join(
        f"[{_format_source(d.metadata)}]\n{d.page_content}"
        for d in top_docs
    )

    llm = ChatOpenAI(
        model=settings.CHAT_MODEL,
        api_key=settings.OPENAI_API_KEY,
        temperature=0.2,
    )

    chain = RAG_PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"question": question, "context": context})

    # Return unique sources
    sources = []
    seen = set()
    for d in top_docs:
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

    # contexts = raw chunk texts, needed by RAGAS for evaluation
    contexts = [doc.page_content for doc in top_docs]

    return answer, sources, contexts