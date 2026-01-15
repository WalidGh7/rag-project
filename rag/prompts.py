from langchain_core.prompts import ChatPromptTemplate

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant. Answer ONLY using the provided context. "
     "If the answer is not in the context, say you don't know. "
     "Cite sources as [source:page]."),
    ("human",
     "Question: {question}\n\nContext:\n{context}")
])
