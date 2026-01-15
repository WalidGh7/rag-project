from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.vectorstore import get_embeddings, get_vectorstore

def load_and_split_pdf(pdf_path: str):
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()  # each doc has metadata incl. page number
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    # Normalize metadata for nicer citations
    for c in chunks:
        c.metadata["source"] = Path(pdf_path).name
        c.metadata["page"] = c.metadata.get("page", None)

    return chunks

def ingest_pdf(pdf_path: str) -> dict:
    embeddings = get_embeddings()
    vs = get_vectorstore(embeddings)

    chunks = load_and_split_pdf(pdf_path)

    # Add/update
    vs.add_documents(chunks)
    # persistence is controlled by persist_directory. :contentReference[oaicite:8]{index=8}
    return {"chunks_added": len(chunks)}
