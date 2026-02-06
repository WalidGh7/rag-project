from pathlib import Path
import pymupdf4llm
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_core.documents import Document

from rag.vectorstore import get_embeddings, get_vectorstore


def load_and_split_pdf(pdf_path: str) -> list[Document]:
    """
    Hybrid chunking strategy:
      1. Convert PDF → Markdown (preserves headings, structure)
      2. Split by headers (chapters / sections / subsections)
      3. Sub-split long sections with RecursiveCharacterTextSplitter
      4. Enrich metadata with chapter, section, subsection, page
    """
    # Step 1: PDF → Markdown with page tags
    md_text = pymupdf4llm.to_markdown(pdf_path, page_chunks=False)

    # Step 2: Split by Markdown headers
    headers_to_split_on = [
        ("#", "chapter"),
        ("##", "section"),
        ("###", "subsection"),
    ]
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )
    header_chunks = header_splitter.split_text(md_text)

    # Step 3: Sub-split long sections (1500 chars ≈ 375 tokens, good for text-embedding-3-small)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
    )
    final_chunks = text_splitter.split_documents(header_chunks)

    # Step 4: Enrich metadata
    source_name = Path(pdf_path).name
    for chunk in final_chunks:
        chunk.metadata["source"] = source_name
        # Extract page number from pymupdf4llm page markers if present
        page = _extract_page_number(chunk.page_content)
        if page is not None:
            chunk.metadata["page"] = page

    return final_chunks


def _extract_page_number(text: str) -> int | None:
    """
    pymupdf4llm inserts page break markers like '-----\n\n' or page references.
    Try to find the last page number reference in the chunk.
    Falls back to None if not found.
    """
    import re
    # pymupdf4llm may include page markers; look for patterns
    matches = re.findall(r"<!--\s*page[:\s]+(\d+)\s*-->", text, re.IGNORECASE)
    if matches:
        return int(matches[-1])
    # Fallback: no page marker found
    return None


def ingest_pdf(pdf_path: str) -> dict:
    embeddings = get_embeddings()
    vs = get_vectorstore(embeddings)

    chunks = load_and_split_pdf(pdf_path)

    vs.add_documents(chunks)
    return {
        "chunks_added": len(chunks),
        "strategy": "hybrid_structural_recursive",
        "chunk_size": 1500,
        "chunk_overlap": 200,
    }
