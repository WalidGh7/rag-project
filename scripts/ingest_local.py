from rag.ingest import ingest_pdf

if __name__ == "__main__":
    out = ingest_pdf("thesis/master_thesis.pdf")
    print(out)
