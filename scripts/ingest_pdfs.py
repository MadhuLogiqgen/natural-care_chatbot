"""Index all PDFs in data/pdfs into Chroma."""

from app.rag.ingest import ingest_pdfs, list_pdfs

if __name__ == "__main__":
    pdfs = list_pdfs()
    if not pdfs:
        print("No PDFs found. Upload PDFs from the Streamlit app or add files to data/pdfs/.")
        raise SystemExit(1)

    files, chunks = ingest_pdfs()
    print(f"Indexed {chunks} chunks from {files} PDF(s).")
