from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader

from app.config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    PDF_DIR,
)


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def _get_collection():
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )


def _load_pdf_chunks(pdf_path: Path) -> list[dict]:
    reader = PdfReader(str(pdf_path))
    records: list[dict] = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        for chunk_index, chunk in enumerate(
            _chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        ):
            records.append(
                {
                    "text": chunk,
                    "metadata": {
                        "source": pdf_path.name,
                        "page": page_num,
                        "chunk": chunk_index,
                    },
                }
            )
    return records


def _safe_filename(filename: str) -> str:
    name = Path(filename).name
    if not name.lower().endswith(".pdf"):
        name = f"{name}.pdf"
    return name


def list_pdfs() -> list[Path]:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(PDF_DIR.glob("*.pdf"))


def list_document_names() -> list[str]:
    return [path.name for path in list_pdfs()]


def save_pdf(filename: str, content: bytes) -> Path:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = _safe_filename(filename)
    pdf_path = PDF_DIR / safe_name
    pdf_path.write_bytes(content)
    return pdf_path


def _add_chunks_to_collection(collection, records: list[dict]) -> int:
    if not records:
        return 0

    all_ids: list[str] = []
    all_docs: list[str] = []
    all_meta: list[dict] = []

    for record in records:
        meta = record["metadata"]
        doc_id = f"{meta['source']}_p{meta['page']}_c{meta['chunk']}"
        all_ids.append(doc_id)
        all_docs.append(record["text"])
        all_meta.append(meta)

    batch_size = 100
    for i in range(0, len(all_docs), batch_size):
        collection.add(
            ids=all_ids[i : i + batch_size],
            documents=all_docs[i : i + batch_size],
            metadatas=all_meta[i : i + batch_size],
        )

    return len(all_docs)


def ingest_pdf(pdf_path: Path) -> int:
    """Index a single PDF, replacing any existing chunks from the same file."""
    collection = _get_collection()
    source_name = pdf_path.name

    try:
        collection.delete(where={"source": source_name})
    except Exception:
        pass

    records = _load_pdf_chunks(pdf_path)
    return _add_chunks_to_collection(collection, records)


def ingest_pdfs() -> tuple[int, int]:
    """Re-index every PDF in the uploads folder from scratch."""
    pdf_files = list_pdfs()
    if not pdf_files:
        return 0, 0

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = _get_collection()

    total_chunks = 0
    for pdf_path in pdf_files:
        total_chunks += _add_chunks_to_collection(
            collection, _load_pdf_chunks(pdf_path)
        )

    return len(pdf_files), total_chunks


def ingest_upload(filename: str, content: bytes) -> tuple[str, int]:
    """Save an uploaded PDF and index it immediately."""
    if not content:
        raise ValueError(f"'{filename}' is empty.")

    pdf_path = save_pdf(filename, content)
    chunks = ingest_pdf(pdf_path)
    if chunks == 0:
        raise ValueError(
            f"Could not extract text from '{pdf_path.name}'. "
            "The PDF may be scanned images without selectable text."
        )
    return pdf_path.name, chunks


def get_indexed_chunk_count() -> int:
    try:
        collection = _get_collection()
        return collection.count()
    except Exception:
        return 0
