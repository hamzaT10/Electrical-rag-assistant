from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import fitz
import torch
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from electrical_rag.core.settings import Settings
from electrical_rag.rag.device_metadata import (
    infer_device_name_from_source,
    infer_device_name_from_text,
)
from electrical_rag.rag.qdrant_store import QdrantVectorStore
from electrical_rag.rag.text_normalization import normalize_ocr_technical_text

logger = logging.getLogger(__name__)


@dataclass
class IngestionStats:
    pdf_files: int
    pages_loaded: int
    pages_ocr: int
    chunks_created: int


def discover_pdfs(data_dir: Path) -> list[Path]:
    return sorted(
        path for path in data_dir.rglob("*") if path.is_file() and path.suffix.lower() == ".pdf"
    )


def _build_ocr_reader(ocr_languages: list[str]):
    try:
        import easyocr
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "OCR fallback requested but OCR dependencies are missing. "
            "Install with: pip install -r requirements-ocr.txt"
        ) from exc

    use_gpu = torch.cuda.is_available()
    return easyocr.Reader(ocr_languages, gpu=use_gpu)


def _read_page_with_ocr(page: fitz.Page, ocr_reader) -> str:
    import numpy as np
    from PIL import Image

    pix = page.get_pixmap(dpi=250)
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    lines = ocr_reader.readtext(np.array(image), detail=0, paragraph=True)
    return "\n".join(lines).strip()


def _should_use_ocr_fallback(text: str) -> bool:
    normalized = " ".join(text.split())

    if not normalized:
        return True

    # Very short extraction is often useless for retrieval.
    if len(normalized) < 40:
        return True

    # If there are almost no letters, the page is probably noisy.
    letter_count = sum(char.isalpha() for char in normalized)
    if letter_count < 20:
        return True

    # Too many replacement/garbage characters is suspicious.
    bad_chars = normalized.count("\ufffd") + normalized.count("�")
    if bad_chars > 0:
        return True

    return False


def extract_pdf_documents(
    pdf_path: Path,
    root_data_dir: Path,
    enable_ocr_fallback: bool,
    ocr_languages: list[str],
) -> tuple[list[Document], int]:
    source_rel = pdf_path.relative_to(root_data_dir).as_posix()
    source_device_name = infer_device_name_from_source(source_rel)
    detected_pdf_device_name = source_device_name
    documents: list[Document] = []
    ocr_pages = 0
    ocr_reader = _build_ocr_reader(ocr_languages) if enable_ocr_fallback else None

    with fitz.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = normalize_ocr_technical_text(page.get_text("text").strip())
            used_ocr = False

            if enable_ocr_fallback and _should_use_ocr_fallback(text):
                ocr_text = _read_page_with_ocr(page, ocr_reader)
                if ocr_text:
                    text = normalize_ocr_technical_text(ocr_text)
                    used_ocr = True
                    ocr_pages += 1

            if not text:
                continue

            page_device_name = infer_device_name_from_text(text)
            if detected_pdf_device_name is None and page_device_name is not None:
                detected_pdf_device_name = page_device_name

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": source_rel,
                        "page": page_number,
                        "ocr": used_ocr,
                        "device_name": detected_pdf_device_name or page_device_name,
                    },
                )
            )


    return documents, ocr_pages


def run_ingestion(
    settings: Settings,
    document_ids_by_source: dict[str, int] | None = None,
) -> IngestionStats:
    if settings.chunk_overlap >= settings.chunk_size:
        raise ValueError("CHUNK_OVERLAP must be strictly smaller than CHUNK_SIZE.")
    if not settings.data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {settings.data_dir}")

    pdf_files = discover_pdfs(settings.data_dir)
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found under: {settings.data_dir}")

    logger.info("Found %s PDF files", len(pdf_files))

    pages: list[Document] = []
    pages_ocr = 0
    for pdf_path in pdf_files:
        logger.info("Loading %s", pdf_path)
        docs, ocr_count = extract_pdf_documents(
            pdf_path=pdf_path,
            root_data_dir=settings.data_dir,
            enable_ocr_fallback=settings.enable_ocr_fallback,
            ocr_languages=settings.ocr_language_list,
        )
        if document_ids_by_source:
            for document in docs:
                source = str(document.metadata.get("source", ""))
                document_id = document_ids_by_source.get(source)
                if document_id is not None:
                    document.metadata["document_id"] = document_id
        pages.extend(docs)
        pages_ocr += ocr_count

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_documents(pages)

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model_name,
        model_kwargs={"device": device},
    )

    settings.vectorstore_dir.mkdir(parents=True, exist_ok=True)
    if settings.vector_backend == "qdrant":
        QdrantVectorStore(settings, embeddings=embeddings).upsert_documents(chunks)
    else:
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(str(settings.vectorstore_dir))

    stats = IngestionStats(
        pdf_files=len(pdf_files),
        pages_loaded=len(pages),
        pages_ocr=pages_ocr,
        chunks_created=len(chunks),
    )

    manifest = asdict(stats)
    manifest.update(
        {
            "data_dir": str(settings.data_dir),
            "vector_backend": settings.vector_backend,
            "vectorstore_dir": str(settings.vectorstore_dir),
            "qdrant_collection": settings.qdrant_collection,
            "embedding_model_name": settings.embedding_model_name,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
        }
    )
    manifest_path = settings.vectorstore_dir / "ingestion_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return stats


def run_document_ingestion(
    settings: Settings,
    pdf_path: Path,
    document_id: int,
) -> IngestionStats:
    if settings.vector_backend != "qdrant":
        raise ValueError("Single-document incremental ingestion requires VECTOR_BACKEND=qdrant.")
    if settings.chunk_overlap >= settings.chunk_size:
        raise ValueError("CHUNK_OVERLAP must be strictly smaller than CHUNK_SIZE.")
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Incremental ingestion only supports PDFs: {pdf_path}")

    logger.info("Loading single document %s", pdf_path)
    pages, pages_ocr = extract_pdf_documents(
        pdf_path=pdf_path,
        root_data_dir=settings.data_dir,
        enable_ocr_fallback=settings.enable_ocr_fallback,
        ocr_languages=settings.ocr_language_list,
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_documents(pages)

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index
        chunk.metadata["document_id"] = document_id

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model_name,
        model_kwargs={"device": device},
    )
    QdrantVectorStore(settings, embeddings=embeddings).upsert_documents(chunks)

    return IngestionStats(
        pdf_files=1,
        pages_loaded=len(pages),
        pages_ocr=pages_ocr,
        chunks_created=len(chunks),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FAISS vectorstore from PDF corpus.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Root folder containing PDFs.")
    parser.add_argument(
        "--vectorstore-dir",
        type=Path,
        default=None,
        help="Output FAISS directory.",
    )
    parser.add_argument(
        "--enable-ocr-fallback",
        action="store_true",
        help="Use OCR when extracted page text is empty.",
    )
    return parser.parse_args()


def main() -> None:
    base = Settings()
    args = parse_args()

    override_data: dict[str, object] = {}
    if args.data_dir is not None:
        override_data["data_dir"] = args.data_dir
    if args.vectorstore_dir is not None:
        override_data["vectorstore_dir"] = args.vectorstore_dir
    if args.enable_ocr_fallback:
        override_data["enable_ocr_fallback"] = True

    settings = base.model_copy(update=override_data)
    logging.basicConfig(level=settings.app_log_level)

    stats = run_ingestion(settings)
    logger.info("Ingestion completed: %s", stats)


if __name__ == "__main__":
    main()
