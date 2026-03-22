"""
document_loader.py — SmartDoc AI v1.3
══════════════════════════════════════
Target: PDF Loading 2-5s (lần đầu) / <0.1s (cache hit)

Tối ưu:
  1. PyMuPDF (fitz) làm loader chính — nhanh hơn PDFPlumber ~3x
  2. MD5 cache: cùng file → trả về ngay <0.1s
  3. Parallel page extraction với ThreadPoolExecutor
  4. Fallback chain: PyMuPDF → PDFPlumber → PyPDF
"""

import os
import hashlib
import pickle
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

from langchain_core.documents import Document

# ── Cache ──────────────────────────────────────────────────────────────────────
CACHE_DIR = Path(__file__).parent.parent / "data" / "loader_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _md5(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def _cache_get(key: str):
    p = CACHE_DIR / f"{key}.pkl"
    if p.exists():
        try:
            with open(p, "rb") as f:
                return pickle.load(f)
        except Exception:
            p.unlink(missing_ok=True)
    return None


def _cache_set(key: str, data):
    try:
        with open(CACHE_DIR / f"{key}.pkl", "wb") as f:
            pickle.dump(data, f)
    except Exception:
        pass


# ── PDF Loaders ────────────────────────────────────────────────────────────────

def _load_pdf_pymupdf_parallel(tmp_path: str) -> List[Document]:
    """
    PyMuPDF với parallel page extraction.
    Nhanh nhất — trích xuất song song từng page bằng ThreadPoolExecutor.
    """
    import fitz  # PyMuPDF

    def _extract_page(args):
        page_num, page_bytes = args
        doc = fitz.open(stream=page_bytes, filetype="pdf")
        text = doc[0].get_text("text")
        doc.close()
        return page_num, text

    # Mở file một lần để lấy bytes từng page
    main_doc = fitz.open(tmp_path)
    n_pages  = main_doc.page_count

    # Với file nhỏ (<5 trang): dùng serial (overhead thread > benefit)
    if n_pages <= 5:
        docs = []
        for i in range(n_pages):
            text = main_doc[i].get_text("text").strip()
            if text:
                docs.append(Document(
                    page_content=text,
                    metadata={"page": i, "source": tmp_path, "total_pages": n_pages}
                ))
        main_doc.close()
        return docs

    # File lớn: extract từng page thành bytes rồi parallel process
    page_data = []
    for i in range(n_pages):
        tmp = fitz.open()
        tmp.insert_pdf(main_doc, from_page=i, to_page=i)
        page_bytes = tmp.tobytes()
        tmp.close()
        page_data.append((i, page_bytes))
    main_doc.close()

    results = {}
    # Dùng max 4 thread — tránh quá nhiều overhead
    with ThreadPoolExecutor(max_workers=min(4, n_pages)) as ex:
        futures = {ex.submit(_extract_page, pd): pd[0] for pd in page_data}
        for fut in as_completed(futures):
            try:
                pnum, text = fut.result()
                results[pnum] = text
            except Exception:
                pass

    docs = []
    for i in sorted(results.keys()):
        text = results[i].strip()
        if text:
            docs.append(Document(
                page_content=text,
                metadata={"page": i, "source": tmp_path, "total_pages": n_pages}
            ))
    return docs


def load_pdf(file_bytes: bytes, filename: str = "doc.pdf") -> List[Document]:
    """
    Load PDF với MD5 cache + PyMuPDF parallel.
    Lần đầu: 1-3s | Cache hit: <0.1s
    """
    key = f"pdf_{_md5(file_bytes)}"
    hit = _cache_get(key)
    if hit is not None:
        return hit

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    docs: List[Document] = []
    try:
        # Primary: PyMuPDF parallel
        docs = _load_pdf_pymupdf_parallel(tmp_path)
        if not docs:
            raise ValueError("Empty result from PyMuPDF")
    except Exception:
        try:
            # Fallback 1: PDFPlumber (chính xác hơn, chậm hơn)
            from langchain_community.document_loaders import PDFPlumberLoader
            docs = PDFPlumberLoader(tmp_path).load()
        except Exception:
            try:
                # Fallback 2: PyPDF
                from langchain_community.document_loaders import PyPDFLoader
                docs = PyPDFLoader(tmp_path).load()
            except Exception as e:
                raise RuntimeError(f"Cannot load PDF: {e}")
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    _cache_set(key, docs)
    return docs


def load_docx(file_bytes: bytes, filename: str = "doc.docx") -> List[Document]:
    """Load DOCX với MD5 cache. Lần đầu: <2s | Cache: <0.1s"""
    key = f"docx_{_md5(file_bytes)}"
    hit = _cache_get(key)
    if hit is not None:
        return hit

    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    docs: List[Document] = []
    try:
        from langchain_community.document_loaders import Docx2txtLoader
        docs = Docx2txtLoader(tmp_path).load()
    except Exception as e:
        raise RuntimeError(f"Cannot load DOCX: {e}")
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    _cache_set(key, docs)
    return docs


def load_document(file_bytes: bytes, filename: str) -> List[Document]:
    """Unified entry — auto-detect format."""
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext == "pdf":
        return load_pdf(file_bytes, filename)
    elif ext in ("docx", "doc"):
        return load_docx(file_bytes, filename)
    raise ValueError(f"Unsupported format: .{ext}")


def clear_loader_cache() -> int:
    n = 0
    for f in CACHE_DIR.glob("*.pkl"):
        f.unlink(missing_ok=True)
        n += 1
    return n


def get_cache_info() -> dict:
    files = list(CACHE_DIR.glob("*.pkl"))
    size  = sum(f.stat().st_size for f in files)
    return {"count": len(files), "size_kb": round(size / 1024, 1)}