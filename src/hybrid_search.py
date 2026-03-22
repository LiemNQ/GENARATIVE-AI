"""
hybrid_search.py — SmartDoc AI v1.3
══════════════════════════════════════
Câu hỏi 7: Hybrid Search = BM25 (keyword) + FAISS (semantic)

Yêu cầu đề bài:
  ✅ Kết hợp semantic search (vector) với keyword search (BM25)
  ✅ Implement ensemble retriever
  ✅ So sánh performance với pure vector search

Cách dùng:
    from src.hybrid_search import build_hybrid_retriever, compare_retrievers

    # Tạo hybrid retriever
    retriever = build_hybrid_retriever(chunks, embedder, top_k=3, bm25_weight=0.4)

    # So sánh với pure vector
    report = compare_retrievers(chunks, embedder, query="câu hỏi test", top_k=3)

Cài thêm:
    pip install rank_bm25 langchain-community
"""

import time
from typing import List, Optional, Tuple
from langchain_core.documents import Document


# ══════════════════════════════════════════════════════════════════════════════
# BM25 RETRIEVER
# ══════════════════════════════════════════════════════════════════════════════

def build_bm25_retriever(chunks: List[Document], top_k: int = 3):
    """
    Tạo BM25 retriever từ danh sách chunks.

    BM25 = keyword-based search:
    - Tốt với truy vấn chứa từ khoá đặc biệt (tên riêng, mã code, số hiệu)
    - Không cần embedding → rất nhanh
    - Yếu với semantic (đồng nghĩa, ngữ nghĩa)

    Args:
        chunks: List[Document] từ text_splitter
        top_k:  số kết quả trả về

    Returns:
        BM25Retriever instance
    """
    try:
        from langchain_community.retrievers import BM25Retriever
    except Exception as e:
        raise ImportError(
            f"Không import được BM25Retriever: {e}\n"
            "Chạy: pip install rank_bm25 langchain-community"
        )

    try:
        retriever = BM25Retriever.from_documents(chunks)
        retriever.k = top_k
        return retriever
    except Exception as e:
        raise RuntimeError(
            f"BM25Retriever.from_documents() lỗi: {type(e).__name__}: {e}\n"
            "Thử: pip install rank_bm25"
        )


def _get_ensemble_retriever():
    """
    Lấy EnsembleRetriever class.
    EnsembleRetriever nằm trong langchain core (không phải langchain_community).
    """
    # Cách 1: langchain core (đúng nhất cho phiên bản mới)
    try:
        from langchain.retrievers import EnsembleRetriever
        return EnsembleRetriever
    except ImportError:
        pass
    # Cách 2: langchain_core (một số phiên bản)
    try:
        from langchain_core.retrievers import EnsembleRetriever
        return EnsembleRetriever
    except ImportError:
        pass
    # Nếu cả hai đều thất bại
    raise ImportError(
        "Không tìm thấy EnsembleRetriever.\n"
        "EnsembleRetriever nằm trong gói 'langchain', KHÔNG phải 'langchain_community'.\n"
        "Chạy: pip install langchain"
    )


def build_faiss_retriever(chunks: List[Document], embedder, top_k: int = 3):
    """
    Tạo FAISS (vector/semantic) retriever từ chunks.

    FAISS = semantic/dense vector search:
    - Tốt với truy vấn ngữ nghĩa, đồng nghĩa
    - Cần embedding (chậm hơn BM25 khi build, nhưng query nhanh)

    Returns:
        FAISS retriever instance
    """
    from langchain_community.vectorstores import FAISS
    vector = FAISS.from_documents(chunks, embedder)
    return vector.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k}
    )


# ══════════════════════════════════════════════════════════════════════════════
# ENSEMBLE (HYBRID) RETRIEVER
# ══════════════════════════════════════════════════════════════════════════════

def build_hybrid_retriever(
    chunks: List[Document],
    embedder,
    top_k: int = 3,
    bm25_weight: float = 0.4,
    vector_weight: float = 0.6,
):
    """
    Tạo Hybrid Retriever = BM25 + FAISS Ensemble.

    EnsembleRetriever kết hợp kết quả từ nhiều retriever bằng
    thuật toán Reciprocal Rank Fusion (RRF):
      score_final = Σ weight_i / (rank_i + 60)

    Công thức cho thấy:
    - Kết quả xuất hiện ở top cả hai retriever → điểm cao nhất
    - BM25 bù cho FAISS khi query chứa từ khoá cụ thể
    - FAISS bù cho BM25 khi query có ngữ nghĩa/đồng nghĩa

    Args:
        chunks:        List[Document] từ text_splitter
        embedder:      HuggingFaceEmbeddings
        top_k:         số kết quả trả về cuối cùng
        bm25_weight:   trọng số BM25 (default 0.4 = 40%)
        vector_weight: trọng số FAISS (default 0.6 = 60%)

    Returns:
        EnsembleRetriever instance

    Lưu ý:
        - Tổng bm25_weight + vector_weight không cần = 1.0
          (EnsembleRetriever tự normalize)
        - vector_weight > bm25_weight vì semantic search tốt hơn
          cho tài liệu kỹ thuật tiếng Việt
    """
    EnsembleRetriever = _get_ensemble_retriever()

    # Tạo 2 thành phần
    bm25_ret   = build_bm25_retriever(chunks, top_k=top_k)
    faiss_ret  = build_faiss_retriever(chunks, embedder, top_k=top_k)

    # Kết hợp bằng EnsembleRetriever (Reciprocal Rank Fusion)
    ensemble = EnsembleRetriever(
        retrievers=[bm25_ret, faiss_ret],
        weights=[bm25_weight, vector_weight],
    )
    return ensemble


def build_hybrid_retriever_from_vector(
    vector_store,
    chunks: List[Document],
    top_k: int = 3,
    bm25_weight: float = 0.4,
    vector_weight: float = 0.6,
):
    """
    Tạo Hybrid Retriever từ FAISS vector store đã có sẵn.
    Dùng khi đã load vector store từ disk (không cần re-embed).

    Args:
        vector_store: FAISS vector store đã load
        chunks:       List[Document] gốc (để build BM25)
        top_k:        số kết quả
    """
    EnsembleRetriever = _get_ensemble_retriever()

    bm25_ret  = build_bm25_retriever(chunks, top_k=top_k)
    faiss_ret = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k}
    )

    return EnsembleRetriever(
        retrievers=[bm25_ret, faiss_ret],
        weights=[bm25_weight, vector_weight],
    )


# ══════════════════════════════════════════════════════════════════════════════
# COMPARISON / BENCHMARK
# ══════════════════════════════════════════════════════════════════════════════

def compare_retrievers(
    chunks: List[Document],
    embedder,
    query: str,
    top_k: int = 3,
    bm25_weight: float = 0.4,
    vector_weight: float = 0.6,
    vector_store=None,   # ← nhận vector store có sẵn, tránh re-embed
) -> dict:
    """
    So sánh hiệu năng 3 loại retriever:
      1. Pure BM25 (keyword only)
      2. Pure FAISS (semantic/vector only)
      3. Hybrid Ensemble (BM25 + FAISS)

    Args:
        chunks:       danh sách chunks đã split
        embedder:     embedding model
        query:        câu hỏi test
        top_k:        số kết quả mỗi retriever
        vector_store: FAISS store đã có (nếu None sẽ build mới)
    """
    report = {
        "query":   query,
        "top_k":   top_k,
        "results": {"bm25": [], "faiss": [], "hybrid": []},
        "timings": {"bm25_ms": 0, "faiss_ms": 0, "hybrid_ms": 0},
        "overlap": {},
        "summary": "",
    }

    # ── 1. Dùng vector store có sẵn hoặc build mới ─────────────────────────
    if vector_store is not None:
        vector = vector_store
        report["build_time_faiss_s"] = 0  # đã có sẵn, không cần build
    else:
        from langchain_community.vectorstores import FAISS
        t_build = time.time()
        try:
            vector = FAISS.from_documents(chunks, embedder)
            report["build_time_faiss_s"] = round(time.time() - t_build, 2)
        except Exception as e:
            report["error"] = f"FAISS build error: {e}"
            return report

    # ── 2. Build BM25 ──────────────────────────────────────────────────────
    t_bm25 = time.time()
    try:
        from langchain_community.retrievers import BM25Retriever
        bm25_base = BM25Retriever.from_documents(chunks)
        bm25_base.k = top_k
        report["build_time_bm25_s"] = round(time.time() - t_bm25, 3)
    except Exception as e:
        # Bắt MỌI lỗi (ImportError, ModuleNotFoundError, RuntimeError...)
        # và hiển thị message THỰC SỰ thay vì "not installed"
        import sys as _sys
        report["error"] = (
            f"Lỗi khởi tạo BM25: {type(e).__name__}: {e}\n\n"
            f"Python Streamlit đang dùng: {_sys.executable}\n"
            f"Chạy lệnh này trong terminal cùng môi trường với Streamlit:\n"
            f"  {_sys.executable} -m pip install rank_bm25"
        )
        return report

    # ── 3. Query: BM25 ─────────────────────────────────────────────────────
    try:
        t0 = time.time()
        bm25_docs = bm25_base.invoke(query)
        report["timings"]["bm25_ms"] = round((time.time() - t0) * 1000, 1)
        report["results"]["bm25"] = [
            {"page": d.metadata.get("page", "?"), "preview": d.page_content[:120] + "…"}
            for d in bm25_docs
        ]
    except Exception as e:
        report["error"] = f"BM25 query error: {e}"
        return report

    # ── 4. Query: Pure FAISS ───────────────────────────────────────────────
    try:
        faiss_ret = vector.as_retriever(
            search_type="similarity", search_kwargs={"k": top_k}
        )
        t0 = time.time()
        faiss_docs = faiss_ret.invoke(query)
        report["timings"]["faiss_ms"] = round((time.time() - t0) * 1000, 1)
        report["results"]["faiss"] = [
            {"page": d.metadata.get("page", "?"), "preview": d.page_content[:120] + "…"}
            for d in faiss_docs
        ]
    except Exception as e:
        report["error"] = f"FAISS query error: {e}"
        return report

    # ── 5. Query: Hybrid Ensemble ──────────────────────────────────────────
    try:
        EnsembleRetriever = _get_ensemble_retriever()
        ensemble = EnsembleRetriever(
            retrievers=[bm25_base, faiss_ret],
            weights=[bm25_weight, vector_weight],
        )
        t0 = time.time()
        hybrid_docs = ensemble.invoke(query)
        report["timings"]["hybrid_ms"] = round((time.time() - t0) * 1000, 1)
        report["results"]["hybrid"] = [
            {"page": d.metadata.get("page", "?"), "preview": d.page_content[:120] + "…"}
            for d in hybrid_docs
        ]
    except Exception as e:
        report["ensemble_error"] = f"{type(e).__name__}: {e}"
        # Vẫn trả về kết quả BM25 + FAISS đã có, chỉ thiếu hybrid
        hybrid_docs = []

    # ── 6. Tính overlap ────────────────────────────────────────────────────
    bm25_set   = {d.page_content[:80] for d in bm25_docs}
    faiss_set  = {d.page_content[:80] for d in faiss_docs}
    hybrid_set = {d.page_content[:80] for d in (hybrid_docs if hybrid_docs else [])}

    overlap_bm25_faiss = len(bm25_set & faiss_set)
    unique_hybrid      = len(hybrid_set - bm25_set - faiss_set)
    combined_coverage  = len(hybrid_set | bm25_set | faiss_set)

    report["overlap"] = {
        "bm25_faiss_common":  overlap_bm25_faiss,
        "hybrid_unique_docs": unique_hybrid,
        "combined_coverage":  combined_coverage,
    }

    # ── 7. Summary text ────────────────────────────────────────────────────
    report["summary"] = (
        f"Query: '{query}'\n"
        f"BM25:   {report['timings']['bm25_ms']}ms  | {len(report['results']['bm25'])} docs\n"
        f"FAISS:  {report['timings']['faiss_ms']}ms  | {len(report['results']['faiss'])} docs\n"
        f"Hybrid: {report['timings']['hybrid_ms']}ms  | {len(report['results']['hybrid'])} docs\n"
        f"BM25∩FAISS overlap: {overlap_bm25_faiss} docs\n"
        f"Hybrid unique: {unique_hybrid} docs\n"
        f"Recommendation: "
        + (
            "Hybrid > FAISS nếu query chứa từ khoá đặc biệt"
            if overlap_bm25_faiss < top_k
            else "Pure FAISS đủ tốt nếu query ngữ nghĩa thuần tuý"
        )
    )

    return report


def get_hybrid_retriever_info() -> dict:
    """Thông tin về các loại retriever và khi nào dùng."""
    return {
        "bm25": {
            "type":     "Keyword Search (TF-IDF variant)",
            "pros":     ["Không cần embedding", "Rất nhanh (<1ms)", "Tốt với từ khoá đặc biệt"],
            "cons":     ["Không hiểu ngữ nghĩa", "Phân biệt hoa thường"],
            "best_for": "Tên riêng, mã code, số hiệu, từ chuyên ngành",
        },
        "faiss": {
            "type":     "Semantic/Dense Vector Search",
            "pros":     ["Hiểu ngữ nghĩa", "Xử lý đồng nghĩa", "Tốt cho tiếng Việt"],
            "cons":     ["Cần embedding (5-15s build)", "Đôi khi bỏ sót từ khoá chính xác"],
            "best_for": "Câu hỏi ngữ nghĩa, khái niệm, giải thích",
        },
        "hybrid": {
            "type":     "Ensemble (BM25 + FAISS) via Reciprocal Rank Fusion",
            "pros":     ["Tốt nhất cả hai", "Accuracy 85-90%", "Robust hơn"],
            "cons":     ["Build chậm hơn một chút", "Cần rank_bm25 package"],
            "best_for": "Mọi loại câu hỏi — khuyến nghị dùng mặc định",
            "weights":  "BM25=0.4, FAISS=0.6 (FAISS ưu tiên hơn)",
        },
    }