"""
reranker.py — SmartDoc AI v1.3
═══════════════════════════════
Câu hỏi 9: Implement Re-ranking với Cross-Encoder

Yêu cầu:
  ✅ Thêm bước re-ranking sau retrieval
  ✅ Sử dụng cross-encoder model để đánh giá lại relevance
  ✅ So sánh với bi-encoder (current approach)
  ✅ Tối ưu hóa latency

Cách hoạt động:
  1. Bi-encoder (FAISS): embed query + docs riêng lẻ → cosine similarity
     Nhanh (~0.05s) nhưng kém chính xác hơn
  2. Cross-encoder: đưa (query, doc) vào cùng → cross-attention → score
     Chậm hơn (~0.5-2s) nhưng CHÍNH XÁC HƠN đáng kể

Pipeline:
  Query → FAISS retrieve top-K (nhiều) → Cross-Encoder re-rank → lấy top-N

Cài:
    pip install sentence-transformers
"""

import time
from typing import List, Optional, Tuple
from langchain_core.documents import Document

# ── Model mặc định ─────────────────────────────────────────────────────────────
# cross-encoder/ms-marco-MiniLM-L-6-v2: nhanh + tốt cho EN
# cross-encoder/mmarco-mMiniLMv2-L12-H384-v1: multilingual (VI/EN)
DEFAULT_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MULTILINGUAL_CROSS_ENCODER = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

# Flag kiểm tra sentence-transformers đã cài chưa
try:
    from sentence_transformers import CrossEncoder as _CE
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False

# ── Cache cross-encoder model ──────────────────────────────────────────────────
_ce_cache: dict = {}


def _load_cross_encoder(model_name: str):
    """Load CrossEncoder, cached singleton."""
    if model_name in _ce_cache:
        return _ce_cache[model_name]
    if not RERANKER_AVAILABLE:
        raise ImportError(
            "sentence-transformers chưa cài.\n"
            "Chạy: pip install sentence-transformers"
        )
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder(model_name)
    _ce_cache[model_name] = ce
    return ce


def rerank_documents(
    query: str,
    documents: List[Document],
    top_k: Optional[int] = None,
    model_name: str = DEFAULT_CROSS_ENCODER,
) -> List[Document]:
    """
    Re-rank documents bằng Cross-Encoder.

    Thêm metadata "rerank_score" vào mỗi document.
    Documents được sort từ cao đến thấp theo score.

    Args:
        query:      câu hỏi người dùng
        documents:  List[Document] từ FAISS/BM25 retriever
        top_k:      lấy top-k sau re-rank (None = lấy tất cả)
        model_name: cross-encoder model

    Returns:
        List[Document] đã re-rank, kèm metadata "rerank_score"

    Performance:
        - 3 docs: ~0.3-0.5s
        - 10 docs: ~1-2s
        - Nên fetch_k=10 từ FAISS, re-rank lấy top_k=3
    """
    if not documents:
        return documents

    if not RERANKER_AVAILABLE:
        # Fallback: trả về nguyên thứ tự FAISS
        return documents[:top_k] if top_k else documents

    try:
        ce = _load_cross_encoder(model_name)

        # Tạo pairs (query, doc_content)
        pairs = [(query, d.page_content[:512]) for d in documents]

        # Cross-encoder scoring
        t0     = time.time()
        scores = ce.predict(pairs)
        _t     = round(time.time() - t0, 3)

        # Gán score vào metadata
        for doc, score in zip(documents, scores):
            doc.metadata["rerank_score"]    = float(score)
            doc.metadata["rerank_time_s"]   = _t

        # Sort theo score giảm dần
        ranked = sorted(documents, key=lambda d: d.metadata.get("rerank_score", 0), reverse=True)

        return ranked[:top_k] if top_k else ranked

    except Exception as e:
        # Fallback an toàn: không raise, trả về docs gốc
        return documents[:top_k] if top_k else documents


def compare_biencoder_vs_crossencoder(
    query: str,
    documents: List[Document],
    model_name: str = DEFAULT_CROSS_ENCODER,
) -> dict:
    """
    So sánh thứ tự ranking của Bi-encoder (FAISS) vs Cross-Encoder.

    Returns:
        dict với:
        - biencoder_order: thứ tự FAISS gốc
        - crossencoder_order: thứ tự sau re-rank
        - order_changed: True nếu thứ tự thay đổi
        - scores: điểm cross-encoder của từng doc
        - rerank_time_s: thời gian re-rank
    """
    if not documents or not RERANKER_AVAILABLE:
        return {
            "error": "sentence-transformers not installed or no documents",
            "biencoder_order":    [],
            "crossencoder_order": [],
            "order_changed":      False,
            "scores":             [],
            "rerank_time_s":      0,
        }

    try:
        ce     = _load_cross_encoder(model_name)
        pairs  = [(query, d.page_content[:512]) for d in documents]
        t0     = time.time()
        scores = ce.predict(pairs)
        elapsed = round(time.time() - t0, 3)

        # Thứ tự Bi-encoder (FAISS): 0, 1, 2, ...
        biencoder_order = list(range(len(documents)))

        # Thứ tự Cross-encoder
        ranked_idx    = sorted(range(len(documents)), key=lambda i: scores[i], reverse=True)
        order_changed = ranked_idx != biencoder_order

        return {
            "biencoder_order":    biencoder_order,
            "crossencoder_order": ranked_idx,
            "order_changed":      order_changed,
            "scores":             [round(float(s), 4) for s in scores],
            "top_doc_preview":    documents[ranked_idx[0]].page_content[:150] + "…",
            "rerank_time_s":      elapsed,
            "model":              model_name,
            "summary": (
                f"Bi-encoder thứ tự: {biencoder_order}\n"
                f"Cross-encoder thứ tự: {ranked_idx}\n"
                f"{'⚠️ Thứ tự THAY ĐỔI' if order_changed else '✅ Thứ tự giống nhau'}\n"
                f"Scores: {[round(float(s),2) for s in scores]}\n"
                f"Re-rank time: {elapsed}s"
            ),
        }
    except Exception as e:
        return {
            "error":              f"{type(e).__name__}: {e}",
            "biencoder_order":    list(range(len(documents))),
            "crossencoder_order": list(range(len(documents))),
            "order_changed":      False,
            "scores":             [],
            "rerank_time_s":      0,
        }


def get_reranker_info() -> dict:
    """Thông tin về re-ranking."""
    return {
        "available":    RERANKER_AVAILABLE,
        "default_model": DEFAULT_CROSS_ENCODER,
        "multilingual":  MULTILINGUAL_CROSS_ENCODER,
        "bi_encoder": {
            "type":   "Separate encoding (FAISS)",
            "speed":  "~0.05s",
            "accuracy": "Good — cosine similarity",
            "how":    "embed(query) ⊗ embed(doc) → cosine sim",
        },
        "cross_encoder": {
            "type":   "Joint encoding (Cross-Attention)",
            "speed":  "~0.5-2s for 3-10 docs",
            "accuracy": "Better — direct interaction",
            "how":    "CrossEncoder([query, doc]) → relevance score",
        },
        "recommendation": (
            "Dùng Bi-encoder để retrieve top-10 (nhanh), "
            "rồi Cross-encoder để re-rank lấy top-3 (chính xác)"
        ),
        "install": "pip install sentence-transformers",
    }