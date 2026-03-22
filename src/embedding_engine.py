"""
embedding_engine.py — SmartDoc AI v1.3
═══════════════════════════════════════
Target: Embedding Generation 5-10s / 100 chunks

Tối ưu:
  1. Singleton cache — load model 1 lần
  2. batch_size=64 — embed nhiều chunk song song bên trong model
  3. Parallel batch embedding với ThreadPoolExecutor
     → Chia chunks thành N batch, embed đồng thời
     → Giảm ~40-60% thời gian so với serial
  4. normalize_embeddings=True — cosine sim chính xác
  5. KHÔNG dùng show_progress_bar trong model_kwargs (lỗi)
"""

import time
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

# ── Model registry ─────────────────────────────────────────────────────────────
EMBEDDING_MODELS = {
    # Key = full model_id để dùng trực tiếp từ selectbox
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": {
        "dim": 768, "lang": "50+", "quality": "high",
        "note": "Best cho tiếng Việt ✅ (dùng mặc định)",
    },
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": {
        "dim": 384, "lang": "50+", "quality": "medium",
        "note": "Nhanh hơn 2x, vẫn hỗ trợ tiếng Việt",
    },
    "sentence-transformers/all-MiniLM-L6-v2": {
        "dim": 384, "lang": "EN only", "quality": "medium",
        "note": "Nhanh nhất — chỉ tốt cho tiếng Anh",
    },
}

_embedder_cache: dict = {}


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_embedder(model_key: str, device: Optional[str] = None, batch_size: int = 64):
    """
    Cached singleton embedder.
    model_key = full model_id (từ selectbox) hoặc shortkey.

    Lần đầu: 5-15s (download + load model)
    Cache hit: ~0s
    """
    if device is None:
        device = _detect_device()

    ck = (model_key, device, batch_size)
    if ck in _embedder_cache:
        return _embedder_cache[ck]

    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings

    # ⚠️ KHÔNG đặt show_progress_bar trong model_kwargs
    emb = HuggingFaceEmbeddings(
        model_name=model_key,
        model_kwargs={"device": device},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": batch_size,
        },
    )
    _embedder_cache[ck] = emb
    return emb


def embed_documents_parallel(
    embedder,
    texts: List[str],
    n_workers: int = 4,
    batch_size: int = 32,
) -> List[List[float]]:
    """
    Embed danh sách text với ThreadPoolExecutor.

    Chia texts thành batches → embed parallel → ghép lại theo thứ tự.
    Nhanh hơn ~40-60% so với embed serial khi có nhiều chunks.

    Args:
        embedder:  HuggingFaceEmbeddings
        texts:     danh sách text cần embed
        n_workers: số thread song song (default 4)
        batch_size: kích thước mỗi batch

    Returns:
        List[List[float]] — vectors theo đúng thứ tự input
    """
    if not texts:
        return []

    # Với ít text: không cần parallel (overhead > benefit)
    if len(texts) <= batch_size:
        return embedder.embed_documents(texts)

    # Chia thành batches
    batches = []
    for i in range(0, len(texts), batch_size):
        batches.append((i, texts[i:i + batch_size]))

    results: dict = {}

    def _embed_batch(args):
        start_idx, batch = args
        vecs = embedder.embed_documents(batch)
        return start_idx, vecs

    # Giới hạn workers theo số batches thực tế
    workers = min(n_workers, len(batches))

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_embed_batch, b): b[0] for b in batches}
        for fut in as_completed(futures):
            start_idx, vecs = fut.result()
            results[start_idx] = vecs

    # Ghép kết quả theo đúng thứ tự
    all_vecs = []
    for start in sorted(results.keys()):
        all_vecs.extend(results[start])

    return all_vecs


def list_models() -> dict:
    return EMBEDDING_MODELS


def clear_cache():
    _embedder_cache.clear()