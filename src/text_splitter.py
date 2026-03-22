"""
text_splitter.py — SmartDoc AI v1.3
═════════════════════════════════════
Target: Retrieval accuracy 85-90%, chunking <1s/100 chunks
"""

from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_PRESETS = {
    "fast":     {"chunk_size": 500,  "chunk_overlap": 50},
    "balanced": {"chunk_size": 1000, "chunk_overlap": 100},  # ← default sweet spot
    "accurate": {"chunk_size": 1500, "chunk_overlap": 150},
    "dense":    {"chunk_size": 2000, "chunk_overlap": 200},
}

# Separators tối ưu VI + EN
_SEP = ["\n\n", "\n", ". ", ".\n", "! ", "? ", "; ", ", ", " ", ""]


def split_documents(
    docs: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
    preset: Optional[str] = None,
) -> List[Document]:
    """
    Split documents → chunks với metadata chunk_id + char_count.
    100 chunks ≈ 0.3-0.8s
    """
    if preset in CHUNK_PRESETS:
        cfg = CHUNK_PRESETS[preset]
        chunk_size, chunk_overlap = cfg["chunk_size"], cfg["chunk_overlap"]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=_SEP,
        length_function=len,
        is_separator_regex=False,
        keep_separator=True,
    )
    chunks = splitter.split_documents(docs)
    total  = len(chunks)
    for i, c in enumerate(chunks):
        c.metadata.update({
            "chunk_id":    i,
            "chunk_total": total,
            "char_count":  len(c.page_content),
        })
    return chunks


def get_chunk_stats(chunks: List[Document]) -> dict:
    if not chunks:
        return {}
    sizes = [len(c.page_content) for c in chunks]
    return {
        "count": len(chunks),
        "avg":   round(sum(sizes) / len(sizes)),
        "min":   min(sizes),
        "max":   max(sizes),
    }