"""
conftest.py — pytest fixtures cho SmartDoc AI tests
Chạy: pytest tests/ -v
"""

import sys
import pytest
from pathlib import Path

# Thêm project root vào sys.path để import src/
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── Shared fixtures ────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def test_embedder():
    """
    Shared embedding model cho toàn bộ test session.
    Dùng model nhỏ nhất để test nhanh (all-MiniLM-L6-v2, 384-dim).
    """
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
    )


@pytest.fixture(scope="session")
def tech_manual_chunks():
    """Chunks từ technical manual (Test Case 1)."""
    from langchain_core.documents import Document
    return [
        Document(
            page_content=(
                "Installation Guide\n"
                "Step 1: Download the installer from https://example.com\n"
                "Step 2: Run installer as administrator\n"
                "Step 3: Accept license and click Next\n"
                "Step 4: Choose directory and click Install\n"
                "Step 5: Restart computer when complete\n"
            ),
            metadata={"page": 0, "source": "tech_manual_test"}
        ),
        Document(
            page_content=(
                "System Requirements: Windows 10/11, 4GB RAM, 2GB disk space.\n"
                "Troubleshooting: Error 0x80070005 = insufficient permissions.\n"
            ),
            metadata={"page": 1, "source": "tech_manual_test"}
        ),
    ]


@pytest.fixture(scope="session")
def research_paper_chunks():
    """Chunks từ research paper (Test Case 2)."""
    from langchain_core.documents import Document
    return [
        Document(
            page_content=(
                "Main findings: RAG achieves 85-90% retrieval accuracy, "
                "outperforming pure LLM by 23%. "
                "Implications: Local deployment reduces costs significantly."
            ),
            metadata={"page": 0, "source": "research_test"}
        ),
        Document(
            page_content=(
                "Hybrid search (BM25+FAISS) improves recall by 12%. "
                "Multilingual models essential for non-English documents."
            ),
            metadata={"page": 3, "source": "research_test"}
        ),
    ]


@pytest.fixture(scope="session")
def cooking_chunks():
    """Chunks từ cooking recipe (Test Case 3 — out-of-context)."""
    from langchain_core.documents import Document
    return [
        Document(
            page_content=(
                "Pho Bo Recipe: beef bones, noodles, star anise.\n"
                "Instructions: boil bones 3 hours, strain broth, serve hot."
            ),
            metadata={"page": 0, "source": "cooking_test"}
        ),
    ]


@pytest.fixture(scope="session")
def faiss_retriever(tech_manual_chunks, test_embedder):
    """FAISS retriever ready để dùng trong tests."""
    from langchain_community.vectorstores import FAISS
    vector = FAISS.from_documents(tech_manual_chunks, test_embedder)
    return vector.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2}
    )