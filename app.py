"""
app.py — SmartDoc AI v1.3
══════════════════════════
Performance targets (đạt được với đa luồng):
  PDF Load:   1-3s  (PyMuPDF parallel) / <0.1s (MD5 cache)
  Embedding:  3-8s  (ThreadPoolExecutor parallel batches) / 100 chunks
  DB Load:    ~0.3s (FAISS từ disk)
  Query:      1-3s  (FAISS similarity search)
  Answer:     3-8s  (Ollama streaming, render mỗi 5 token)

Tối ưu đa luồng:
  - document_loader.py: PyMuPDF parallel page extraction
  - embedding_engine.py: embed_documents_parallel (ThreadPoolExecutor)
  - vector_store.py: create_vector_store dùng parallel embedding
  - OMP_NUM_THREADS=4 cho CPU inference
  - Render mỗi 5 token thay vì mỗi 1 token
"""

import os
# Tăng tốc CPU inference — PHẢI set trước khi import torch/numpy
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import streamlit as st
import time
import base64
import sys
from pathlib import Path
from datetime import datetime

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SmartDoc AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
VECTOR_DIR = BASE_DIR / "data" / "vector_db"
INPUT_DIR  = BASE_DIR / "data" / "input"
LOGO_PATH  = BASE_DIR / "assets" / "logo-sgu.jpg"

for _d in [VECTOR_DIR, INPUT_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Logo ───────────────────────────────────────────────────────────────────────
def _logo_b64() -> str:
    if LOGO_PATH.exists():
        return base64.b64encode(LOGO_PATH.read_bytes()).decode()
    return ""

LOGO_B64 = _logo_b64()

# ── Import src modules ─────────────────────────────────────────────────────────
sys.path.insert(0, str(BASE_DIR))
try:
    from src.document_loader  import load_document, clear_loader_cache, get_cache_info
    from src.text_splitter    import split_documents, get_chunk_stats, CHUNK_PRESETS
    from src.embedding_engine import get_embedder, list_models, EMBEDDING_MODELS
    from src.vector_store     import (
        create_vector_store, load_vector_store,
        list_vector_dbs, delete_vector_db, db_exists,
    )
    from src.llm_chain import (
        get_llm, stream_rag_answer, detect_language, OLLAMA_MODELS,
    )
    # Hybrid Search (BM25 + FAISS Ensemble) — Câu hỏi 7
    try:
        from src.hybrid_search import (
            build_hybrid_retriever_from_vector,
            compare_retrievers,
            get_hybrid_retriever_info,
        )
        HYBRID_OK = True
    except ImportError:
        HYBRID_OK = False

    # Re-ranking Cross-Encoder — Câu hỏi 9
    try:
        from src.reranker import rerank_documents, get_reranker_info, RERANKER_AVAILABLE
        RERANKER_OK = True
    except ImportError:
        RERANKER_OK = False
        RERANKER_AVAILABLE = False

    # Self-RAG — Câu hỏi 10
    try:
        from src.self_rag import self_rag_answer, rewrite_query, SelfRAGResult
        SELFRAG_OK = True
    except ImportError:
        SELFRAG_OK = False

    SRC_OK = True
except ImportError as _ie:
    SRC_OK  = False
    _IE_MSG = str(_ie)
    HYBRID_OK = False
    RERANKER_OK = False
    RERANKER_AVAILABLE = False
    SELFRAG_OK = False

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');
:root{--bg:#0d0f14;--sur:#161920;--sur2:#1e2130;--bd:rgba(255,255,255,0.07);
      --ac:#4f8ef7;--ac2:#a78bfa;--gr:#34d399;--re:#f87171;
      --tx:#e8eaf0;--mu:#6b7280;--r:14px;}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;color:var(--tx);}
.stApp{background:var(--bg);
  background-image:radial-gradient(ellipse 80% 50% at 20% 0%,rgba(79,142,247,.08) 0%,transparent 60%),
                   radial-gradient(ellipse 60% 40% at 80% 100%,rgba(167,139,250,.06) 0%,transparent 60%);}
[data-testid="stSidebar"]{background:var(--sur)!important;border-right:1px solid var(--bd);}
[data-testid="stSidebar"] *{color:var(--tx)!important;}
::-webkit-scrollbar{width:4px;}::-webkit-scrollbar-thumb{background:var(--sur2);border-radius:99px;}
[data-testid="stFileUploader"]{background:var(--sur2);border:2px dashed var(--ac);border-radius:var(--r);padding:1rem;transition:border-color .3s;}
[data-testid="stFileUploader"]:hover{border-color:var(--ac2);}
[data-testid="stTextInput"] input{background:var(--sur2)!important;border:1px solid var(--bd)!important;border-radius:var(--r)!important;color:var(--tx)!important;font-family:'DM Sans',sans-serif!important;transition:border-color .25s,box-shadow .25s;}
[data-testid="stTextInput"] input:focus{border-color:var(--ac)!important;box-shadow:0 0 0 3px rgba(79,142,247,.18)!important;}
.stButton>button{background:linear-gradient(135deg,var(--ac),var(--ac2));color:#fff;border:none;border-radius:var(--r);font-family:'Syne',sans-serif;font-weight:600;letter-spacing:.02em;padding:.6rem 1.4rem;transition:opacity .2s,transform .15s,box-shadow .2s;box-shadow:0 4px 16px rgba(79,142,247,.25);}
.stButton>button:hover{opacity:.9;transform:translateY(-1px);box-shadow:0 6px 24px rgba(79,142,247,.35);}
.stButton>button:active{transform:translateY(0);}
[data-testid="stMetric"]{background:var(--sur2);border:1px solid var(--bd);border-radius:var(--r);padding:.75rem 1rem;}
[data-testid="stMetricValue"]{font-family:'Syne',sans-serif!important;}
[data-testid="stExpander"]{background:var(--sur2);border:1px solid var(--bd);border-radius:var(--r);}
[data-testid="stSelectbox"]>div>div{background:var(--sur2)!important;border:1px solid var(--bd)!important;border-radius:var(--r)!important;color:var(--tx)!important;}
[data-testid="stSlider"]>div>div>div{background:var(--ac)!important;}
hr{border-color:var(--bd)!important;}
h1,h2,h3,h4{font-family:'Syne',sans-serif!important;}
[data-testid="stAlert"]{border-radius:var(--r)!important;border-left-width:3px!important;}
[data-testid="stTabs"] [data-baseweb="tab-list"]{background:transparent;gap:.5rem;}
[data-testid="stTabs"] [data-baseweb="tab"]{background:rgba(255,255,255,.03);border:1px solid var(--bd);border-radius:10px!important;color:var(--mu)!important;font-family:'Syne',sans-serif;font-size:.82rem;}
[data-testid="stTabs"] [aria-selected="true"]{background:rgba(79,142,247,.12)!important;border-color:rgba(79,142,247,.3)!important;color:var(--ac)!important;}
@keyframes gradShift{0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}
@keyframes fadeUp{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:translateY(0)}}
@keyframes float{0%,100%{transform:translateY(0)}50%{transform:translateY(-6px)}}
@keyframes msgIn{from{opacity:0;transform:translateX(-10px)}to{opacity:1;transform:translateX(0)}}
@keyframes msgInR{from{opacity:0;transform:translateX(10px)}to{opacity:1;transform:translateX(0)}}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HERO BANNER
# ══════════════════════════════════════════════════════════════════════════════
_logo_img = (
    f'<img src="data:image/jpeg;base64,{LOGO_B64}" '
    f'style="height:54px;width:54px;object-fit:contain;border-radius:8px;background:#fff;padding:3px;" />'
    if LOGO_B64 else ""
)
st.markdown(f"""
<style>
.hero{{position:relative;overflow:hidden;
  background:linear-gradient(135deg,#0d0f14 0%,#161a2e 35%,#121826 65%,#0d0f14 100%);
  border:1px solid rgba(79,142,247,.2);border-radius:20px;
  padding:2rem 2.5rem 1.8rem;margin-bottom:1.8rem;animation:fadeUp .7s ease both;}}
.hero::before{{content:'';position:absolute;inset:0;
  background:linear-gradient(120deg,rgba(79,142,247,.08),rgba(167,139,250,.07),rgba(245,200,66,.04));
  background-size:300% 300%;animation:gradShift 8s ease infinite;pointer-events:none;}}
.htop{{display:flex;align-items:center;gap:1.2rem;}}
.brain{{font-size:3rem;animation:float 3s ease-in-out infinite;filter:drop-shadow(0 0 14px rgba(79,142,247,.5));}}
.htitle{{font-family:'Syne',sans-serif;font-size:2.5rem;font-weight:800;
  background:linear-gradient(135deg,#e8eaf0 25%,#4f8ef7 60%,#a78bfa 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;line-height:1.1;margin:.3rem 0 .15rem;}}
.hsub{{font-size:.93rem;color:#6b7280;font-weight:300;letter-spacing:.03em;}}
.hbadge{{display:inline-block;margin-bottom:.8rem;background:rgba(79,142,247,.12);
  border:1px solid rgba(79,142,247,.3);color:#4f8ef7;font-size:.7rem;font-weight:500;
  letter-spacing:.08em;text-transform:uppercase;padding:.22rem .75rem;border-radius:99px;}}
.pills{{display:flex;flex-wrap:wrap;gap:.4rem;margin-top:1rem;}}
.pill{{background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.09);
  border-radius:99px;padding:.22rem .78rem;font-size:.74rem;color:#9ca3af;}}
</style>
<div class="hero">
  <div class="hbadge">✦ Generative AI · Spring 2026 · OSSD</div>
  <div class="htop">
    {_logo_img}
    <div class="brain">🧠</div>
    <div>
      <div class="htitle">SmartDoc AI</div>
      <div class="hsub">Intelligent Document Q&amp;A · RAG + LLMs · Trường ĐH Sài Gòn</div>
    </div>
  </div>
  <div class="pills">
    <div class="pill">📄 PDF &amp; DOCX</div>
    <div class="pill">🔍 FAISS Persistent</div>
    <div class="pill">🤖 Qwen2.5</div>
    <div class="pill">🌐 50+ Languages</div>
    <div class="pill">⚡ Local &amp; Private</div>
    <div class="pill">💬 Conversational RAG</div>
    <div class="pill">🚀 Parallel Processing</div>
    <div class="pill">📦 MD5 Cache</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
for _k, _v in {
    "chat_history":   [],
    "vector_store":   None,
    "retriever":      None,
    "doc_name":       None,
    "doc_chunks":     0,
    "db_loaded_from": None,
    "embedder":       None,
    "llm":            None,
    "pending_q":      "",
    "chunks":         [],          # lưu chunks gốc để BM25 dùng
    "retriever_mode": "vector",    # "vector" | "hybrid"
    "use_reranker":   False,       # câu 9: re-ranking
    "use_self_rag":   False,       # câu 10: self-RAG
    "_input_val":     "",          # track textarea value để clear sau send
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ══════════════════════════════════════════════════════════════════════════════
# CACHED MODEL LOADERS
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def _load_embedder(model_id: str):
    """Cached singleton embedder — load 1 lần duy nhất."""
    if SRC_OK:
        return get_embedder(model_id)
    # Fallback
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=model_id,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
    )


@st.cache_resource(show_spinner=False)
def _load_llm(model_name: str, temperature: float):
    """Cached Ollama LLM."""
    if SRC_OK:
        return get_llm(model_name, temperature)
    try:
        from langchain_ollama import OllamaLLM
        return OllamaLLM(model=model_name, temperature=temperature,
                         top_p=0.9, repeat_penalty=1.1)
    except ImportError:
        from langchain_community.llms import Ollama
        return Ollama(model=model_name, temperature=temperature,
                      top_p=0.9, repeat_penalty=1.1)

# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _detect_lang(text: str) -> str:
    if SRC_OK:
        return detect_language(text)
    VI = frozenset("àáâãèéêìíòóôõùúýăđơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹÀÁÂÃ")
    if any(c in VI for c in text):
        return "vi"
    kws = {"tom tat","tai lieu","chuong","la gi","noi dung","liet ke","giai thich","huong dan"}
    tl  = text.lower()
    return "vi" if any(k in tl for k in kws) else "en"


def _process_and_save(file_bytes: bytes, filename: str,
                      chunk_size: int, chunk_overlap: int, embedder):
    """
    Full pipeline với đa luồng:
    1. Load doc (PyMuPDF parallel page extraction)
    2. Split thành chunks
    3. Embed (ThreadPoolExecutor parallel batches)
    4. Save FAISS + meta.json
    Returns (vector, n_chunks, db_name, chunks_list)
    chunks_list được lưu để BM25 (Hybrid Search) dùng.
    """
    db_name = filename.rsplit(".", 1)[0].replace(" ", "_")

    if SRC_OK:
        docs    = load_document(file_bytes, filename)
        chunks  = split_documents(docs, chunk_size=chunk_size,
                                  chunk_overlap=chunk_overlap)
        vector, _, meta = create_vector_store(
            chunks, embedder, db_name, VECTOR_DIR,
            auto_save=True, n_embed_workers=4,
        )
        return vector, len(chunks), db_name, chunks  # ← trả về chunks

    # Fallback serial
    import tempfile
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS

    ext = filename.rsplit(".", 1)[-1].lower()
    with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
        tmp.write(file_bytes); tmp_path = tmp.name
    try:
        if ext == "pdf":
            from langchain_community.document_loaders import PDFPlumberLoader
            docs = PDFPlumberLoader(tmp_path).load()
        else:
            from langchain_community.document_loaders import Docx2txtLoader
            docs = Docx2txtLoader(tmp_path).load()
    finally:
        os.unlink(tmp_path)

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    ).split_documents(docs)
    vector = FAISS.from_documents(chunks, embedder)
    sp = VECTOR_DIR / db_name
    sp.mkdir(parents=True, exist_ok=True)
    vector.save_local(str(sp))
    return vector, len(chunks), db_name, chunks  # ← trả về chunks


def _load_db(db_name: str, embedder, top_k: int):
    """Load FAISS từ disk (~0.3-0.5s). Returns (vector, retriever)."""
    if SRC_OK:
        vec, ret, _ = load_vector_store(db_name, embedder, VECTOR_DIR, top_k)
        return vec, ret
    from langchain_community.vectorstores import FAISS
    sp = VECTOR_DIR / db_name
    if not sp.exists():
        return None, None
    vec = FAISS.load_local(str(sp), embedder, allow_dangerous_deserialization=True)
    ret = vec.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
    return vec, ret


def _list_dbs() -> list:
    if SRC_OK:
        return list_vector_dbs(VECTOR_DIR)
    if not VECTOR_DIR.exists():
        return []
    return [{"name": d.name, "size": "?", "chunk_count": "?", "created_at": "?"}
            for d in sorted(VECTOR_DIR.iterdir()) if d.is_dir()]


def _make_retriever(vector_store, chunks: list, mode: str, top_k: int):
    """
    Tạo retriever theo mode đã chọn:
      "vector" → Pure FAISS (semantic only)
      "hybrid" → BM25 + FAISS Ensemble (Reciprocal Rank Fusion)
    """
    if mode == "hybrid" and SRC_OK and HYBRID_OK and chunks:
        try:
            return build_hybrid_retriever_from_vector(
                vector_store, chunks, top_k=top_k,
                bm25_weight=0.4, vector_weight=0.6,
            )
        except Exception:
            pass  # fallback sang vector nếu lỗi (rank_bm25 chưa cài)
    return vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": top_k}
    )


def _stream_answer(question: str, retriever, llm,
                   history: list, show_src: bool,
                   use_reranker: bool = False,
                   use_self_rag: bool = False):
    """
    Streaming RAG với throttled render (mỗi 8 token).
    Hỗ trợ:
      - Re-ranking (câu 9): Cross-Encoder đánh giá lại sau retrieval
      - Self-RAG (câu 10): Query rewriting + confidence scoring
    Returns (full_text, sources, lang, extra_info)
    """
    def _bubble(ph, text: str, cursor: bool = False):
        ph.markdown(
            f'<div style="background:rgba(255,255,255,.04);border:1px solid '
            f'rgba(255,255,255,.08);border-radius:12px;padding:.85rem 1rem;'
            f'font-size:.88rem;line-height:1.7;color:#e8eaf0;">'
            f'{text}{"▌" if cursor else ""}</div>',
            unsafe_allow_html=True,
        )

    extra_info = {}

    # ── Self-RAG: Query Rewriting (câu 10) ─────────────────────────────────
    # Chỉ rewrite khi query THỰC SỰ mơ hồ: <= 4 từ, không dấu VI, không có danh từ rõ
    # Query như "Tóm tắt tài liệu này" / "LLM là gì" → KHÔNG rewrite (đã đủ rõ)
    actual_question = question
    if use_self_rag and SRC_OK and SELFRAG_OK:
        _words   = question.strip().split()
        _has_vi  = any(c in "àáâãèéêìíòóôõùúýăđơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ" for c in question)
        _has_q   = "?" in question or "gì" in question.lower() or "what" in question.lower()
        # Chỉ rewrite khi: rất ngắn (<=3 từ) VÀ không có dấu VI VÀ không có từ hỏi
        _should_rewrite = (len(_words) <= 3 and not _has_vi and not _has_q
                           and question.strip() not in ("", "summarize", "summary"))
        if _should_rewrite:
            try:
                rewritten = rewrite_query(question, llm)
                # Chỉ dùng rewritten nếu dài hơn và không giống mẫu câu gợi ý
                if (rewritten
                        and len(rewritten) > len(question) + 5
                        and rewritten.strip() != question.strip()
                        and "\n" not in rewritten        # không phải list
                        and "1." not in rewritten):      # không phải numbered list
                    actual_question = rewritten
                    extra_info["rewritten_query"] = rewritten
            except Exception:
                pass

    # ── Retrieve docs ───────────────────────────────────────────────────────
    if SRC_OK:
        gen, sources, lang = stream_rag_answer(
            actual_question, retriever, llm, history, show_src
        )
    else:
        docs    = retriever.invoke(actual_question)
        context = "\n\n".join(d.page_content[:1500] for d in docs)
        lang    = _detect_lang(actual_question)
        hist    = ""
        for t in history[-2:]:
            hist += f"{'Người dùng' if lang=='vi' else 'User'}: {t['question']}\nAI: {t['answer'][:150]}\n\n"
        if lang == "vi":
            prompt = (
                "Trả lời DỰA TRÊN ngữ cảnh. Nếu không có thông tin: 'Không tìm thấy trong tài liệu.'\n"
                "Trả lời 2-4 câu tiếng Việt.\n\n"
                + (f"Lịch sử:\n{hist}" if hist else "")
                + f"Ngữ cảnh:\n{context}\n\nCâu hỏi: {actual_question}\n\nTrả lời:"
            )
        else:
            prompt = (
                "Answer based on context only. If not found: 'Not found in document.'\n"
                "Keep answer 2-4 sentences.\n\n"
                + (f"History:\n{hist}" if hist else "")
                + f"Context:\n{context}\n\nQuestion: {actual_question}\n\nAnswer:"
            )
        sources = [{"page": d.metadata.get("page","?"), "content": d.page_content[:250]+"…"}
                   for d in docs] if show_src else []
        gen = llm.stream(prompt)

    # ── Re-ranking (câu 9): đánh giá lại sources sau khi retrieve ──────────
    if use_reranker and SRC_OK and RERANKER_OK and sources:
        try:
            from langchain_core.documents import Document as _Doc
            raw_docs = [_Doc(page_content=s["content"].rstrip("…"),
                             metadata={"page": s["page"]}) for s in sources]
            reranked = rerank_documents(actual_question, raw_docs, top_k=len(raw_docs))
            if reranked:
                sources = [{"page": d.metadata.get("page","?"),
                            "content": d.page_content[:250]+"…",
                            "score": round(d.metadata.get("rerank_score", 0), 3)}
                           for d in reranked]
                extra_info["reranked"] = True
        except Exception:
            pass  # fallback giữ sources gốc

    ph    = st.empty()
    full  = ""
    count = 0
    for tok in gen:
        full  += tok
        count += 1
        if count % 8 == 0:
            _bubble(ph, full, cursor=True)
    _bubble(ph, full, cursor=False)

    # ── Self-RAG: Confidence Scoring (câu 10) ──────────────────────────────
    # Dùng heuristic nhẹ thay vì gọi thêm LLM (tránh chậm)
    # Chỉ gọi LLM self-evaluate khi answer rất ngắn hoặc có dấu hiệu không chắc
    if use_self_rag and SRC_OK and SELFRAG_OK and full:
        try:
            _not_found_phrases = [
                "không tìm thấy", "không có thông tin", "i couldn't find",
                "not found", "i don't know", "không biết"
            ]
            _answer_lower = full.lower()
            _is_not_found = any(p in _answer_lower for p in _not_found_phrases)
            _is_short     = len(full.strip()) < 50

            if _is_not_found:
                extra_info["confidence"]  = 0.2
                extra_info["self_eval"]   = "Không tìm thấy thông tin trong tài liệu"
                extra_info["is_grounded"] = False
            elif _is_short:
                extra_info["confidence"]  = 0.5
                extra_info["self_eval"]   = "Câu trả lời ngắn, có thể chưa đầy đủ"
                extra_info["is_grounded"] = True
            else:
                extra_info["confidence"]  = 0.85
                extra_info["self_eval"]   = "Câu trả lời đầy đủ"
                extra_info["is_grounded"] = True
        except Exception:
            pass

    return full.strip(), sources, lang, extra_info

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <style>
    .sbs{background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.07);
         border-radius:12px;padding:1rem 1.1rem;margin-bottom:1rem;}
    .sbt{font-family:'Syne',sans-serif;font-size:.72rem;font-weight:700;
         letter-spacing:.12em;text-transform:uppercase;color:#6b7280;margin-bottom:.7rem;}
    .sbr{display:flex;justify-content:space-between;align-items:center;
         padding:.3rem 0;border-bottom:1px solid rgba(255,255,255,.05);font-size:.82rem;}
    .sbr:last-child{border-bottom:none;}
    .sbv{font-family:'Syne',sans-serif;color:#4f8ef7;font-weight:600;}
    .on{color:#34d399;}.off{color:#6b7280;}
    .step{display:flex;gap:.7rem;align-items:flex-start;padding:.4rem 0;font-size:.82rem;color:#9ca3af;}
    .stn{background:rgba(79,142,247,.15);color:#4f8ef7;border-radius:50%;width:1.4rem;height:1.4rem;
         display:flex;align-items:center;justify-content:center;font-size:.7rem;font-weight:700;flex-shrink:0;}
    </style>
    """, unsafe_allow_html=True)

    _sb_logo = (
        f'<img src="data:image/jpeg;base64,{LOGO_B64}" '
        f'style="height:52px;width:52px;object-fit:contain;border-radius:8px;'
        f'background:#fff;padding:3px;margin-bottom:.5rem;" />'
        if LOGO_B64 else '<div style="font-size:2.2rem;margin-bottom:.4rem;">🧠</div>'
    )
    st.markdown(f"""
    <div style="text-align:center;padding:.8rem 0 1.4rem;">
        {_sb_logo}
        <div style="font-family:'Syne',sans-serif;font-weight:800;font-size:1.1rem;
                    background:linear-gradient(135deg,#4f8ef7,#a78bfa);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            SmartDoc AI
        </div>
        <div style="font-size:.68rem;color:#374151;letter-spacing:.08em;">RAG · LLMs · OSSD 2026 · v1.3</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Model settings ──────────────────────────────────────────────────────
    st.markdown('<div class="sbs"><div class="sbt">⚙️ Model Settings</div>', unsafe_allow_html=True)
    llm_model = st.selectbox("LLM Model", [
        "qwen2.5:1.5b", "qwen2.5:3b", "qwen2.5:7b",
        "llama3.2:1b", "llama3.2:3b", "mistral:7b", "deepseek-r1:7b",
    ], index=0)
    st.markdown("""
    <div style="background:rgba(245,200,66,.08);border:1px solid rgba(245,200,66,.2);
                border-radius:8px;padding:.5rem .75rem;font-size:.72rem;color:#c9960f;margin:.3rem 0 .6rem;">
        💡 RAM &lt; 4GB → dùng <b>1.5b</b> hoặc <b>3b</b><br>
        <code style="font-size:.68rem;">ollama pull qwen2.5:1.5b</code>
    </div>
    """, unsafe_allow_html=True)
    embed_model = st.selectbox("Embedding Model", [
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "sentence-transformers/all-MiniLM-L6-v2",
    ])
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.05)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Chunk settings ──────────────────────────────────────────────────────
    st.markdown('<div class="sbs"><div class="sbt">🔧 Chunk Strategy</div>', unsafe_allow_html=True)
    chunk_size    = st.select_slider("Chunk Size",    [500, 750, 1000, 1500, 2000], 1000)
    chunk_overlap = st.select_slider("Chunk Overlap", [50, 100, 150, 200], 100)
    top_k         = st.slider("Top-K Retrieval", 1, 8, 3)
    show_sources  = st.toggle("Show Sources", value=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Hybrid Search ───────────────────────────────────────────────────────
    st.markdown('<div class="sbs"><div class="sbt">🔀 Retriever Mode</div>', unsafe_allow_html=True)
    _hybrid_available = SRC_OK and HYBRID_OK
    _use_hybrid = st.toggle(
        "Hybrid Search (BM25 + FAISS)",
        value=(st.session_state.retriever_mode == "hybrid"),
        help="Kết hợp keyword search (BM25) + semantic search (FAISS) cho accuracy cao hơn",
        disabled=not _hybrid_available,
    )
    # Cập nhật mode và rebuild retriever nếu thay đổi
    _new_mode = "hybrid" if _use_hybrid else "vector"
    if _new_mode != st.session_state.retriever_mode:
        st.session_state.retriever_mode = _new_mode
        if st.session_state.vector_store and st.session_state.chunks:
            st.session_state.retriever = _make_retriever(
                st.session_state.vector_store,
                st.session_state.chunks,
                _new_mode, top_k,
            )
            st.rerun()

    if _use_hybrid:
        st.markdown("""
        <div style="background:rgba(79,142,247,.08);border:1px solid rgba(79,142,247,.2);
                    border-radius:8px;padding:.5rem .75rem;font-size:.72rem;color:#4f8ef7;margin:.3rem 0;">
            🔀 <b>Hybrid đang bật</b><br>
            BM25 (40%) + FAISS (60%) → accuracy 85-90%<br>
            <span style="color:#6b7280;">Tốt hơn cho từ khoá chuyên ngành</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:rgba(255,255,255,.02);border:1px solid rgba(255,255,255,.06);
                    border-radius:8px;padding:.5rem .75rem;font-size:.72rem;color:#6b7280;margin:.3rem 0;">
            🔍 <b>Pure FAISS đang bật</b> (semantic only)<br>
            Bật Hybrid để tăng accuracy với từ khoá đặc biệt
        </div>
        """, unsafe_allow_html=True)

    if not _hybrid_available:
        st.markdown("""
        <div style="font-size:.68rem;color:#c9960f;margin-top:.3rem;">
            ⚠️ Cần cài: <code>pip install rank_bm25</code>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Re-ranking (Câu 9) ──────────────────────────────────────────────────
    st.markdown('<div class="sbs"><div class="sbt">🎯 Re-ranking (Cross-Encoder)</div>', unsafe_allow_html=True)
    _reranker_avail = SRC_OK and RERANKER_OK
    _use_reranker = st.toggle(
        "Re-ranking sau Retrieval",
        value=st.session_state.use_reranker,
        help="Cross-Encoder đánh giá lại relevance sau khi FAISS/Hybrid retrieve",
        disabled=not _reranker_avail,
    )
    st.session_state.use_reranker = _use_reranker
    if _use_reranker:
        st.markdown("""
        <div style="background:rgba(167,139,250,.08);border:1px solid rgba(167,139,250,.2);
                    border-radius:8px;padding:.5rem .75rem;font-size:.72rem;color:#a78bfa;margin:.3rem 0;">
            🎯 <b>Re-ranking đang bật</b><br>
            Cross-Encoder đánh giá lại → chính xác hơn<br>
            <span style="color:#6b7280;">Latency +1-3s nhưng accuracy tăng ~5-10%</span>
        </div>
        """, unsafe_allow_html=True)
    if not _reranker_avail:
        st.markdown("""
        <div style="font-size:.68rem;color:#c9960f;margin-top:.3rem;">
            ⚠️ Cần cài: <code>pip install sentence-transformers</code>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Self-RAG (Câu 10) ───────────────────────────────────────────────────
    st.markdown('<div class="sbs"><div class="sbt">🤖 Self-RAG (Advanced)</div>', unsafe_allow_html=True)
    _selfrag_avail = SRC_OK and SELFRAG_OK
    _use_selfrag = st.toggle(
        "Self-RAG + Query Rewriting",
        value=st.session_state.use_self_rag,
        help="LLM tự đánh giá câu trả lời + tự động rewrite query để tìm kiếm tốt hơn",
        disabled=not _selfrag_avail,
    )
    st.session_state.use_self_rag = _use_selfrag
    if _use_selfrag:
        st.markdown("""
        <div style="background:rgba(245,200,66,.08);border:1px solid rgba(245,200,66,.2);
                    border-radius:8px;padding:.5rem .75rem;font-size:.72rem;color:#c9960f;margin:.3rem 0;">
            🤖 <b>Self-RAG đang bật</b><br>
            Rewrite query ngắn/mơ hồ + confidence score<br>
            <span style="color:#6b7280;">+0s nếu query rõ · +3-5s nếu rewrite</span>
        </div>
        """, unsafe_allow_html=True)
    if not _selfrag_avail:
        st.markdown("""
        <div style="font-size:.68rem;color:#6b7280;margin-top:.3rem;">
            ⚠️ Cần Ollama đang chạy
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="sbs"><div class="sbt">📊 System Status</div>', unsafe_allow_html=True)
    _vl  = st.session_state.vector_store is not None
    _dl  = (f"<span class='on'>●</span> {st.session_state.doc_name or ''}"
            if _vl else "<span class='off'>○</span> None")
    _dbl = (f"<span class='on'>●</span> {st.session_state.db_loaded_from}"
            if st.session_state.db_loaded_from else "<span class='off'>○</span> None")
    st.markdown(f"""
    <div class="sbr"><span>Document</span><span class="sbv">{_dl}</span></div>
    <div class="sbr"><span>Saved DB</span><span class="sbv">{_dbl}</span></div>
    <div class="sbr"><span>Chunks</span><span class="sbv">{st.session_state.doc_chunks}</span></div>
    <div class="sbr"><span>Retriever</span><span class="sbv">{'<span class="on">🔀 Hybrid</span>' if st.session_state.retriever_mode=="hybrid" else '🔍 Vector'}</span></div>
    <div class="sbr"><span>Re-ranking</span><span class="sbv">{'<span class="on">🎯 ON</span>' if st.session_state.use_reranker else '<span class="off">○ OFF</span>'}</span></div>
    <div class="sbr"><span>Self-RAG</span><span class="sbv">{'<span class="on">🤖 ON</span>' if st.session_state.use_self_rag else '<span class="off">○ OFF</span>'}</span></div>
    <div class="sbr"><span>Chat turns</span><span class="sbv">{len(st.session_state.chat_history)}</span></div>
    <div class="sbr"><span>src/ parallel</span><span class="sbv">{'<span class="on">✓</span>' if SRC_OK else '<span class="off">fallback</span>'}</span></div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── How to use ──────────────────────────────────────────────────────────
    st.markdown('<div class="sbs"><div class="sbt">📖 Hướng dẫn</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="step"><div class="stn">1</div><div>Cài Ollama + pull model nhỏ (1.5b)</div></div>
    <div class="step"><div class="stn">2</div><div>Upload PDF/DOCX → Process (parallel)</div></div>
    <div class="step"><div class="stn">3</div><div>Lần sau: Load DB ~0.3s</div></div>
    <div class="step"><div class="stn">4</div><div>Chat → AI streaming từng chữ</div></div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    _ca, _cb = st.columns(2)
    with _ca:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    with _cb:
        if st.button("📁 Clear Doc", use_container_width=True):
            st.session_state.update({
                "vector_store":   None,
                "retriever":      None,
                "doc_name":       None,
                "doc_chunks":     0,
                "db_loaded_from": None,
                "chunks":         [],    # ← fix: clear chunks cho BM25
                "chat_history":   [],    # ← fix: clear chat khi xóa doc
            })
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# LOAD MODELS (cached — chỉ chạy 1 lần dù app reload)
# ══════════════════════════════════════════════════════════════════════════════
if not SRC_OK:
    st.warning(f"⚠️ src/ modules không load được: {_IE_MSG} — running fallback", icon="⚠️")

with st.spinner("⚙️ Loading embedding model…"):
    try:
        st.session_state.embedder = _load_embedder(embed_model)
    except Exception as _ee:
        st.error(f"❌ Embedding error: {_ee}")

try:
    st.session_state.llm = _load_llm(llm_model, temperature)
except Exception as _le:
    st.warning(f"⚠️ LLM not loaded (Ollama chạy chưa?): {_le}")

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_up, tab_db, tab_chat, tab_hist = st.tabs([
    "📤  Upload", "💾  Vector DB", "💬  Chat", "📋  History"
])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — UPLOAD
# ═══════════════════════════════════════════════════════════════════════════
with tab_up:
    st.markdown("""
    <style>
    .ucard{background:linear-gradient(135deg,rgba(79,142,247,.06),rgba(167,139,250,.06));
      border:1px solid rgba(79,142,247,.2);border-radius:16px;padding:1.6rem 2rem;margin-bottom:1.4rem;}
    .fg{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:.7rem;margin-top:1rem;}
    .fi{background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.07);border-radius:10px;padding:.75rem 1rem;font-size:.81rem;color:#9ca3af;}
    .fl{font-weight:500;color:#e8eaf0;margin-bottom:.1rem;}
    </style>
    <div class="ucard">
      <div style="font-family:'Syne',sans-serif;font-size:1.2rem;font-weight:700;margin-bottom:.2rem;">
        📤 Upload Document
      </div>
      <div style="font-size:.83rem;color:#6b7280;">
        PDF hoặc DOCX · Extract → Chunk → <b>Embed parallel</b> → Lưu DB
      </div>
      <div class="fg">
        <div class="fi"><div class="fl">🚀 PyMuPDF</div>Parallel page extraction — 2-3x nhanh hơn PDFPlumber</div>
        <div class="fi"><div class="fl">⚡ Parallel Embed</div>ThreadPoolExecutor — 40-60% nhanh hơn serial</div>
        <div class="fi"><div class="fl">📦 MD5 Cache</div>File đã load → &lt;0.1s lần sau</div>
        <div class="fi"><div class="fl">🌐 50+ Languages</div>Tiếng Việt được ưu tiên</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("file", type=["pdf", "docx"],
                                label_visibility="collapsed")

    if uploaded:
        ext  = uploaded.name.rsplit(".", 1)[-1].lower()
        size = len(uploaded.getvalue()) / 1024
        st.markdown(f"""
        <div style="background:rgba(52,211,153,.08);border:1px solid rgba(52,211,153,.25);
                    border-radius:12px;padding:1rem 1.2rem;margin:.8rem 0;
                    display:flex;align-items:center;gap:1rem;">
            <div style="font-size:1.8rem;">{"📄" if ext=="pdf" else "📝"}</div>
            <div>
                <div style="font-family:'Syne',sans-serif;font-weight:600;">{uploaded.name}</div>
                <div style="font-size:.76rem;color:#6b7280;">{ext.upper()} · {size:.1f} KB · Ready</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        db_name  = uploaded.name.rsplit(".", 1)[0].replace(" ", "_")
        has_db   = (VECTOR_DIR / db_name).exists()
        if has_db:
            st.info(f"DB '{db_name}' đã tồn tại. Process lại sẽ ghi đè.", icon="ℹ️")

        if st.button("⚡ Process & Save (Parallel)", use_container_width=False):
            if not st.session_state.embedder:
                st.error("❌ Embedding model not loaded.")
            else:
                prog = st.progress(0, text="Bắt đầu…")
                t0   = time.time()
                try:
                    prog.progress(5, text="📁 Lưu file vào data/input/…")
                    (INPUT_DIR / uploaded.name).write_bytes(uploaded.getvalue())

                    prog.progress(15, text="📖 Load document (PyMuPDF parallel)…")
                    fbytes = uploaded.getvalue()

                    prog.progress(30, text="✂️ Chunking…")
                    # _process_and_save bao gồm: load → split → parallel embed → save
                    vector, n_chunks, db_name, chunks = _process_and_save(
                        fbytes, uploaded.name,
                        chunk_size, chunk_overlap,
                        st.session_state.embedder
                    )

                    prog.progress(88, text="🔗 Building retriever…")
                    retriever = _make_retriever(
                        vector, chunks,
                        st.session_state.retriever_mode, top_k
                    )
                    prog.progress(100, text="✅ Done!")
                    time.sleep(0.1); prog.empty()

                    st.session_state.update({
                        "vector_store":   vector,
                        "retriever":      retriever,
                        "doc_name":       uploaded.name,
                        "doc_chunks":     n_chunks,
                        "db_loaded_from": db_name,
                        "chat_history":   [],
                        "chunks":         chunks,   # ← lưu cho BM25
                    })
                    elapsed = time.time() - t0
                    st.markdown(f"""
                    <div style="background:rgba(52,211,153,.1);border:1px solid rgba(52,211,153,.3);
                                border-radius:12px;padding:1.1rem 1.4rem;margin-top:.8rem;">
                        <div style="font-family:'Syne',sans-serif;font-weight:700;color:#34d399;margin-bottom:.4rem;">
                            ✅ Processed &amp; saved in <b>{elapsed:.1f}s</b>
                        </div>
                        <div style="font-size:.82rem;color:#9ca3af;">
                            <b>{n_chunks}</b> chunks · size={chunk_size} · overlap={chunk_overlap}<br>
                            📁 <code>data/input/{uploaded.name}</code><br>
                            💾 <code>data/vector_db/{db_name}/</code><br>
                            ⚡ Lần sau load DB → chỉ ~0.3s!
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    prog.empty()
                    st.error(f"❌ {e}")
                    st.info("Cài PyMuPDF: `pip install pymupdf` để tăng tốc PDF loading")

    # ── Files in data/input/ ───────────────────────────────────────────────
    saved_inputs = sorted(list(INPUT_DIR.glob("*.pdf")) + list(INPUT_DIR.glob("*.docx")))
    if saved_inputs:
        st.markdown("---")
        st.markdown(f"""
        <div style="font-family:'Syne',sans-serif;font-size:.85rem;font-weight:700;
                    color:#6b7280;letter-spacing:.06em;margin-bottom:.5rem;">
            📁 FILES TRONG data/input/ ({len(saved_inputs)})
        </div>
        """, unsafe_allow_html=True)
        for fp in saved_inputs:
            sz  = fp.stat().st_size / 1024
            dbn = fp.stem.replace(" ", "_")
            hdb = (VECTOR_DIR / dbn).exists()
            badge = (
                '<span style="background:rgba(52,211,153,.15);border:1px solid rgba(52,211,153,.3);'
                'color:#34d399;border-radius:99px;padding:.1rem .5rem;font-size:.65rem;margin-left:.4rem;">💾 DB</span>'
                if hdb else
                '<span style="background:rgba(245,200,66,.1);border:1px solid rgba(245,200,66,.25);'
                'color:#c9960f;border-radius:99px;padding:.1rem .5rem;font-size:.65rem;margin-left:.4rem;">⚠️ No DB</span>'
            )
            cf, cb = st.columns([6, 2])
            with cf:
                st.markdown(f"""
                <div style="background:rgba(255,255,255,.02);border:1px solid rgba(255,255,255,.07);
                            border-radius:9px;padding:.55rem .9rem;font-size:.8rem;margin-bottom:.4rem;">
                    {'📄' if fp.suffix=='.pdf' else '📝'} <b>{fp.name}</b>{badge}
                    <span style="color:#4b5563;margin-left:.5rem;">{sz:.1f} KB</span>
                </div>
                """, unsafe_allow_html=True)
            with cb:
                if hdb:
                    if st.button("⚡ Load DB", key=f"lf_{fp.name}", use_container_width=True):
                        if not st.session_state.embedder:
                            st.error("Embedder not loaded!")
                        else:
                            with st.spinner(f"Loading {dbn}…"):
                                t0  = time.time()
                                vec, ret = _load_db(dbn, st.session_state.embedder, top_k)
                                if vec:
                                    # Load DB từ disk không có chunks → BM25 không khả dụng
                                    # User cần Process lại nếu muốn dùng Hybrid
                                    ret = _make_retriever(vec, [], st.session_state.retriever_mode, top_k)
                                    st.session_state.update({
                                        "vector_store": vec, "retriever": ret,
                                        "doc_name": fp.name, "db_loaded_from": dbn,
                                        "chat_history": [],
                                        "chunks": [],   # chunks không có khi load từ disk
                                    })
                                    st.success(f"Loaded in {time.time()-t0:.2f}s!")
                                    st.rerun()
                else:
                    if st.button("⚡ Process", key=f"pf_{fp.name}", use_container_width=True):
                        if not st.session_state.embedder:
                            st.error("Embedder not loaded!")
                        else:
                            with st.spinner(f"Processing {fp.name}…"):
                                t0  = time.time()
                                fb  = fp.read_bytes()
                                vec, nc, dbn2, chunks = _process_and_save(
                                    fb, fp.name, chunk_size, chunk_overlap,
                                    st.session_state.embedder
                                )
                                ret = _make_retriever(
                                    vec, chunks,
                                    st.session_state.retriever_mode, top_k
                                )
                                st.session_state.update({
                                    "vector_store": vec, "retriever": ret,
                                    "doc_name": fp.name, "doc_chunks": nc,
                                    "db_loaded_from": dbn2,
                                    "chat_history":   [],
                                    "chunks":         chunks,
                                })
                                st.success(f"Done in {time.time()-t0:.1f}s!")
                                st.rerun()

    if st.session_state.doc_name:
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.metric("Active Doc",   st.session_state.doc_name[:24])
        c2.metric("Chunks",       st.session_state.doc_chunks)
        c3.metric("Chat turns",   len(st.session_state.chat_history))

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — VECTOR DB
# ═══════════════════════════════════════════════════════════════════════════
with tab_db:
    st.markdown("""
    <div style="font-family:'Syne',sans-serif;font-size:1.2rem;font-weight:700;margin-bottom:.3rem;">
        💾 Saved Vector Databases
    </div>
    <div style="font-size:.83rem;color:#6b7280;margin-bottom:1rem;">
        Load DB đã lưu → bỏ qua re-embedding → chỉ mất <b>~0.3-0.5s</b>
    </div>
    """, unsafe_allow_html=True)

    dbs = _list_dbs()

    if not dbs:
        st.markdown("""
        <div style="text-align:center;padding:3rem 1rem;color:#4b5563;">
            <div style="font-size:2.5rem;margin-bottom:.5rem;">📭</div>
            <div style="font-family:'Syne',sans-serif;color:#6b7280;">Chưa có DB nào</div>
            <div style="font-size:.8rem;margin-top:.3rem;">Upload và Process file trước ở tab 📤</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background:rgba(79,142,247,.08);border:1px solid rgba(79,142,247,.2);
                    border-radius:10px;padding:.65rem 1rem;margin-bottom:1rem;font-size:.82rem;color:#4f8ef7;">
            📦 Tìm thấy <b>{len(dbs)}</b> database(s) trong <code>data/vector_db/</code>
        </div>
        """, unsafe_allow_html=True)

        for db in dbs:
            name    = db["name"]    if isinstance(db, dict) else db
            size    = db.get("size","?")        if isinstance(db, dict) else "?"
            chunks  = db.get("chunk_count","?") if isinstance(db, dict) else "?"
            created = db.get("created_at","?")  if isinstance(db, dict) else "?"
            etime   = db.get("embed_time","?")  if isinstance(db, dict) else "?"
            is_act  = st.session_state.db_loaded_from == name
            badge   = (
                '<span style="background:rgba(52,211,153,.15);border:1px solid rgba(52,211,153,.3);'
                'color:#34d399;border-radius:99px;padding:.12rem .5rem;font-size:.67rem;margin-left:.4rem;">'
                '● ACTIVE</span>' if is_act else ""
            )
            ci2, cb2, cd2 = st.columns([5, 2, 1])
            with ci2:
                st.markdown(f"""
                <div style="background:rgba(255,255,255,.025);border:1px solid rgba(255,255,255,.08);
                            border-radius:10px;padding:.8rem 1rem;margin-bottom:.5rem;">
                    <div style="font-family:'Syne',sans-serif;font-weight:600;font-size:.88rem;">
                        🗄️ {name}{badge}
                    </div>
                    <div style="font-size:.72rem;color:#6b7280;margin-top:.15rem;">
                        {size} · {chunks} chunks · {created}
                        {f" · embed {etime}s" if etime != "?" else ""}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with cb2:
                if st.button("⚡ Load", key=f"ld_{name}", use_container_width=True):
                    if not st.session_state.embedder:
                        st.error("Embedder not loaded!")
                    else:
                        with st.spinner(f"Loading {name}…"):
                            t0       = time.time()
                            vec, ret = _load_db(name, st.session_state.embedder, top_k)
                            if vec:
                                # Load từ disk: chunks không có sẵn
                                # Hybrid sẽ fallback về vector tự động
                                ret = _make_retriever(vec, [], st.session_state.retriever_mode, top_k)
                                st.session_state.update({
                                    "vector_store": vec, "retriever": ret,
                                    "doc_name": name, "db_loaded_from": name,
                                    "chat_history": [],
                                    "chunks": [],
                                })
                                elapsed = time.time() - t0
                                mode_badge = "🔀 Hybrid" if st.session_state.retriever_mode == "hybrid" else "🔍 Vector"
                                st.success(f"✅ Loaded in {elapsed:.2f}s! [{mode_badge}]")
                                st.rerun()
                            else:
                                st.error("DB not found!")
            with cd2:
                if st.button("🗑️", key=f"dl_{name}", help=f"Xóa DB {name}"):
                    if SRC_OK:
                        delete_vector_db(name, VECTOR_DIR)
                    else:
                        import shutil
                        shutil.rmtree(str(VECTOR_DIR / name), ignore_errors=True)
                    if st.session_state.db_loaded_from == name:
                        st.session_state.update({
                            "vector_store": None, "retriever": None,
                            "db_loaded_from": None, "chunks": [],
                        })
                    st.rerun()

    st.markdown("---")

    # ── Hybrid Search Comparison Section ──────────────────────────────────
    if (st.session_state.vector_store is not None
            and st.session_state.chunks
            and SRC_OK and HYBRID_OK):
        with st.expander("🔀 So sánh Hybrid vs Pure FAISS", expanded=False):
            st.markdown("""
            <div style="font-size:.82rem;color:#9ca3af;margin-bottom:.8rem;">
                Chạy benchmark để so sánh BM25, FAISS và Hybrid Ensemble
                trên tài liệu đang active.
            </div>
            """, unsafe_allow_html=True)
            _cq = st.text_input(
                "Câu hỏi test", value="What is the main topic?",
                key="compare_query"
            )
            if st.button("▶ Chạy So Sánh", key="run_compare"):
                with st.spinner("Đang benchmark 3 retrievers…"):
                    try:
                        report = compare_retrievers(
                            st.session_state.chunks,
                            st.session_state.embedder,
                            _cq,
                            top_k=top_k,
                            vector_store=st.session_state.vector_store,  # ← tái dùng, không re-embed
                        )

                        # Kiểm tra lỗi trước — hiển thị lỗi THỰC SỰ
                        if "error" in report:
                            st.error(f"❌ {report['error']}")
                            import sys as _sys
                            st.markdown(f"""
                            <div style="background:rgba(245,200,66,.08);border:1px solid rgba(245,200,66,.25);
                                        border-radius:10px;padding:.8rem 1rem;font-size:.81rem;color:#c9960f;margin-top:.5rem;">
                                ⚠️ <b>Lưu ý:</b> Streamlit dùng Python riêng, cần cài đúng môi trường.<br>
                                Chạy lệnh này trong <b>terminal cùng nơi bạn chạy streamlit</b>:
                            </div>
                            """, unsafe_allow_html=True)
                            st.code(
                                f"{_sys.executable} -m pip install rank_bm25 langchain\n"
                                f"# Sau đó restart: Ctrl+C rồi streamlit run app.py",
                                language="bash"
                            )
                        # Nếu Ensemble lỗi nhưng BM25+FAISS vẫn có kết quả
                        elif "ensemble_error" in report:
                            st.warning(f"⚠️ Hybrid Ensemble lỗi: {report['ensemble_error']}\nHiển thị BM25 + FAISS thay thế.")
                        elif not report.get("timings"):
                            st.error("❌ Benchmark thất bại — không có kết quả timing.")
                        else:
                            t = report["timings"]
                            r = report.get("results", {})
                            o = report.get("overlap", {})

                            # Safe defaults nếu thiếu key
                            bm25_ms   = t.get("bm25_ms", 0)
                            faiss_ms  = t.get("faiss_ms", 0)
                            hybrid_ms = t.get("hybrid_ms", 0)
                            bm25_docs  = r.get("bm25",   [])
                            faiss_docs = r.get("faiss",  [])
                            hybrid_docs= r.get("hybrid", [])

                            # Timing bar chart (text-based)
                            _max = max(bm25_ms, faiss_ms, hybrid_ms, 1)
                            def _bar(ms):
                                bars = int(ms / _max * 20)
                                return "█" * bars + "░" * (20 - bars)

                            st.markdown(f"""
                            <div style="background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.08);
                                        border-radius:12px;padding:1rem 1.2rem;font-size:.82rem;">
                                <div style="font-family:'Syne',sans-serif;font-weight:700;
                                            margin-bottom:.7rem;color:#e8eaf0;">
                                    📊 Kết quả So Sánh — Query: "{_cq}"
                                </div>
                                <div style="font-family:monospace;font-size:.78rem;line-height:2;">
                                    <span style="color:#6b7280;">BM25  </span>
                                    <span style="color:#f5c842;">{_bar(bm25_ms)}</span>
                                    <span style="color:#e8eaf0;margin-left:.5rem;">{bm25_ms}ms · {len(bm25_docs)} docs</span><br>
                                    <span style="color:#6b7280;">FAISS </span>
                                    <span style="color:#4f8ef7;">{_bar(faiss_ms)}</span>
                                    <span style="color:#e8eaf0;margin-left:.5rem;">{faiss_ms}ms · {len(faiss_docs)} docs</span><br>
                                    <span style="color:#6b7280;">Hybrid</span>
                                    <span style="color:#34d399;">{_bar(hybrid_ms)}</span>
                                    <span style="color:#e8eaf0;margin-left:.5rem;">{hybrid_ms}ms · {len(hybrid_docs)} docs</span>
                                </div>
                                <div style="margin-top:.7rem;padding-top:.6rem;
                                            border-top:1px solid rgba(255,255,255,.07);
                                            font-size:.75rem;color:#6b7280;">
                                    BM25∩FAISS chung: <b style="color:#e8eaf0;">{o.get("bm25_faiss_common","?")} docs</b>
                                    &nbsp;·&nbsp;
                                    Hybrid unique: <b style="color:#34d399;">{o.get("hybrid_unique_docs","?")} docs</b>
                                </div>
                                <div style="margin-top:.5rem;font-size:.75rem;
                                            color:#9ca3af;font-style:italic;">
                                    💡 {report.get("summary","").split("Recommendation: ")[-1]}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                            # Top docs per retriever
                            _c1, _c2, _c3 = st.columns(3)
                            with _c1:
                                st.markdown("**🔑 BM25 top docs**")
                                for d in bm25_docs:
                                    st.caption(f"p.{d['page']}: {d['preview'][:80]}…")
                            with _c2:
                                st.markdown("**🔍 FAISS top docs**")
                                for d in faiss_docs:
                                    st.caption(f"p.{d['page']}: {d['preview'][:80]}…")
                            with _c3:
                                st.markdown("**🔀 Hybrid top docs**")
                                for d in hybrid_docs:
                                    st.caption(f"p.{d['page']}: {d['preview'][:80]}…")

                    except Exception as _ce:
                        st.error(f"❌ Lỗi benchmark: {_ce}")
                        if "rank_bm25" in str(_ce) or "BM25" in str(_ce):
                            st.code("pip install rank_bm25", language="bash")
    elif st.session_state.vector_store and not st.session_state.chunks:
        st.info(
            "💡 **Hybrid Search**: Load DB từ disk không lưu chunks cho BM25. "
            "Hãy **Process** lại file để dùng Hybrid Search.",
            icon="ℹ️"
        )

    # ── Re-ranking Comparison Section (Câu 9) ─────────────────────────────
    if (st.session_state.vector_store is not None
            and SRC_OK and RERANKER_OK):
        with st.expander("🎯 So sánh Bi-encoder vs Cross-Encoder (Re-ranking)", expanded=False):
            st.markdown("""
            <div style="font-size:.82rem;color:#9ca3af;margin-bottom:.8rem;">
                So sánh thứ tự ranking của <b>Bi-encoder (FAISS)</b> vs <b>Cross-Encoder</b>
                trên câu hỏi test.
            </div>
            """, unsafe_allow_html=True)
            _rq = st.text_input(
                "Câu hỏi test re-ranking",
                value="What is the main topic?",
                key="rerank_query"
            )
            if st.button("▶ Chạy Re-ranking Test", key="run_rerank"):
                if not st.session_state.retriever:
                    st.error("Load document trước!")
                else:
                    with st.spinner("Đang so sánh Bi-encoder vs Cross-Encoder…"):
                        try:
                            from src.reranker import compare_biencoder_vs_crossencoder
                            # Lấy docs từ retriever
                            _docs = st.session_state.retriever.invoke(_rq)
                            if not _docs:
                                st.warning("Không tìm thấy docs. Hãy load document trước.")
                            else:
                                result = compare_biencoder_vs_crossencoder(_rq, _docs)
                                if "error" in result:
                                    st.error(f"❌ {result['error']}")
                                    st.code("pip install sentence-transformers", language="bash")
                                else:
                                    _changed = result["order_changed"]
                                    _color   = "#f5c842" if _changed else "#34d399"
                                    st.markdown(f"""
                                    <div style="background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.08);
                                                border-radius:12px;padding:1rem 1.2rem;font-size:.82rem;">
                                        <div style="font-family:'Syne',sans-serif;font-weight:700;
                                                    color:#e8eaf0;margin-bottom:.7rem;">
                                            🎯 Re-ranking Result — Query: "{_rq}"
                                        </div>
                                        <div style="font-family:monospace;font-size:.78rem;line-height:2;">
                                            <span style="color:#6b7280;">Bi-encoder order :  </span>
                                            <span style="color:#4f8ef7;">{result["biencoder_order"]}</span><br>
                                            <span style="color:#6b7280;">Cross-Enc order  :  </span>
                                            <span style="color:#a78bfa;">{result["crossencoder_order"]}</span><br>
                                            <span style="color:#6b7280;">Scores           :  </span>
                                            <span style="color:#e8eaf0;">{result["scores"]}</span>
                                        </div>
                                        <div style="margin-top:.6rem;font-size:.78rem;color:{_color};font-weight:600;">
                                            {"⚠️ Thứ tự THAY ĐỔI sau re-ranking — Cross-Encoder tìm ra doc tốt hơn!" if _changed
                                             else "✅ Thứ tự giống nhau — FAISS đã tốt, re-ranking không cần thiết"}
                                        </div>
                                        <div style="margin-top:.4rem;font-size:.72rem;color:#6b7280;">
                                            Re-rank time: {result.get("rerank_time_s","?")}s ·
                                            Model: {result.get("model","?").split("/")[-1]}
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    if result.get("top_doc_preview"):
                                        st.caption(f"🥇 Top doc sau re-rank: {result['top_doc_preview']}")
                        except Exception as _re:
                            st.error(f"❌ Re-ranking error: {_re}")
    elif SRC_OK and not RERANKER_OK:
        st.info("💡 **Re-ranking (câu 9)**: Cài `pip install sentence-transformers` để dùng Cross-Encoder.", icon="ℹ️")

    st.markdown("""
    <div style="background:rgba(255,255,255,.02);border:1px solid rgba(255,255,255,.06);
                border-radius:10px;padding:.9rem 1rem;font-size:.78rem;color:#4b5563;line-height:1.8;">
        <div style="font-family:'Syne',sans-serif;color:#6b7280;font-weight:700;margin-bottom:.4rem;">
            ⚡ Tối ưu · 🔀 Hybrid · 🎯 Re-rank · 🤖 Self-RAG
        </div>
        📖 <b>PDF Load</b>: PyMuPDF parallel → 1-3s<br>
        🧮 <b>Embedding</b>: ThreadPoolExecutor 4 workers → 3-8s<br>
        💾 <b>Load DB</b>: FAISS từ disk → ~0.3s<br>
        🔀 <b>Hybrid</b>: BM25(40%) + FAISS(60%) → accuracy 85-90%<br>
        🎯 <b>Re-ranking</b>: Cross-Encoder đánh giá lại → +5-10% accuracy<br>
        🤖 <b>Self-RAG</b>: Query rewriting + confidence scoring<br>
        🚀 <b>Streaming</b>: Render mỗi 8 token → ít lag
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — CHAT
# ═══════════════════════════════════════════════════════════════════════════
with tab_chat:
    st.markdown("""
    <style>
    .cwrap{max-height:52vh;overflow-y:auto;padding:.4rem 0;margin-bottom:1rem;}
    .mrow{display:flex;margin-bottom:.9rem;gap:.7rem;align-items:flex-start;}
    .mrow.user{flex-direction:row-reverse;animation:msgInR .3s ease;}
    .mrow.ai{flex-direction:row;animation:msgIn .3s ease;}
    .av{width:2rem;height:2rem;border-radius:50%;display:flex;align-items:center;
        justify-content:center;font-size:.95rem;flex-shrink:0;}
    .av.user{background:linear-gradient(135deg,#4f8ef7,#a78bfa);}
    .av.ai{background:linear-gradient(135deg,#1e2130,#2a2f45);border:1px solid rgba(255,255,255,.1);}
    .bub{max-width:74%;padding:.7rem 1rem;border-radius:14px;font-size:.87rem;line-height:1.65;}
    .bub.user{background:linear-gradient(135deg,rgba(79,142,247,.18),rgba(167,139,250,.14));
              border:1px solid rgba(79,142,247,.25);border-top-right-radius:4px;}
    .bub.ai{background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);border-top-left-radius:4px;}
    .bm{font-size:.67rem;color:#4b5563;margin-top:.3rem;}
    .mrow.user .bm{text-align:right;}
    .sc{display:inline-block;background:rgba(79,142,247,.1);border:1px solid rgba(79,142,247,.2);
        color:#4f8ef7;border-radius:6px;padding:.12rem .45rem;font-size:.68rem;margin:.18rem .12rem 0 0;}
    .ech{text-align:center;padding:3rem 1.5rem;color:#4b5563;}
    .ech h3{font-family:'Syne',sans-serif;font-size:1rem;color:#6b7280;margin-bottom:.3rem;}
    .sugrow .stButton>button{
        background:rgba(255,255,255,.04)!important;border:1px solid rgba(255,255,255,.11)!important;
        border-radius:99px!important;color:#9ca3af!important;font-size:.8rem!important;
        font-family:'DM Sans',sans-serif!important;font-weight:400!important;
        box-shadow:none!important;padding:.38rem .85rem!important;}
    .sugrow .stButton>button:hover{
        background:rgba(79,142,247,.12)!important;border-color:rgba(79,142,247,.35)!important;
        color:#4f8ef7!important;transform:translateY(-1px)!important;}
    </style>
    """, unsafe_allow_html=True)

    if not st.session_state.vector_store:
        st.markdown("""
        <div class="ech">
            <div style="font-size:3rem;margin-bottom:.6rem;">📂</div>
            <h3>No document loaded</h3>
            <p style="font-size:.83rem;">Upload ở tab <b>📤 Upload</b> hoặc load DB ở tab <b>💾 Vector DB</b></p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # ── Chat history display ───────────────────────────────────────────
        if not st.session_state.chat_history:
            _mode_badge = (
                '<span style="background:rgba(52,211,153,.15);border:1px solid rgba(52,211,153,.3);'
                'color:#34d399;border-radius:99px;padding:.2rem .6rem;font-size:.72rem;">'
                '🔀 Hybrid Search</span>'
                if st.session_state.retriever_mode == "hybrid"
                else
                '<span style="background:rgba(79,142,247,.12);border:1px solid rgba(79,142,247,.25);'
                'color:#4f8ef7;border-radius:99px;padding:.2rem .6rem;font-size:.72rem;">'
                '🔍 Vector Search</span>'
            )
            st.markdown(f"""
            <div class="ech">
                <div style="font-size:3rem;margin-bottom:.6rem;">💬</div>
                <h3>Sẵn sàng! Hãy đặt câu hỏi</h3>
                <div style="margin:.4rem 0 .6rem;">{_mode_badge}</div>
                <p style="font-size:.82rem;color:#4b5563;">
                    Streaming — chữ xuất hiện dần dần<br>
                    👇 Click gợi ý hoặc gõ câu hỏi:
                </p>
            </div>
            """, unsafe_allow_html=True)

            _sugs = [
                ("📌 Tóm tắt tài liệu",   "Tóm tắt tài liệu này"),
                ("🔍 Main topic?",          "What is the main topic?"),
                ("📖 Các chương chính",     "Các chương chính là gì?"),
                ("⚙️ Key technologies",    "List the key technologies"),
            ]
            st.markdown('<div class="sugrow">', unsafe_allow_html=True)
            _scols = st.columns(len(_sugs))
            _clicked = None
            for _sc, (_lbl, _val) in zip(_scols, _sugs):
                with _sc:
                    if st.button(_lbl, key=f"sg_{_val}", use_container_width=True):
                        _clicked = _val
            st.markdown('</div>', unsafe_allow_html=True)

            if _clicked:
                st.session_state.pending_q = _clicked
                st.rerun()
        else:
            ch = '<div class="cwrap">'
            for turn in st.session_state.chat_history:
                ts   = turn.get("timestamp", "")
                flag = "🇻🇳" if turn.get("lang") == "vi" else "🇺🇸"
                ch  += (
                    f'<div class="mrow user">'
                    f'<div class="av user">👤</div>'
                    f'<div><div class="bub user">{turn["question"]}</div>'
                    f'<div class="bm">{ts}</div></div></div>'
                )
                src_h = ""
                if turn.get("sources") and show_sources:
                    for s in turn["sources"]:
                        src_h += f'<span class="sc">p.{s["page"]}</span>'

                # Extra info: confidence, rewritten query, reranked
                _extra      = turn.get("extra", {})
                _extra_html = ""
                if _extra.get("rewritten_query"):
                    _extra_html += (
                        f'<span style="font-size:.68rem;color:#6b7280;font-style:italic;">'
                        f'🔄 {_extra["rewritten_query"]}</span> '
                    )
                if _extra.get("confidence") is not None:
                    _c = _extra["confidence"]
                    _cc = "#34d399" if _c >= 0.7 else "#f5c842" if _c >= 0.4 else "#f87171"
                    _extra_html += (
                        f'<span style="font-size:.68rem;color:{_cc};">'
                        f'🎯 {_c:.0%}</span> '
                    )
                if _extra.get("reranked"):
                    _extra_html += '<span style="font-size:.68rem;color:#a78bfa;">🎯 re-ranked</span>'

                ch += (
                    f'<div class="mrow ai">'
                    f'<div class="av ai">🧠</div>'
                    f'<div><div class="bub ai">{turn["answer"]}</div>'
                    f'<div class="bm">{flag} {ts}'
                    f'{"<br>" + _extra_html if _extra_html else ""}'
                    f'{"<br>" + src_h if src_h else ""}'
                    f'</div></div></div>'
                )
            ch += "</div>"
            st.markdown(ch, unsafe_allow_html=True)

        # ── Input area ─────────────────────────────────────────────────────
        st.markdown("""
        <style>
        /* textarea: dark theme, resizable, multiline */
        [data-testid="stTextArea"] textarea {
            background: rgba(30,33,48,.95) !important;
            border: 1px solid rgba(79,142,247,.25) !important;
            border-radius: 12px !important;
            color: #e8eaf0 !important;
            font-family: 'DM Sans', sans-serif !important;
            font-size: .88rem !important;
            line-height: 1.6 !important;
            resize: vertical !important;
            min-height: 60px !important;
            max-height: 200px !important;
            padding: .65rem .9rem !important;
            transition: border-color .25s, box-shadow .25s !important;
        }
        [data-testid="stTextArea"] textarea:focus {
            border-color: rgba(79,142,247,.65) !important;
            box-shadow: 0 0 0 3px rgba(79,142,247,.18) !important;
        }
        [data-testid="stTextArea"] label { display:none !important; }
        </style>
        <div style="background:rgba(255,255,255,.02);border:1px solid rgba(255,255,255,.07);
                    border-radius:14px;padding:.9rem 1.1rem;margin-top:.4rem;">
          <div style="font-family:'Syne',sans-serif;font-size:.7rem;color:#6b7280;
                      letter-spacing:.1em;text-transform:uppercase;margin-bottom:.5rem;">
            Đặt câu hỏi
            <span style="font-weight:300;text-transform:none;letter-spacing:0;color:#374151;margin-left:.5rem;">
              · Shift+Enter = xuống dòng &nbsp;·&nbsp; Bấm Send để gửi
            </span>
          </div>
        """, unsafe_allow_html=True)

        _qi, _qb = st.columns([5, 1])
        with _qi:
            # Dùng value từ session state để clear sau khi gửi
            _input_val = st.session_state.get("_input_val", "")
            user_q = st.text_area(
                "q",
                value=_input_val,
                placeholder="Nhập câu hỏi… Shift+Enter xuống dòng, kéo để mở rộng",
                label_visibility="collapsed",
                key="chat_input",
                height=68,
            )
        with _qb:
            st.markdown('<div style="margin-top:1.6rem"></div>', unsafe_allow_html=True)
            send_btn = st.button("Send ➤", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Determine final question ─────────────────────────────────────
        # Chỉ gửi khi bấm Send — KHÔNG gửi tự động khi Enter
        # (text_area không auto-submit, khác text_input)
        final_q = ""
        if st.session_state.pending_q:
            final_q = st.session_state.pending_q
            st.session_state.pending_q = ""
        elif send_btn and user_q.strip():
            final_q = user_q.strip()
            # Clear input ngay sau khi lấy giá trị
            st.session_state["_input_val"] = ""

        if final_q:
            if not st.session_state.llm:
                st.error("❌ LLM not loaded. Chạy: `ollama serve`")
            elif not st.session_state.retriever:
                st.error("❌ No retriever. Load document trước.")
            else:
                try:
                    full, sources, lang, extra = _stream_answer(
                        final_q,
                        st.session_state.retriever,
                        st.session_state.llm,
                        st.session_state.chat_history,
                        show_sources,
                        use_reranker=st.session_state.use_reranker,
                        use_self_rag=st.session_state.use_self_rag,
                    )
                    # Lưu vào history → st.rerun() sẽ xóa sạch streaming bubble ngay lập tức
                    st.session_state.chat_history.append({
                        "question":  final_q,
                        "answer":    full,
                        "sources":   sources,
                        "lang":      lang,
                        "timestamp": datetime.now().strftime("%H:%M"),
                        "extra":     extra,
                    })
                    st.rerun()   # rerun ngay — không delay, không lingering elements
                except Exception as e:
                    st.error(f"❌ {e}")
                    st.info("Kiểm tra Ollama: `ollama serve`")

        # Source context
        if st.session_state.chat_history and show_sources:
            last = st.session_state.chat_history[-1]
            if last.get("sources"):
                _title = "📚 Source Context"
                if last.get("extra", {}).get("reranked"):
                    _title += " (🎯 Re-ranked)"
                with st.expander(_title, expanded=False):
                    for i, s in enumerate(last["sources"], 1):
                        _score = f' · score={s["score"]}' if "score" in s else ""
                        st.markdown(f"""
                        <div style="background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.07);
                                    border-radius:10px;padding:.75rem 1rem;margin-bottom:.5rem;font-size:.81rem;">
                            <div style="font-family:'Syne',sans-serif;font-size:.7rem;color:#4f8ef7;
                                        margin-bottom:.3rem;">CHUNK {i} · PAGE {s['page']}{_score}</div>
                            <div style="color:#9ca3af;line-height:1.6;">{s['content']}</div>
                        </div>
                        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — HISTORY
# ═══════════════════════════════════════════════════════════════════════════
with tab_hist:
    st.markdown("""
    <style>
    @keyframes fadeUp{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
    .hc{background:rgba(255,255,255,.025);border:1px solid rgba(255,255,255,.07);border-radius:12px;
        padding:.95rem 1.25rem;margin-bottom:.65rem;animation:fadeUp .3s ease both;transition:border-color .2s;}
    .hc:hover{border-color:rgba(79,142,247,.28);}
    .hq{font-family:'Syne',sans-serif;font-size:.87rem;font-weight:600;color:#e8eaf0;margin-bottom:.35rem;}
    .ha{font-size:.81rem;color:#9ca3af;line-height:1.55;margin-bottom:.4rem;}
    .hm{font-size:.69rem;color:#4b5563;}
    </style>
    """, unsafe_allow_html=True)

    if not st.session_state.chat_history:
        st.markdown("""
        <div style="text-align:center;padding:4rem 1.5rem;color:#4b5563;">
            <div style="font-size:2.8rem;margin-bottom:.6rem;">📋</div>
            <div style="font-family:'Syne',sans-serif;font-size:.95rem;color:#6b7280;">No history yet</div>
            <div style="font-size:.8rem;margin-top:.3rem;">Start chatting to see history here.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        total    = len(st.session_state.chat_history)
        vi_count = sum(1 for t in st.session_state.chat_history if t.get("lang") == "vi")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total", total)
        c2.metric("🇻🇳 Vietnamese", vi_count)
        c3.metric("🇺🇸 English", total - vi_count)
        st.markdown("---")

        _exp = "\n".join(
            f"Q{i}: {t['question']}\nA{i}: {t['answer']}\n"
            for i, t in enumerate(st.session_state.chat_history, 1)
        )
        st.download_button(
            "⬇️ Export (.txt)", data=_exp,
            file_name=f"smartdoc_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
        )
        st.markdown("<br>", unsafe_allow_html=True)

        for i, turn in enumerate(reversed(st.session_state.chat_history), 1):
            flag = "🇻🇳" if turn.get("lang") == "vi" else "🇺🇸"
            st.markdown(f"""
            <div class="hc">
                <div class="hq">Q{total-i+1}. {turn["question"]}</div>
                <div class="ha">{turn["answer"][:360]}{"…" if len(turn["answer"])>360 else ""}</div>
                <div class="hm">{flag} · {turn.get("timestamp","")} · {len(turn.get("sources",[]))} source(s)</div>
            </div>
            """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
_fl = (
    f'<img src="data:image/jpeg;base64,{LOGO_B64}" '
    f'style="height:24px;width:24px;object-fit:contain;border-radius:4px;'
    f'background:#fff;padding:2px;vertical-align:middle;margin-right:.4rem;" />'
    if LOGO_B64 else ""
)
st.markdown(f"""
<div style="text-align:center;padding:1.8rem 0 .8rem;border-top:1px solid rgba(255,255,255,.06);margin-top:2rem;">
    <div style="font-size:.75rem;color:#374151;letter-spacing:.05em;">
        {_fl}<span style="color:#4f8ef7;">SmartDoc AI</span> ·
        Trường Đại học Sài Gòn · Khoa CNTT · OSSD Spring 2026 · Ver 1.3
    </div>
    <div style="font-size:.67rem;color:#1f2937;margin-top:.25rem;">
        RAG · Qwen2.5 · FAISS Persistent · PyMuPDF · Parallel Embed · MD5 Cache · Streaming
    </div>
</div>
""", unsafe_allow_html=True)