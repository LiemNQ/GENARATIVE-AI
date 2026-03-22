"""
llm_chain.py — SmartDoc AI v1.3
══════════════════════════════════
Target: Answer Generation 3-8s | Accuracy 80-85%

Tối ưu:
  1. detect_language: ký tự dấu + từ khóa không dấu → không sai ngôn ngữ
  2. Prompt ngắn gọn → ít token → LLM trả lời nhanh hơn
  3. History: đúng N lượt (fix lỗi slice)
  4. Singleton LLM cache → không reload model mỗi lần
  5. OMP_NUM_THREADS=4 để tăng tốc CPU inference
"""

import os
from typing import Generator, List, Tuple, Optional

# Tăng tốc CPU inference cho Ollama
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

# ── Vietnamese char set ────────────────────────────────────────────────────────
_VI_CHARS = frozenset(
    "àáâãèéêìíòóôõùúýăđơư"
    "ạảấầẩẫậắằẳẵặẹẻẽếềểễệ"
    "ỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ"
    "ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĂĐƠƯ"
)

# Từ khóa tiếng Việt không dấu phổ biến
_VI_KW = frozenset({
    "tom tat", "tai lieu", "chuong", "muc luc", "noi dung",
    "la gi", "ket qua", "giai thich", "liet ke", "tim kiem",
    "huong dan", "mo ta", "bao nhieu", "nhu the nao", "vi sao",
    "yeu cau", "muc tieu", "tinh nang", "diem chinh", "phan tich",
    "so sanh", "danh sach", "cong nghe", "kien truc", "thiet ke",
})

# ── LLM cache ──────────────────────────────────────────────────────────────────
_llm_cache: dict = {}


def get_llm(
    model_name: str = "qwen2.5:1.5b",
    temperature: float = 0.7,
    top_p: float = 0.9,
    repeat_penalty: float = 1.1,
):
    """Cached Ollama LLM. Ưu tiên langchain-ollama (mới hơn)."""
    key = (model_name, round(temperature, 2))
    if key in _llm_cache:
        return _llm_cache[key]
    try:
        from langchain_ollama import OllamaLLM
        llm = OllamaLLM(
            model=model_name,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
        )
    except ImportError:
        from langchain_community.llms import Ollama
        llm = Ollama(
            model=model_name,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
        )
    _llm_cache[key] = llm
    return llm


def clear_llm_cache():
    _llm_cache.clear()


# ── Language detection ─────────────────────────────────────────────────────────

def detect_language(text: str) -> str:
    """
    Detect VI vs EN.
    Priority:
      1. Có ký tự dấu tiếng Việt → chắc chắn VI
      2. Có từ khóa VI không dấu → VI
      3. Không có → EN
    """
    # 1 ký tự dấu là đủ
    for c in text:
        if c in _VI_CHARS:
            return "vi"
    # Từ khóa không dấu
    tl = text.lower().strip()
    for kw in _VI_KW:
        if kw in tl:
            return "vi"
    return "en"


# ── Prompt builder ─────────────────────────────────────────────────────────────

def build_prompt(
    question: str,
    context: str,
    history: List[dict],
    lang: str,
    n_history: int = 2,
) -> str:
    """
    Prompt ngắn gọn, rõ ràng cho Qwen2.5.
    n_history: số Q&A giữ lại (default 2).
    History answer bị cắt còn 200 ký tự → tiết kiệm token → nhanh hơn.
    """
    recent = history[-n_history:] if history else []
    hist   = ""
    if recent:
        lines = []
        for t in recent:
            u   = "Người dùng" if lang == "vi" else "User"
            ans = t['answer'][:200]
            # Bỏ qua history entry nếu answer trông như list gợi ý (LLM bị nhiễu)
            _looks_bad = (
                ans.strip().startswith("1.") or
                ans.count("\n1.") > 0 or
                ans.count('"') >= 4    # nhiều quoted strings = list gợi ý
            )
            if _looks_bad:
                continue
            lines.append(f"{u}: {t['question']}")
            lines.append(f"AI: {ans}")
        if lines:
            hist = "\n".join(lines) + "\n\n"

    if lang == "vi":
        return (
            "Bạn là trợ lý AI. Hãy trả lời câu hỏi DỰA TRÊN ngữ cảnh bên dưới.\n"
            "Trả lời TRỰC TIẾP bằng tiếng Việt, 2-4 câu. Dùng thông tin có trong ngữ cảnh.\n\n"
            + (f"Lịch sử hội thoại:\n{hist}" if hist else "")
            + f"=== NGỮ CẢNH TÀI LIỆU ===\n{context}\n=== HẾT NGỮ CẢNH ===\n\n"
            f"Câu hỏi: {question}\n\nTrả lời:"
        )
    return (
        "You are an AI assistant. Answer the question using the context below.\n"
        "Answer DIRECTLY in 2-4 sentences using information from the context.\n\n"
        + (f"Conversation history:\n{hist}" if hist else "")
        + f"=== DOCUMENT CONTEXT ===\n{context}\n=== END CONTEXT ===\n\n"
        f"Question: {question}\n\nAnswer:"
    )


# ── Main streaming function ────────────────────────────────────────────────────

def stream_rag_answer(
    question: str,
    retriever,
    llm,
    history: List[dict],
    show_sources: bool = True,
    n_history: int = 2,
) -> Tuple[Generator, List[dict], str]:
    """
    1. Retrieve docs (0.05-0.1s)
    2. Detect language
    3. Build prompt (tối ưu token)
    4. Return streaming generator

    Returns: (token_generator, sources, lang)
    """
    docs    = retriever.invoke(question)

    # Với câu hỏi tóm tắt → cần nhiều chunks hơn để có đủ context
    _summary_kw = ["tóm tắt", "tom tat", "summarize", "summary", "tổng quan",
                   "overview", "giới thiệu", "mô tả tài liệu", "nội dung chính",
                   "tóm lại", "toàn bộ", "khái quát"]
    _is_summary = any(kw in question.lower() for kw in _summary_kw)
    if _is_summary:
        try:
            # Thử lấy thêm chunks qua search_kwargs mở rộng
            _vs = getattr(retriever, "vectorstore",
                  getattr(retriever, "_vectorstore", None))
            if _vs is not None:
                _extra = _vs.similarity_search(question, k=6)
                _seen = {d.page_content[:50] for d in docs}
                for d in _extra:
                    if d.page_content[:50] not in _seen:
                        docs.append(d)
                        _seen.add(d.page_content[:50])
        except Exception:
            pass  # fallback: dùng docs gốc

    # Cắt mỗi chunk còn 1500 ký tự → giảm token prompt → LLM nhanh hơn
    context = "\n\n".join(d.page_content[:1500] for d in docs)
    lang    = detect_language(question)

    # Prompt đặc biệt cho câu hỏi tóm tắt
    if _is_summary:
        if lang == "vi":
            prompt = (
                "Bạn là trợ lý AI. Dưới đây là nội dung tài liệu.\n"
                "Hãy TÓM TẮT trực tiếp nội dung này bằng tiếng Việt, 6-10 câu.\n"
                "Bao gồm: chủ đề chính, mục tiêu, công nghệ/phương pháp sử dụng.\n\n"
                f"=== NỘI DUNG TÀI LIỆU ===\n{context}\n=== HẾT ===\n\n"
                "Tóm tắt:"
            )
        else:
            prompt = (
                "You are an AI assistant. Below is the document content.\n"
                "Provide a DIRECT summary in 6-10 sentences.\n"
                "Include: main topic, objectives, technologies/methods used.\n\n"
                f"=== DOCUMENT CONTENT ===\n{context}\n=== END ===\n\n"
                "Summary:"
            )
    else:
        prompt = build_prompt(question, context, history, lang, n_history)

    sources = []
    if show_sources:
        for d in docs:
            sources.append({
                "page":    d.metadata.get("page", "?"),
                "content": d.page_content[:250] + "…",
            })

    return llm.stream(prompt), sources, lang


# ── Model reference ────────────────────────────────────────────────────────────
OLLAMA_MODELS = {
    "qwen2.5:1.5b":   {"ram": "~1 GB",   "speed": "2-4s"},
    "qwen2.5:3b":     {"ram": "~2 GB",   "speed": "3-6s"},
    "qwen2.5:7b":     {"ram": "~4.3 GB", "speed": "5-10s"},
    "llama3.2:1b":    {"ram": "~1 GB",   "speed": "2-4s"},
    "llama3.2:3b":    {"ram": "~2 GB",   "speed": "3-6s"},
    "mistral:7b":     {"ram": "~4.1 GB", "speed": "5-10s"},
    "deepseek-r1:7b": {"ram": "~4.7 GB", "speed": "6-12s"},
}