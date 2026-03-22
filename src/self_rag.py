"""
self_rag.py — SmartDoc AI v1.3
════════════════════════════════
Câu hỏi 10: Advanced RAG với Self-RAG

Yêu cầu:
  ✅ Implement Self-RAG: LLM tự đánh giá câu trả lời
  ✅ Query rewriting: Tự động cải thiện câu hỏi
  ✅ Multi-hop reasoning
  ✅ Confidence scoring

Cơ chế Self-RAG:
  1. Query Rewriting: LLM viết lại câu hỏi rõ ràng hơn để retrieve tốt hơn
  2. Retrieval: Tìm docs với query đã được cải thiện
  3. Generation: LLM tạo câu trả lời từ context
  4. Self-Evaluation: LLM tự đánh giá:
     - Is Relevant? context có liên quan không?
     - Is Grounded? câu trả lời có dựa trên context không?
     - Is Useful? câu trả lời có hữu ích không?
  5. Confidence Scoring: tổng hợp điểm đánh giá
  6. Multi-hop: nếu confidence thấp → rewrite query → retrieve lại
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class SelfRAGResult:
    """Kết quả từ Self-RAG pipeline."""
    question:        str
    rewritten_query: Optional[str]    # câu hỏi đã rewrite (None nếu không cần)
    answer:          str
    sources_text:    str

    # Self-evaluation
    is_relevant:   bool  = True   # context có liên quan không
    is_grounded:   bool  = True   # answer có dựa trên context không
    is_useful:     bool  = True   # answer có hữu ích không
    self_evaluation: str = ""     # nhận xét tự đánh giá
    confidence:    float = 1.0    # 0.0 - 1.0

    # Multi-hop
    hop_count:     int   = 1      # số lần retrieve
    hops:          list  = field(default_factory=list)  # lịch sử các lần hop


# ── Query Rewriting ────────────────────────────────────────────────────────────

def rewrite_query(question: str, llm, lang: str = "auto") -> str:
    """
    Dùng LLM để viết lại câu hỏi rõ ràng hơn cho retrieval.

    Ví dụ:
      Input:  "nó là gì?"
      Output: "RAG (Retrieval-Augmented Generation) là gì?"

    Args:
        question: câu hỏi gốc
        llm:      Ollama LLM instance
        lang:     "vi", "en", hoặc "auto" (tự detect)

    Returns:
        Câu hỏi đã rewrite, hoặc câu gốc nếu rewrite thất bại
    """
    # Auto-detect language
    if lang == "auto":
        from src.llm_chain import detect_language
        lang = detect_language(question)

    if lang == "vi":
        prompt = (
            "Viết lại câu hỏi sau để rõ ràng và đầy đủ hơn cho việc tìm kiếm tài liệu.\n"
            "Chỉ trả về câu hỏi đã viết lại, không giải thích thêm.\n"
            "Nếu câu hỏi đã rõ ràng, trả về nguyên bản.\n\n"
            f"Câu hỏi gốc: {question}\n\n"
            "Câu hỏi đã viết lại:"
        )
    else:
        prompt = (
            "Rewrite the following question to be more specific and clear for document retrieval.\n"
            "Return ONLY the rewritten question, no explanation.\n"
            "If the question is already clear, return it as-is.\n\n"
            f"Original question: {question}\n\n"
            "Rewritten question:"
        )

    try:
        result = ""
        for tok in llm.stream(prompt):
            result += tok
            if len(result) > 300:   # cắt nếu LLM sinh quá dài
                break
        rewritten = result.strip().strip('"').strip("'")

        # Kiểm tra rewrite có ý nghĩa không
        if (rewritten
                and len(rewritten) > 5
                and rewritten.lower() != question.lower()):
            return rewritten
        return question  # fallback về câu gốc

    except Exception:
        return question


# ── Self-Evaluation ────────────────────────────────────────────────────────────

def self_evaluate(
    question: str,
    answer: str,
    sources_text: str,
    llm,
    lang: str = "auto",
) -> dict:
    """
    LLM tự đánh giá câu trả lời theo 3 tiêu chí:
      1. Relevant: context có chứa thông tin liên quan không?
      2. Grounded: answer có được hỗ trợ bởi context không?
      3. Useful: answer có trả lời được câu hỏi không?

    Returns:
        dict với is_relevant, is_grounded, is_useful, confidence, evaluation_text
    """
    if lang == "auto":
        from src.llm_chain import detect_language
        lang = detect_language(question)

    if lang == "vi":
        prompt = (
            "Đánh giá câu trả lời RAG theo các tiêu chí sau.\n"
            "Trả lời theo format JSON chính xác:\n"
            '{"relevant": true/false, "grounded": true/false, "useful": true/false, "reason": "..."}\n\n'
            f"Câu hỏi: {question}\n\n"
            f"Ngữ cảnh tài liệu:\n{sources_text[:800]}\n\n"
            f"Câu trả lời AI:\n{answer[:400]}\n\n"
            "Tiêu chí:\n"
            "- relevant: ngữ cảnh có chứa thông tin liên quan đến câu hỏi không?\n"
            "- grounded: câu trả lời có dựa trên ngữ cảnh không (không bịa đặt)?\n"
            "- useful: câu trả lời có hữu ích, trả lời được câu hỏi không?\n\n"
            "Đánh giá JSON:"
        )
    else:
        prompt = (
            "Evaluate the RAG answer using these criteria.\n"
            "Return ONLY valid JSON:\n"
            '{"relevant": true/false, "grounded": true/false, "useful": true/false, "reason": "..."}\n\n'
            f"Question: {question}\n\n"
            f"Document context:\n{sources_text[:800]}\n\n"
            f"AI Answer:\n{answer[:400]}\n\n"
            "Criteria:\n"
            "- relevant: does the context contain info related to the question?\n"
            "- grounded: is the answer supported by context (not hallucinated)?\n"
            "- useful: does the answer actually answer the question?\n\n"
            "JSON evaluation:"
        )

    try:
        raw = ""
        for tok in llm.stream(prompt):
            raw += tok
            if len(raw) > 400:
                break

        # Parse JSON từ response
        raw = raw.strip()
        # Tìm JSON block
        json_match = re.search(r'\{[^}]+\}', raw, re.DOTALL)
        if json_match:
            import json
            data = json.loads(json_match.group())
            relevant  = bool(data.get("relevant",  True))
            grounded  = bool(data.get("grounded",  True))
            useful    = bool(data.get("useful",    True))
            reason    = str(data.get("reason",     ""))

            # Tính confidence: trung bình 3 tiêu chí
            confidence = (int(relevant) + int(grounded) + int(useful)) / 3.0

            return {
                "is_relevant":    relevant,
                "is_grounded":    grounded,
                "is_useful":      useful,
                "self_evaluation": reason,
                "confidence":     round(confidence, 2),
            }
    except Exception:
        pass

    # Fallback nếu parse lỗi
    return {
        "is_relevant":    True,
        "is_grounded":    True,
        "is_useful":      True,
        "self_evaluation": "Evaluation unavailable",
        "confidence":     0.7,
    }


# ── Multi-hop Reasoning ────────────────────────────────────────────────────────

def multi_hop_rewrite(
    question: str,
    answer: str,
    llm,
    lang: str = "auto",
) -> Optional[str]:
    """
    Nếu câu trả lời chưa đủ, tạo follow-up query để tìm thêm thông tin.
    Dùng cho multi-hop reasoning.

    Returns:
        Follow-up query (str) hoặc None nếu không cần tìm thêm
    """
    if lang == "auto":
        from src.llm_chain import detect_language
        lang = detect_language(question)

    if lang == "vi":
        prompt = (
            f"Câu hỏi: {question}\n"
            f"Câu trả lời hiện tại: {answer[:300]}\n\n"
            "Nếu câu trả lời CHƯA đầy đủ hoặc cần tìm thêm thông tin liên quan, "
            "hãy viết 1 câu hỏi bổ sung ngắn gọn để tìm kiếm thêm.\n"
            "Nếu câu trả lời đã đủ, viết: DONE\n\n"
            "Câu hỏi bổ sung (hoặc DONE):"
        )
    else:
        prompt = (
            f"Question: {question}\n"
            f"Current answer: {answer[:300]}\n\n"
            "If the answer is INCOMPLETE or needs more context, "
            "write a brief follow-up query to search for more info.\n"
            "If the answer is sufficient, write: DONE\n\n"
            "Follow-up query (or DONE):"
        )

    try:
        result = ""
        for tok in llm.stream(prompt):
            result += tok
            if len(result) > 200:
                break
        result = result.strip()
        if result.upper().startswith("DONE") or not result:
            return None
        return result
    except Exception:
        return None


# ── Main Self-RAG Function ─────────────────────────────────────────────────────

def self_rag_answer(
    question: str,
    answer: str,
    sources_text: str,
    llm,
    lang: str = "auto",
) -> SelfRAGResult:
    """
    Self-RAG evaluation pipeline:
    1. Self-evaluate câu trả lời
    2. Tính confidence score
    3. Ghi nhận kết quả

    Args:
        question:     câu hỏi người dùng
        answer:       câu trả lời đã generate
        sources_text: context đã dùng
        llm:          Ollama LLM instance
        lang:         ngôn ngữ

    Returns:
        SelfRAGResult với confidence + self_evaluation
    """
    if lang == "auto":
        try:
            from src.llm_chain import detect_language
            lang = detect_language(question)
        except Exception:
            lang = "en"

    # Self-evaluate
    eval_result = self_evaluate(question, answer, sources_text, llm, lang)

    return SelfRAGResult(
        question        = question,
        rewritten_query = None,      # đã rewrite trước khi gọi hàm này
        answer          = answer,
        sources_text    = sources_text,
        is_relevant     = eval_result["is_relevant"],
        is_grounded     = eval_result["is_grounded"],
        is_useful       = eval_result["is_useful"],
        self_evaluation = eval_result["self_evaluation"],
        confidence      = eval_result["confidence"],
        hop_count       = 1,
    )


def full_self_rag_pipeline(
    question: str,
    retriever,
    llm,
    history: list,
    max_hops: int = 2,
    confidence_threshold: float = 0.5,
    lang: str = "auto",
) -> SelfRAGResult:
    """
    Full Self-RAG pipeline với multi-hop:
    1. Rewrite query
    2. Retrieve + Generate
    3. Self-evaluate
    4. Nếu confidence < threshold → multi-hop → retrieve lại
    5. Trả về kết quả tốt nhất

    Args:
        max_hops:             số lần tối đa re-retrieve (default 2)
        confidence_threshold: ngưỡng confidence để dừng (default 0.5)
    """
    if lang == "auto":
        try:
            from src.llm_chain import detect_language
            lang = detect_language(question)
        except Exception:
            lang = "en"

    hops = []
    best_result = None
    current_query = rewrite_query(question, llm, lang)

    for hop in range(max_hops):
        # Retrieve
        docs    = retriever.invoke(current_query)
        context = "\n\n".join(d.page_content[:1500] for d in docs)

        # Generate
        try:
            from src.llm_chain import build_prompt, detect_language
            _lang  = detect_language(current_query)
            prompt = build_prompt(current_query, context, history, _lang)
            answer = ""
            for tok in llm.stream(prompt):
                answer += tok
            answer = answer.strip()
        except Exception:
            answer = "Generation failed"

        # Self-evaluate
        sources_text = context
        eval_r = self_evaluate(question, answer, sources_text, llm, lang)

        result = SelfRAGResult(
            question        = question,
            rewritten_query = current_query if current_query != question else None,
            answer          = answer,
            sources_text    = sources_text,
            is_relevant     = eval_r["is_relevant"],
            is_grounded     = eval_r["is_grounded"],
            is_useful       = eval_r["is_useful"],
            self_evaluation = eval_r["self_evaluation"],
            confidence      = eval_r["confidence"],
            hop_count       = hop + 1,
            hops            = hops + [{"query": current_query, "confidence": eval_r["confidence"]}],
        )
        hops = result.hops

        # Cập nhật best result
        if best_result is None or result.confidence > best_result.confidence:
            best_result = result

        # Đủ confidence → dừng
        if result.confidence >= confidence_threshold:
            break

        # Multi-hop: tạo follow-up query
        follow_up = multi_hop_rewrite(question, answer, llm, lang)
        if follow_up is None:
            break
        current_query = follow_up

    return best_result or SelfRAGResult(
        question=question, rewritten_query=None,
        answer="", sources_text="", confidence=0.0,
    )