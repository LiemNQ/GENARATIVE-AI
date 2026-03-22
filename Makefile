# ══════════════════════════════════════════════════════════════════
#  SmartDoc AI v1.3 — Makefile
#  Môn: Open Source Software Development · Spring 2026
#  Trường Đại học Sài Gòn
# ══════════════════════════════════════════════════════════════════

.PHONY: help install run test clean setup check

GREEN  := \033[0;32m
YELLOW := \033[1;33m
CYAN   := \033[0;36m
RESET  := \033[0m

# ── Default ───────────────────────────────────────────────────────
help:
	@echo ""
	@echo "$(CYAN)╔══════════════════════════════════════════╗$(RESET)"
	@echo "$(CYAN)║       SmartDoc AI v1.3 — Makefile       ║$(RESET)"
	@echo "$(CYAN)╚══════════════════════════════════════════╝$(RESET)"
	@echo ""
	@echo "$(YELLOW)Setup:$(RESET)"
	@echo "  make setup        Tạo venv + cài tất cả dependencies"
	@echo "  make install      Cài dependencies vào môi trường hiện tại"
	@echo ""
	@echo "$(YELLOW)Run:$(RESET)"
	@echo "  make run          Chạy app trên http://localhost:8501"
	@echo "  make run-port P=8888   Chạy trên port tùy chọn"
	@echo ""
	@echo "$(YELLOW)Test:$(RESET)"
	@echo "  make test         Chạy toàn bộ 20 test cases"
	@echo "  make test-v       Chạy test verbose (pytest)"
	@echo "  make test-quick   Chạy test nhanh (không cần Ollama)"
	@echo ""
	@echo "$(YELLOW)Clean:$(RESET)"
	@echo "  make clean        Xóa cache, __pycache__"
	@echo "  make clean-db     Xóa vector databases"
	@echo "  make clean-all    Clean + xóa venv"
	@echo ""
	@echo "$(YELLOW)Check:$(RESET)"
	@echo "  make check        Kiểm tra môi trường"
	@echo ""

# ── Setup ─────────────────────────────────────────────────────────
setup:
	@echo "$(CYAN)>> Tạo virtual environment...$(RESET)"
	python -m venv venv
	@echo "$(CYAN)>> Upgrade pip...$(RESET)"
	venv/Scripts/pip install --upgrade pip 2>/dev/null || venv/bin/pip install --upgrade pip
	@echo "$(CYAN)>> Cài dependencies...$(RESET)"
	venv/Scripts/pip install -r requirements.txt 2>/dev/null || venv/bin/pip install -r requirements.txt
	@echo ""
	@echo "$(GREEN)Setup xong! Kích hoạt venv:$(RESET)"
	@echo "  Windows : venv\\Scripts\\activate"
	@echo "  Mac/Linux: source venv/bin/activate"
	@echo "  Sau đó  : make run"

install:
	pip install --upgrade pip
	pip install -r requirements.txt
	@echo "$(GREEN)Cài xong!$(RESET)"

# ── Run ───────────────────────────────────────────────────────────
run:
	@echo "$(CYAN)>> Khởi động SmartDoc AI...$(RESET)"
	streamlit run app.py --server.port=8501

run-port:
	streamlit run app.py --server.port=$(P)

# ── Test ──────────────────────────────────────────────────────────
test:
	@echo "$(CYAN)>> Chạy test suite...$(RESET)"
	python tests/test_rag_logic.py

test-v:
	pytest tests/ -v --tb=short

test-quick:
	pytest tests/ -v -k "not performance and not self_rag" --tb=short

# ── Clean ─────────────────────────────────────────────────────────
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
	find . -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null; true
	rm -f data/loader_cache/*.pkl 2>/dev/null; true
	@echo "$(GREEN)Clean xong!$(RESET)"

clean-db:
	rm -rf data/vector_db/*/
	@echo "$(GREEN)Vector DBs đã xóa!$(RESET)"

clean-all: clean
	rm -rf venv/
	@echo "$(GREEN)Clean all xong!$(RESET)"

# ── Check ─────────────────────────────────────────────────────────
check:
	@echo "$(CYAN)>> Kiểm tra môi trường...$(RESET)"
	@echo ""
	@python --version
	@pip show streamlit 2>/dev/null | grep Version || echo "streamlit: NOT INSTALLED"
	@pip show langchain 2>/dev/null | grep Version || echo "langchain: NOT INSTALLED"
	@pip show faiss-cpu 2>/dev/null | grep Version || echo "faiss-cpu: NOT INSTALLED"
	@pip show sentence-transformers 2>/dev/null | grep Version || echo "sentence-transformers: NOT INSTALLED"
	@pip show pymupdf 2>/dev/null | grep Version || echo "pymupdf: NOT INSTALLED"
	@pip show rank-bm25 2>/dev/null | grep Version || echo "rank-bm25: NOT INSTALLED"
	@ollama --version 2>/dev/null || echo "Ollama: NOT INSTALLED"