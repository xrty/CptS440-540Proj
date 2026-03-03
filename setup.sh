#!/usr/bin/env bash
# =============================================================================
# setup.sh — One-command setup for the RAG Academic Paper QA System
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
#
# What this script does:
#   1. Creates a Python virtual environment (venv/) if one doesn't exist,
#      or reuses the currently active environment (conda / system Python)
#   2. Installs all dependencies from requirements.txt
#   3. Creates .env from .env.example if it doesn't exist yet
#   4. Creates required empty directories (data/papers/, indexes/)
#   5. Downloads 20 papers from ArXiv and builds the search indexes
#
# After running this script, start the QA system with:
#   python src/pipeline.py
# (activate your environment first if using venv: source venv/bin/activate)
# =============================================================================

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${GREEN}[setup]${NC} $*"; }
warn()  { echo -e "${YELLOW}[warn]${NC}  $*"; }
error() { echo -e "${RED}[error]${NC} $*"; exit 1; }

# ── Step 0: Find Python ───────────────────────────────────────────────────────
info "Checking Python version..."
PYTHON=$(command -v python3 2>/dev/null || command -v python 2>/dev/null || true)
[ -z "$PYTHON" ] && error "Python not found. Install Python 3.10+."

VERSION=$($PYTHON --version 2>&1 | awk '{print $2}')
MAJOR=$(echo "$VERSION" | cut -d. -f1)
MINOR=$(echo "$VERSION" | cut -d. -f2)
if [ "$MAJOR" -lt 3 ] || { [ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 10 ]; }; then
    error "Python 3.10+ required. Found: $VERSION"
fi
info "Using Python $VERSION at $PYTHON"

# ── Step 1: Virtual environment (skip if already inside conda or another venv) ─
IN_CONDA="${CONDA_DEFAULT_ENV:-}"
IN_VENV="${VIRTUAL_ENV:-}"

if [ -n "$IN_CONDA" ]; then
    info "Conda environment detected ($IN_CONDA) — skipping venv creation."
    PIP="pip"
elif [ -n "$IN_VENV" ]; then
    info "Virtual environment already active — skipping venv creation."
    PIP="pip"
elif [ -d "venv" ]; then
    info "Found existing venv/ — activating."
    # shellcheck disable=SC1091
    source venv/bin/activate
    PIP="pip"
else
    info "Creating virtual environment (venv/)..."
    $PYTHON -m venv venv
    # shellcheck disable=SC1091
    source venv/bin/activate
    PIP="pip"
    info "Virtual environment created and activated."
fi

# ── Step 2: Install dependencies ──────────────────────────────────────────────
info "Installing dependencies from requirements.txt..."
$PIP install --quiet --upgrade pip
$PIP install --quiet -r requirements.txt
info "Dependencies installed."

# ── Step 3: Set up .env ───────────────────────────────────────────────────────
if [ ! -f ".env" ]; then
    cp .env.example .env
    warn ".env created from .env.example"
    warn ">>> Open .env and set your GROQ_API_KEY before running the pipeline."
    warn "    Get a free key at: https://console.groq.com"
else
    info ".env already exists — skipping."
fi

# ── Step 4: Create required directories ───────────────────────────────────────
mkdir -p data/papers indexes/faiss_index
info "Directories ready."

# ── Step 5: Download papers and build indexes ─────────────────────────────────
info "Downloading 20 papers from ArXiv (cs.AI + cs.CL)..."
python src/ingest.py --arxiv --max 20

info "Building BM25 + FAISS indexes..."
python src/build_index.py

# ── Done ───────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}  Setup complete!${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
if [ -n "$IN_CONDA" ]; then
    echo "  Your conda environment ($IN_CONDA) is already active."
elif [ -d "venv" ] && [ -z "$IN_VENV" ]; then
    echo "  Activate the virtual environment first:"
    echo "    source venv/bin/activate"
    echo ""
fi
echo "  Start an interactive QA session:"
echo "    python src/pipeline.py"
echo ""
echo "  Ask a single question:"
echo "    python src/pipeline.py --question \"What is attention mechanism?\""
echo ""
echo "  Change retrieval mode:"
echo "    python src/pipeline.py --mode bm25"
echo "    python src/pipeline.py --mode dense"
echo "    python src/pipeline.py --mode hybrid    # default"
echo ""
if grep -q "your_groq_api_key_here" .env 2>/dev/null; then
    warn "Don't forget to set GROQ_API_KEY in .env for LLM answer generation!"
fi
