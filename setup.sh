#!/bin/bash
# ============================================================
#  VidLens — One-Command Setup Script (Linux)
#  Run this: bash setup.sh
# ============================================================

set -e  # Stop on any error

# Colors for pretty output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

print_step() { echo -e "\n${BLUE}${BOLD}==>${NC} ${BOLD}$1${NC}"; }
print_ok()   { echo -e "${GREEN}✅ $1${NC}"; }
print_warn() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_err()  { echo -e "${RED}❌ $1${NC}"; exit 1; }

echo -e "${BOLD}"
echo "  ██╗   ██╗██╗██████╗ ██╗     ███████╗███╗   ██╗███████╗"
echo "  ██║   ██║██║██╔══██╗██║     ██╔════╝████╗  ██║██╔════╝"
echo "  ██║   ██║██║██║  ██║██║     █████╗  ██╔██╗ ██║███████╗"
echo "  ╚██╗ ██╔╝██║██║  ██║██║     ██╔══╝  ██║╚██╗██║╚════██║"
echo "   ╚████╔╝ ██║██████╔╝███████╗███████╗██║ ╚████║███████║"
echo "    ╚═══╝  ╚═╝╚═════╝ ╚══════╝╚══════╝╚═╝  ╚═══╝╚══════╝"
echo -e "${NC}"
echo -e "  ${BOLD}Open Source Video Intelligence Toolkit${NC}"
echo -e "  Setting up your environment...\n"

# ------------------------------------------------------------------
# Step 1: Check Python
# ------------------------------------------------------------------
print_step "Checking Python version..."

if ! command -v python3 &>/dev/null; then
    print_warn "Python3 not found. Installing..."
    sudo apt-get update -qq && sudo apt-get install -y python3 python3-pip python3-venv
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
REQUIRED="3.9"

if python3 -c "import sys; exit(0 if sys.version_info >= (3,9) else 1)"; then
    print_ok "Python $PYTHON_VERSION found"
else
    print_err "Python 3.9+ required, but found $PYTHON_VERSION. Please upgrade Python."
fi

# ------------------------------------------------------------------
# Step 2: Install system dependencies
# ------------------------------------------------------------------
print_step "Installing system dependencies (FFmpeg, OpenCV libs)..."

if command -v apt-get &>/dev/null; then
    sudo apt-get update -qq
    sudo apt-get install -y \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        python3-pip \
        python3-venv \
        2>/dev/null
    print_ok "System dependencies installed"
else
    print_warn "apt-get not found. If you hit errors, manually install: ffmpeg libgl1"
fi

# ------------------------------------------------------------------
# Step 3: Create virtual environment
# ------------------------------------------------------------------
print_step "Creating Python virtual environment (.venv)..."

if [ -d ".venv" ]; then
    print_warn ".venv already exists — skipping creation"
else
    python3 -m venv .venv
    print_ok "Virtual environment created at .venv/"
fi

# Activate it
source .venv/bin/activate
print_ok "Virtual environment activated"

# ------------------------------------------------------------------
# Step 4: Upgrade pip
# ------------------------------------------------------------------
print_step "Upgrading pip..."
pip install --upgrade pip --quiet
print_ok "pip upgraded"

# ------------------------------------------------------------------
# Step 5: Detect GPU
# ------------------------------------------------------------------
print_step "Detecting GPU..."

HAS_GPU=false
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    if [ -n "$GPU_NAME" ]; then
        HAS_GPU=true
        print_ok "NVIDIA GPU found: $GPU_NAME"
        echo -e "   ${GREEN}Models will run on GPU (much faster!)${NC}"
    fi
fi

if [ "$HAS_GPU" = false ]; then
    print_warn "No NVIDIA GPU detected — models will run on CPU (slower but works fine)"
fi

# ------------------------------------------------------------------
# Step 6: Install PyTorch
# ------------------------------------------------------------------
print_step "Installing PyTorch..."

if [ "$HAS_GPU" = true ]; then
    echo "   Installing PyTorch with CUDA support..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --quiet
else
    echo "   Installing PyTorch (CPU only)..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet
fi
print_ok "PyTorch installed"

# ------------------------------------------------------------------
# Step 7: Install VidLens
# ------------------------------------------------------------------
print_step "Installing VidLens and all dependencies..."

pip install -e ".[yolo,ui]" --quiet

# Install CLIP separately (from GitHub)
echo "   Installing CLIP..."
pip install git+https://github.com/openai/CLIP.git --quiet 2>/dev/null || \
    pip install openai-clip --quiet || \
    print_warn "CLIP install failed — scene classification won't work. Run: pip install openai-clip"

print_ok "VidLens installed"

# ------------------------------------------------------------------
# Step 8: Pre-download model weights
# ------------------------------------------------------------------
print_step "Pre-downloading model weights (this may take a minute)..."

python3 -c "
from ultralytics import YOLO
print('  Downloading YOLOv8n (object detection)...')
YOLO('yolov8n.pt')
print('  Downloading YOLOv8n-pose (pose estimation)...')
YOLO('yolov8n-pose.pt')
print('  Done!')
" 2>/dev/null && print_ok "Model weights downloaded" || print_warn "Weight download failed — they'll auto-download on first use"

# ------------------------------------------------------------------
# Step 9: Create activation helper
# ------------------------------------------------------------------
print_step "Creating helper scripts..."

cat > activate.sh << 'EOF'
#!/bin/bash
# Run this every time you open a new terminal: source activate.sh
source .venv/bin/activate
echo "✅ VidLens environment activated. Try: vidlens --help"
EOF
chmod +x activate.sh

cat > run.sh << 'EOF'
#!/bin/bash
# Quick run helper: ./run.sh myvideo.mp4 objects pose
source .venv/bin/activate
if [ -z "$1" ]; then
    echo "Usage: ./run.sh <video.mp4> [lens1] [lens2] ..."
    echo "Example: ./run.sh myvideo.mp4 objects"
    echo "Example: ./run.sh myvideo.mp4 objects pose faces"
    exit 1
fi
VIDEO="$1"
shift
LENSES=""
for lens in "$@"; do
    LENSES="$LENSES --lens $lens"
done
if [ -z "$LENSES" ]; then
    LENSES="--lens objects"
fi
vidlens analyze "$VIDEO" $LENSES
EOF
chmod +x run.sh

print_ok "Helper scripts created"

# ------------------------------------------------------------------
# Done!
# ------------------------------------------------------------------
echo ""
echo -e "${GREEN}${BOLD}============================================${NC}"
echo -e "${GREEN}${BOLD}  ✅ VidLens is ready!${NC}"
echo -e "${GREEN}${BOLD}============================================${NC}"
echo ""
echo -e "${BOLD}Quick start:${NC}"
echo ""
echo -e "  ${YELLOW}# Activate environment (do this each new terminal):${NC}"
echo -e "  source activate.sh"
echo ""
echo -e "  ${YELLOW}# Analyze a video:${NC}"
echo -e "  vidlens analyze myvideo.mp4 --lens objects"
echo ""
echo -e "  ${YELLOW}# Run multiple analyses:${NC}"
echo -e "  vidlens analyze myvideo.mp4 --lens objects --lens pose"
echo ""
echo -e "  ${YELLOW}# Blur faces for privacy:${NC}"
echo -e "  vidlens anonymize myvideo.mp4 --mode blur"
echo ""
echo -e "  ${YELLOW}# Launch web UI (no command line needed!):${NC}"
echo -e "  vidlens ui"
echo ""
echo -e "  ${YELLOW}# See all options:${NC}"
echo -e "  vidlens --help"
echo ""
echo -e "${BLUE}Output files are saved to: ./output/${NC}"
echo ""
