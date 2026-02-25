#!/bin/bash
# Deploy COD + ORD pipelines on Blackwell GPU VM
# Run from /opt/chirality on the VM
#
# Usage:
#   ssh rtx6000
#   cd /opt/chirality
#   bash deploy/run_pipelines.sh
#
# Or deploy individually:
#   bash deploy/run_pipelines.sh cod
#   bash deploy/run_pipelines.sh ord

set -e

CHIRALITY_DIR="${CHIRALITY_DIR:-/opt/chirality}"
cd "$CHIRALITY_DIR"

# Ensure ORD dependency is installed
pip install ord-schema>=0.3 2>/dev/null || echo "ord-schema install skipped (may already be installed)"

# Create work directories
mkdir -p cod_work ord_work

run_cod() {
    echo "Starting COD pipeline in tmux session 'cod'..."
    tmux kill-session -t cod 2>/dev/null || true
    tmux new-session -d -s cod \
        "cd $CHIRALITY_DIR && python scripts/cod_pipeline.py \
            --db cod_work/cod_screen.db \
            --work-dir cod_work \
            --workers 8 \
            --resume \
            --limit-per-sg 500 \
            2>&1 | tee cod_work/pipeline.log"
    echo "COD pipeline running. Monitor with: tmux attach -t cod"
}

run_ord() {
    echo "Starting ORD pipeline in tmux session 'ord'..."
    tmux kill-session -t ord 2>/dev/null || true

    # Check if ORD data exists, otherwise prompt
    if [ -d "ord_data" ]; then
        ORD_SRC="--data-dir ord_data"
    elif [ -f "ord_work/reactions.json" ]; then
        ORD_SRC="--resume"
    else
        echo "WARNING: No ORD data found."
        echo "  Option 1: Clone ORD data repo to ord_data/"
        echo "    git clone --depth 1 https://github.com/open-reaction-database/ord-data.git ord_data"
        echo "  Option 2: Provide pre-extracted JSON:"
        echo "    python scripts/ord_pipeline.py --json-file reactions.json --db ord_work/ord_screen.db"
        echo ""
        echo "Skipping ORD pipeline start."
        return 1
    fi

    tmux new-session -d -s ord \
        "cd $CHIRALITY_DIR && python scripts/ord_pipeline.py \
            $ORD_SRC \
            --db ord_work/ord_screen.db \
            --work-dir ord_work \
            --workers 8 \
            --train-model \
            2>&1 | tee ord_work/pipeline.log"
    echo "ORD pipeline running. Monitor with: tmux attach -t ord"
}

case "${1:-all}" in
    cod) run_cod ;;
    ord) run_ord ;;
    all)
        run_cod
        run_ord
        echo ""
        echo "Both pipelines launched. Monitor:"
        echo "  tmux attach -t cod"
        echo "  tmux attach -t ord"
        ;;
    *)
        echo "Usage: $0 [cod|ord|all]"
        exit 1
        ;;
esac
