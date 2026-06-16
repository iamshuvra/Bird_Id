#!/usr/bin/env bash
# tune.sh
#
# WHAT THIS DOES:
#   A single place to set all training and inference parameters.
#   Run it in one of three modes:
#
#     bash tune.sh train      - train once with the settings below
#     bash tune.sh sweep      - train multiple times with different learning rates
#     bash tune.sh infer      - run inference on a video using a saved model
#     bash tune.sh models     - list all saved models
#
# EXAMPLE USAGE:
#   bash tune.sh train
#   bash tune.sh infer
#   bash tune.sh models

# ============================================================
#  SECTION 1 — PARAMETERS (edit these before running)
# ============================================================

# ── Dataset ─────────────────────────────────────────────────
# Path to your dataset .yaml file.
# This tells YOLO where to find train/val images and labels.
# (We will create this file when preparing the FBD-SV-2024 dataset.)
DATA_YAML="datasets/bird.yaml"

# ── Base model ───────────────────────────────────────────────
# Which YOLOv8 model size to start from.
# n=nano (fastest, least accurate), s=small, m=medium, l=large, x=extra-large
# Start with "yolov8n.pt" — switch to "yolov8s.pt" if accuracy is too low.
BASE_MODEL="yolov8n.pt"

# ── Learning rate ────────────────────────────────────────────
# How fast the model updates its weights each step.
# Too high → training is unstable.  Too low → training is very slow.
# Good starting values to try: 0.001, 0.005, 0.01
LEARNING_RATE=0.01

# Final learning rate = LEARNING_RATE * LR_FINAL_FRACTION
# e.g. 0.01 * 0.01 = 0.0001 at the end (a cosine decay)
LR_FINAL_FRACTION=0.01

# ── Training length ──────────────────────────────────────────
# How many times to go through the full dataset.
# More epochs = better model (up to a point). 50 is a good start.
EPOCHS=50

# Stop early if no improvement for this many epochs (saves time).
PATIENCE=10

# ── Hardware ─────────────────────────────────────────────────
# "cpu"  → use the CPU (slow but works everywhere)
# "0"    → use GPU number 0 (fast, needs CUDA)
DEVICE="cpu"

# How many images to process at once.
# Lower this (to 8 or 4) if you get an "out of memory" error.
BATCH_SIZE=16

# Resize all images to this size before training.
# 640 is standard for YOLOv8.
IMAGE_SIZE=640

# ── Learning rate sweep (used only by the "sweep" mode) ──────
# List of learning rates to try, one run each.
# Results will be saved in separate folders so you can compare.
LR_SWEEP="0.001 0.005 0.01 0.05"

# ── Inference ────────────────────────────────────────────────
# Video file to run inference on.
VIDEO_FILE="data/raw_clips/my_clip.mp4"

# Which saved model weights to use for inference.
# After training, models are saved to:  runs/detect/<name>/weights/best.pt
# Change this to the path printed at the end of a training run.
INFER_MODEL="runs/detect/train_run/weights/best.pt"

# Minimum confidence score to count as a bird (0.0 to 1.0).
CONFIDENCE=0.4

# Set to "true" to open a live video window with boxes drawn.
# Set to "false" to just print results to the terminal.
SHOW_VIDEO="false"

# Set to "true" to save a .jpg crop of every detected bird.
SAVE_CROPS="false"

# ============================================================
#  SECTION 2 — INTERNAL LOGIC  (you usually don't edit below)
# ============================================================

# Exit immediately if any command fails
set -e

# ── Helper: print a section header ───────────────────────────
header() {
    echo ""
    echo "============================================"
    echo "  $1"
    echo "============================================"
}

# ── Helper: list all saved model weight files ────────────────
list_models() {
    header "Saved models"
    # Find all best.pt files under runs/
    if [ -d "runs" ]; then
        find runs -name "best.pt" | sort
        echo ""
        find runs -name "last.pt" | sort
    else
        echo "  No runs/ folder found. Train a model first."
    fi
    echo ""
}

# ── Mode: train (single run) ─────────────────────────────────
do_train() {
    header "Training — single run"
    echo "  Model    : $BASE_MODEL"
    echo "  LR       : $LEARNING_RATE"
    echo "  Epochs   : $EPOCHS"
    echo "  Batch    : $BATCH_SIZE"
    echo "  Device   : $DEVICE"

    # Build a descriptive run name so the saved folder tells you what settings were used
    # e.g. "lr0.01_ep50_yolov8n"
    RUN_NAME="lr${LEARNING_RATE}_ep${EPOCHS}_$(basename $BASE_MODEL .pt)"

    python train.py \
        --data       "$DATA_YAML"          \
        --model      "$BASE_MODEL"         \
        --lr         "$LEARNING_RATE"      \
        --lr_final   "$LR_FINAL_FRACTION"  \
        --epochs     "$EPOCHS"             \
        --patience   "$PATIENCE"           \
        --batch      "$BATCH_SIZE"         \
        --imgsz      "$IMAGE_SIZE"         \
        --device     "$DEVICE"             \
        --name       "$RUN_NAME"

    echo ""
    echo "Best weights: runs/detect/${RUN_NAME}/weights/best.pt"
    echo "Update INFER_MODEL in tune.sh to use this model for inference."
}

# ── Mode: sweep (train multiple learning rates) ───────────────
do_sweep() {
    header "Learning rate sweep: $LR_SWEEP"

    for LR in $LR_SWEEP; do
        echo ""
        echo "--- Starting run with LR=$LR ---"

        RUN_NAME="lr${LR}_ep${EPOCHS}_$(basename $BASE_MODEL .pt)"

        python train.py \
            --data       "$DATA_YAML"          \
            --model      "$BASE_MODEL"         \
            --lr         "$LR"                 \
            --lr_final   "$LR_FINAL_FRACTION"  \
            --epochs     "$EPOCHS"             \
            --patience   "$PATIENCE"           \
            --batch      "$BATCH_SIZE"         \
            --imgsz      "$IMAGE_SIZE"         \
            --device     "$DEVICE"             \
            --name       "$RUN_NAME"

        echo "  Saved: runs/detect/${RUN_NAME}/weights/best.pt"
    done

    header "Sweep complete — all saved models:"
    list_models

    echo "Compare results in runs/detect/ then set INFER_MODEL to the best one."
}

# ── Mode: infer (run on a video) ─────────────────────────────
do_infer() {
    header "Inference"
    echo "  Video  : $VIDEO_FILE"
    echo "  Model  : $INFER_MODEL"
    echo "  Conf   : $CONFIDENCE"

    # Check the model file exists before trying to run
    if [ ! -f "$INFER_MODEL" ]; then
        echo ""
        echo "ERROR: Model file not found: $INFER_MODEL"
        echo "       Train a model first (bash tune.sh train)"
        echo "       Then update INFER_MODEL in tune.sh"
        exit 1
    fi

    # Check the video file exists
    if [ ! -f "$VIDEO_FILE" ]; then
        echo ""
        echo "ERROR: Video file not found: $VIDEO_FILE"
        echo "       Update VIDEO_FILE in tune.sh"
        exit 1
    fi

    # Build the --show flag only if SHOW_VIDEO is true
    SHOW_FLAG=""
    if [ "$SHOW_VIDEO" = "true" ]; then
        SHOW_FLAG="--show"
    fi

    # Build the --save_crops flag only if SAVE_CROPS is true
    CROPS_FLAG=""
    if [ "$SAVE_CROPS" = "true" ]; then
        CROPS_FLAG="--save_crops"
    fi

    # Run detection + counting
    python detect.py \
        --video      "$VIDEO_FILE"   \
        --model      "$INFER_MODEL"  \
        --confidence "$CONFIDENCE"   \
        $SHOW_FLAG                   \

    echo ""
    echo "--- Species identification ---"

    # Run species identification (currently returns Unidentifiable — Week 4 will fix this)
    python identify.py \
        --video         "$VIDEO_FILE"   \
        --model         "$INFER_MODEL"  \
        --confidence    "$CONFIDENCE"   \
        $CROPS_FLAG                     \
}

# ============================================================
#  SECTION 3 — COMMAND DISPATCHER
# ============================================================

# Read the first argument passed to the script (e.g. "train", "sweep", "infer")
MODE="${1:-help}"

case "$MODE" in

    train)
        do_train
        ;;

    sweep)
        do_sweep
        ;;

    infer)
        do_infer
        ;;

    models)
        list_models
        ;;

    help | --help | -h | *)
        echo ""
        echo "Usage: bash tune.sh <mode>"
        echo ""
        echo "  train    Train once with the parameters set at the top of this file"
        echo "  sweep    Train multiple times across different learning rates"
        echo "  infer    Run the saved model on a video file"
        echo "  models   List all saved model weight files"
        echo ""
        echo "Edit the PARAMETERS section at the top of tune.sh before running."
        ;;
esac
