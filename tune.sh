#!/usr/bin/env bash
# tune.sh
#
# WHAT THIS DOES:
#   A single place to set all parameters and run every step of the pipeline.
#
# -- DATA PREPARATION (run these first, in order) ----------------------------
#     bash tune.sh extract    - Step 1: extract .jpg frames from video clips
#     bash tune.sh convert    - Step 2: convert XML labels to YOLO .txt format
#     bash tune.sh verify     - Step 3: draw one box on one frame to check labels
#     bash tune.sh check      - show file counts for all data/ subfolders
#
# -- TRAINING ----------------------------------------------------------------
#     bash tune.sh train      - train once with the settings below
#     bash tune.sh sweep      - train multiple times with different learning rates
#     bash tune.sh models     - list all saved model weight files
#
# -- INFERENCE ---------------------------------------------------------------
#     bash tune.sh infer      - run the saved model on a video file
#
# EXAMPLE (full pipeline from scratch):
#   bash tune.sh extract
#   bash tune.sh convert
#   bash tune.sh verify        ← open data/check.jpg and confirm box is on a bird
#   bash tune.sh train
#   bash tune.sh infer

# ============================================================
#  SECTION 1 — PARAMETERS (edit these before running)
# ============================================================

# -- Data root ------------------------------------------------
# Root folder that contains videos/, labels/, images/ etc.
DATA_ROOT="data/"

# -- Dataset -------------------------------------------------
# Path to your dataset .yaml file.
# This tells YOLO where to find train/val images and labels.
DATA_YAML="datasets/bird.yaml"

# -- Verify: which image to check -----------------------------
# Used by "bash tune.sh verify".
# Leave blank to auto-pick the first image that has a bird label.
# Or set to a specific frame, e.g. "data/images/train/bird_1_000000.jpg"
VERIFY_IMAGE=""

# -- Base model -----------------------------------------------
# Which YOLOv8 model size to start from.
# n=nano (fastest, least accurate), s=small, m=medium, l=large, x=extra-large
# Start with "yolov8n.pt" — switch to "yolov8s.pt" if accuracy is too low.
BASE_MODEL="yolov8n.pt"

# -- Learning rate --------------------------------------------
# How fast the model updates its weights each step.
# Too high -> training is unstable.  Too low -> training is very slow.
# Good starting values to try: 0.001, 0.005, 0.01
LEARNING_RATE=0.01

# Final learning rate = LEARNING_RATE * LR_FINAL_FRACTION
# e.g. 0.01 * 0.01 = 0.0001 at the end (a cosine decay)
LR_FINAL_FRACTION=0.01

# -- Training length ------------------------------------------
# How many times to go through the full dataset.
# More epochs = better model (up to a point). 50 is a good start.
EPOCHS=50

# Stop early if no improvement for this many epochs (saves time).
PATIENCE=10

# -- Hardware -------------------------------------------------
# "cpu"  -> use the CPU (slow but works everywhere)
# "0"    -> use GPU number 0 (fast, needs CUDA)
DEVICE="cpu"

# How many images to process at once.
# Lower this (to 8 or 4) if you get an "out of memory" error.
BATCH_SIZE=16

# Resize all images to this size before training.
# 640 is standard for YOLOv8.
IMAGE_SIZE=640

# -- Learning rate sweep (used only by the "sweep" mode) ------
# List of learning rates to try, one run each.
# Results will be saved in separate folders so you can compare.
LR_SWEEP="0.001 0.005 0.01 0.05"

# -- Inference ------------------------------------------------
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

# -- Helper: print a section header ---------------------------
header() {
    echo ""
    echo "============================================"
    echo "  $1"
    echo "============================================"
}

# -- Helper: list all saved model weight files ----------------
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

# -- Mode: extract frames from videos -------------------------
do_extract() {
    header "Step 1 — Extract frames from video clips"
    echo "  Videos in : ${DATA_ROOT}videos/train/  and  ${DATA_ROOT}videos/val/"
    echo "  Frames out: ${DATA_ROOT}images/train/  and  ${DATA_ROOT}images/val/"
    echo ""
    echo "  This reads every .mp4 and saves each frame as a .jpg."
    echo "  Takes a few minutes for 483 clips (~28k frames total)."
    echo ""

    # Check the videos folder exists
    if [ ! -d "${DATA_ROOT}videos/train" ]; then
        echo "ERROR: ${DATA_ROOT}videos/train/ not found."
        echo "       Make sure the FBD-SV-2024 dataset is in the data/ folder."
        exit 1
    fi

    python data/scripts/split_video_frames_for_object_detection.py \
        --data_root_path "$DATA_ROOT"

    echo ""
    echo "Done. Check file counts with: bash tune.sh check"
}

# -- Mode: convert XML labels to YOLO format -------------------
do_convert() {
    header "Step 2 — Convert XML labels to YOLO .txt format"
    echo "  XML source : ${DATA_ROOT}labels/train/  and  ${DATA_ROOT}labels/val/"
    echo "  TXT output : alongside the images in ${DATA_ROOT}images/train|val/"
    echo ""

    # Check that frame extraction has been run first
    if [ ! -d "${DATA_ROOT}images/train" ]; then
        echo "ERROR: ${DATA_ROOT}images/train/ not found."
        echo "       Run frame extraction first: bash tune.sh extract"
        exit 1
    fi

    python datasets/convert_labels.py --data_root "$DATA_ROOT"

    echo ""
    echo "Done. Verify labels with: bash tune.sh verify"
}

# -- Mode: verify one label visually ---------------------------
do_verify() {
    header "Step 3 — Verify one label visually"
    echo "  Will draw a bounding box on one frame and save to data/check.jpg"
    echo "  Open data/check.jpg — the green box should be tightly around a bird."
    echo ""

    # Build the --image flag only if VERIFY_IMAGE is set
    IMAGE_FLAG=""
    if [ -n "$VERIFY_IMAGE" ]; then
        IMAGE_FLAG="--image $VERIFY_IMAGE"
    fi

    python datasets/verify_label.py \
        --data_root "$DATA_ROOT"    \
        $IMAGE_FLAG

    echo ""
    echo "If the box looks correct -> ready to train: bash tune.sh train"
    echo "If the box looks wrong   -> check datasets/convert_labels.py"
}

# -- Helper: count matching files in a folder -----------------
# (defined here at top level — bash does not allow nested functions)
count_files() {
    local folder="$1"
    local pattern="$2"   # e.g. "*.jpg" or "*.xml"
    if [ -d "$folder" ]; then
        find "$folder" -maxdepth 1 -name "$pattern" | wc -l
    else
        echo "MISSING"
    fi
}

# -- Mode: check data folder file counts -----------------------
do_check() {
    header "Data folder summary"

    echo ""
    echo "  Folder                        Files"
    echo "  -----------------------------------------------------"

    # Videos
    printf "  %-30s %s\n" "data/videos/train/ (.mp4)" \
        "$(count_files ${DATA_ROOT}videos/train '*.mp4')"
    printf "  %-30s %s\n" "data/videos/val/   (.mp4)" \
        "$(count_files ${DATA_ROOT}videos/val   '*.mp4')"

    # XML labels
    printf "  %-30s %s\n" "data/labels/train/ (.xml)" \
        "$(count_files ${DATA_ROOT}labels/train '*.xml')"
    printf "  %-30s %s\n" "data/labels/val/   (.xml)" \
        "$(count_files ${DATA_ROOT}labels/val   '*.xml')"

    # Extracted images
    printf "  %-30s %s\n" "data/images/train/ (.jpg)" \
        "$(count_files ${DATA_ROOT}images/train '*.jpg')"
    printf "  %-30s %s\n" "data/images/val/   (.jpg)" \
        "$(count_files ${DATA_ROOT}images/val   '*.jpg')"

    # Converted YOLO labels
    printf "  %-30s %s\n" "data/images/train/ (.txt)" \
        "$(count_files ${DATA_ROOT}images/train '*.txt')"
    printf "  %-30s %s\n" "data/images/val/   (.txt)" \
        "$(count_files ${DATA_ROOT}images/val   '*.txt')"

    echo ""
    echo "  EXPECTED COUNTS (FBD-SV-2024 full dataset):"
    echo "    videos:          400 train,  83 val"
    echo "    XML labels:   23,979 train,  4,715 val"
    echo "    images (.jpg): ~23,979 train, ~4,715 val  (after extract)"
    echo "    labels (.txt): ~23,979 train, ~4,715 val  (after convert)"
    echo ""
    echo "  If .jpg count is 0 -> run: bash tune.sh extract"
    echo "  If .txt count is 0 -> run: bash tune.sh convert"
}

# -- Mode: train (single run) ---------------------------------
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

# -- Mode: sweep (train multiple learning rates) ---------------
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

# -- Mode: infer (run on a video) -----------------------------
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
        $SHOW_FLAG

    echo ""
    echo "--- Species identification ---"

    # Run species identification (currently returns Unidentifiable -- Week 4 will fix this)
    python identify.py \
        --video         "$VIDEO_FILE"   \
        --model         "$INFER_MODEL"  \
        --confidence    "$CONFIDENCE"   \
        $CROPS_FLAG
}

# ============================================================
#  SECTION 3 — COMMAND DISPATCHER
# ============================================================

# Read the first argument passed to the script (e.g. "train", "sweep", "infer")
MODE="${1:-help}"

case "$MODE" in

    extract)
        do_extract
        ;;

    convert)
        do_convert
        ;;

    verify)
        do_verify
        ;;

    check)
        do_check
        ;;

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
        echo "  -- Data preparation (run in this order) --"
        echo "  extract  Extract .jpg frames from all video clips"
        echo "  convert  Convert XML labels to YOLO .txt format"
        echo "  verify   Draw one label box on one frame -> open data/check.jpg"
        echo "  check    Show file counts for all data/ subfolders"
        echo ""
        echo "  -- Training ------------------------------"
        echo "  train    Train once with the parameters set at the top of this file"
        echo "  sweep    Train multiple times across different learning rates"
        echo "  models   List all saved model weight files"
        echo ""
        echo "  -- Inference -----------------------------"
        echo "  infer    Run the saved model on a video file"
        echo ""
        echo "Edit the PARAMETERS section at the top of tune.sh before running."
        ;;
esac
