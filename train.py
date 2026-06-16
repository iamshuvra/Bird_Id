# train.py
#
# WHAT THIS DOES:
#   Trains a YOLOv8 model on your bird dataset.
#   All settings (learning rate, epochs, etc.) come in as command-line arguments
#   so tune.sh can call this with different values without editing the Python code.
#
# HOW TO RUN DIRECTLY:
#   python train.py --data datasets/bird.yaml --lr 0.01 --epochs 50
#
# HOW IT IS NORMALLY USED:
#   Called by tune.sh — you edit the parameters there, not here.

import argparse
import os
from ultralytics import YOLO


def run_training(args):
    """
    Trains YOLOv8 with the given settings and saves the best weights.

    args : the parsed command-line arguments (from argparse below)
    """

    print("\n" + "="*50)
    print("  TRAINING RUN")
    print("="*50)
    print(f"  Base model   : {args.model}")
    print(f"  Dataset yaml : {args.data}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Epochs       : {args.epochs}")
    print(f"  Batch size   : {args.batch}")
    print(f"  Image size   : {args.imgsz}")
    print(f"  Run name     : {args.name}")
    print("="*50 + "\n")

    # Load the base model
    # If args.model is a path like "runs/train/exp1/weights/best.pt" it resumes from there.
    # If it is "yolov8n.pt" it downloads and starts fresh.
    model = YOLO(args.model)

    # Train the model.
    # ultralytics saves:
    #   runs/detect/<name>/weights/best.pt    ← best weights during training
    #   runs/detect/<name>/weights/last.pt    ← weights at the very last epoch
    results = model.train(
        data      = args.data,        # path to dataset.yaml
        epochs    = args.epochs,      # how many full passes through the dataset
        imgsz     = args.imgsz,       # resize all images to this size (pixels)
        batch     = args.batch,       # images processed at once (lower if GPU runs out of memory)
        lr0       = args.lr,          # initial learning rate
        lrf       = args.lr_final,    # final learning rate = lr0 * lrf (as a fraction)
        name      = args.name,        # folder name for this run's outputs
        exist_ok  = True,             # overwrite if the folder already exists
        device    = args.device,      # "cpu", "0" for GPU 0, "0,1" for two GPUs
        patience  = args.patience,    # stop early if no improvement after this many epochs
        save      = True,             # save best.pt and last.pt
        plots     = True,             # save training curve plots
        verbose   = True,
    )

    # Tell the user where the saved model is
    best_weights = f"runs/detect/{args.name}/weights/best.pt"
    print("\n" + "="*50)
    print("  TRAINING COMPLETE")
    print(f"  Best weights saved to: {best_weights}")
    print("="*50 + "\n")

    return best_weights


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train YOLOv8 for bird detection")

    # Dataset
    parser.add_argument("--data",       required=True,              help="Path to dataset .yaml file")

    # Model selection
    parser.add_argument("--model",      default="yolov8n.pt",       help="Base model: yolov8n/s/m/l/x.pt or path to checkpoint")

    # Learning rate
    parser.add_argument("--lr",         default=0.01,   type=float, help="Initial learning rate (default: 0.01)")
    parser.add_argument("--lr_final",   default=0.01,   type=float, help="Final LR as fraction of lr0 (default: 0.01)")

    # Training length
    parser.add_argument("--epochs",     default=50,     type=int,   help="Number of training epochs (default: 50)")
    parser.add_argument("--patience",   default=10,     type=int,   help="Early stop if no improvement after N epochs")

    # Hardware
    parser.add_argument("--batch",      default=16,     type=int,   help="Batch size (default: 16; lower if out of memory)")
    parser.add_argument("--imgsz",      default=640,    type=int,   help="Input image size in pixels (default: 640)")
    parser.add_argument("--device",     default="cpu",              help="Device: cpu / 0 / 0,1 (default: cpu)")

    # Run naming
    parser.add_argument("--name",       default="train_run",        help="Name for this run (used as output folder name)")

    args = parser.parse_args()

    run_training(args)
