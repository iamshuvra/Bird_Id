# datasets/verify_label.py
#
# WHAT THIS DOES:
#   Opens one image and draws its YOLO bounding box on it, then saves the result.
#   Use this after convert_labels.py to confirm the boxes are correct before training.
#
# HOW TO RUN:
#   python datasets/verify_label.py --data_root data/
#
# WHAT TO CHECK:
#   Open data/check.jpg — the green box should be tightly around the bird.
#   If the box is wildly wrong (huge, empty, in the wrong corner), something is
#   wrong with the label conversion. Compare the .xml and .txt files manually.
#
# HOW TO CHECK A SPECIFIC IMAGE:
#   python datasets/verify_label.py --image data/images/train/bird_5_000010.jpg

import os
import cv2
import argparse


def draw_yolo_boxes(image_path, txt_path, out_path):
    """
    Draws YOLO bounding boxes from a .txt label file onto an image and saves it.

    image_path : path to the .jpg image
    txt_path   : path to the matching YOLO .txt label file
    out_path   : where to save the output image with boxes drawn
    """

    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: Could not open image: {image_path}")
        return

    h, w = img.shape[:2]  # image height and width in pixels
    print(f"Image: {image_path}  ({w}x{h})")

    # Read the label file
    if not os.path.exists(txt_path):
        print(f"ERROR: Label file not found: {txt_path}")
        print(f"       Run convert_labels.py first.")
        return

    with open(txt_path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]

    if not lines:
        print(f"  No birds in this frame (empty label file) — try a different image.")
        return

    print(f"  Found {len(lines)} bird box(es)")

    for line in lines:
        parts  = line.split()
        # YOLO format: class_id cx cy bw bh  (all fractions 0.0–1.0)
        class_id = int(parts[0])
        cx = float(parts[1])
        cy = float(parts[2])
        bw = float(parts[3])
        bh = float(parts[4])

        # Convert fractions back to pixel coordinates for drawing
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)

        # Draw a green rectangle on the image
        cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

        # Write "bird" label above the box
        cv2.putText(img, "bird", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        print(f"  Box: x1={x1} y1={y1} x2={x2} y2={y2}")

    # Save the annotated image
    cv2.imwrite(out_path, img)
    print(f"\nSaved to: {out_path}")
    print("Open that file to visually confirm the box is around a bird.")


def find_image_with_birds(img_dir, txt_dir=None):
    """
    Finds the first image that has at least one bird label (non-empty .txt).
    Returns the image path, or None if not found.
    """
    if txt_dir is None:
        txt_dir = img_dir

    for filename in sorted(os.listdir(img_dir)):
        if not filename.endswith(".jpg"):
            continue

        txt_path = os.path.join(txt_dir, filename.replace(".jpg", ".txt"))
        if not os.path.exists(txt_path):
            continue

        # Check if the label file is non-empty (has at least one bird)
        with open(txt_path) as f:
            if f.read().strip():
                return os.path.join(img_dir, filename), txt_path

    return None, None


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify one YOLO label visually")
    parser.add_argument("--data_root", default="data/",
                        help="Root data folder (default: data/)")
    parser.add_argument("--image", default=None,
                        help="Specific image to check (optional). "
                             "If omitted, auto-picks the first image with a bird.")
    parser.add_argument("--out", default="data/check.jpg",
                        help="Where to save the annotated output image (default: data/check.jpg)")
    args = parser.parse_args()

    if args.image:
        # User specified a particular image
        img_path = args.image
        txt_path = img_path.replace(".jpg", ".txt")
    else:
        # Auto-find the first training image that has a bird label
        img_dir = os.path.join(args.data_root, "images", "train")
        print(f"Looking for an image with birds in: {img_dir}")
        img_path, txt_path = find_image_with_birds(img_dir)

        if img_path is None:
            print("No labelled images found.")
            print("Make sure you have run:")
            print("  1. python data/scripts/split_video_frames_for_object_detection.py --data_root_path data/")
            print("  2. python datasets/convert_labels.py --data_root data/")
            exit(1)

    draw_yolo_boxes(img_path, txt_path, args.out)
