# identify.py
#
# GOAL: For each bird detected in a video, identify its species.
#
# HOW TO RUN:
#   python identify.py --video data/raw_clips/my_clip.mp4
#
# OPTIONAL FLAGS:
#   --save_crops    : save each bird crop as a .jpg in data/crops/
#   --confidence    : change detection threshold (default 0.4)
#
# CURRENT STATUS:
#   Detection works. Species always returns "Unidentifiable" until
#   the classifier is trained in Week 4.

import cv2
import argparse

from datasets.video_reader import VideoReader
from models.detector       import BirdDetector
from models.classifier     import SpeciesClassifier


def crop_bird(frame, box):
    """
    Cuts out the bird region from a frame using its bounding box.

    frame : numpy array (H x W x 3)
    box   : dict with x1, y1, x2, y2 in pixels

    Returns the cropped numpy array, or None if the box has no area.
    """
    h, w = frame.shape[:2]
    x1   = max(0, box["x1"])
    y1   = max(0, box["y1"])
    x2   = min(w, box["x2"])
    y2   = min(h, box["y2"])

    if x2 <= x1 or y2 <= y1:
        return None

    # frame[y_start:y_end, x_start:x_end] slices out the rectangular region
    return frame[y1:y2, x1:x2]


def run(video_path, model_weights="yolov8n.pt", species_weights=None,
        confidence=0.4, save_crops=False):
    """
    Runs detection + species identification on every frame of a video.
    Prints a timestamped list of every bird sighting.

    video_path      : path to the .mp4 file
    model_weights   : YOLO detector weights
    species_weights : species classifier weights (None = stub, returns Unidentifiable)
    confidence      : minimum detection score
    save_crops      : if True, save each bird crop to data/crops/
    """

    reader     = VideoReader(video_path)
    detector   = BirdDetector(weights=model_weights, confidence=confidence)
    classifier = SpeciesClassifier(weights=species_weights)

    if save_crops:
        import os
        os.makedirs("data/crops", exist_ok=True)
        print("Saving crops → data/crops/")

    all_sightings = []
    crop_counter  = 0

    print("\nScanning video...\n")

    for frame, timestamp in reader:

        birds = detector.find_birds(frame)

        for b in birds:
            # Crop just the bird out of the frame
            crop = crop_bird(frame, b)

            # Ask the classifier what species it is
            species = classifier.classify(crop)

            # Record this sighting
            all_sightings.append({
                "time_sec"   : round(timestamp, 2),
                "species"    : species,
                "confidence" : b["confidence"],
            })

            # Optionally save the crop image
            if save_crops and crop is not None:
                path = f"data/crops/bird_{crop_counter:05d}_t{timestamp:.1f}s.jpg"
                cv2.imwrite(path, crop)
                crop_counter += 1

    # Print the full sighting log
    print(f"── Sightings ({len(all_sightings)} detections) ────────────────")
    for s in all_sightings:
        print(f"  t={s['time_sec']:6.2f}s  conf={s['confidence']:.2f}  species={s['species']}")

    if save_crops:
        print(f"\nSaved {crop_counter} crop images → data/crops/")

    return all_sightings


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Identify bird species in a video")
    parser.add_argument("--video",           required=True,         help="Path to the .mp4 video file")
    parser.add_argument("--model",           default="yolov8n.pt",  help="YOLOv8 detector weights")
    parser.add_argument("--species_model",   default=None,          help="Species classifier weights (optional)")
    parser.add_argument("--confidence",      default=0.4, type=float, help="Detection confidence threshold")
    parser.add_argument("--save_crops",      action="store_true",   help="Save bird crops to data/crops/")
    args = parser.parse_args()

    run(args.video, args.model, args.species_model, args.confidence, args.save_crops)
