# detect.py
#
# GOAL: Count the number of unique birds that appear in a video.
#
# HOW TO RUN:
#   python detect.py --video data/raw_clips/my_clip.mp4
#
# OPTIONAL FLAGS:
#   --show          : open a window to watch the video with boxes drawn on it
#   --confidence    : change detection threshold (default 0.4)
#
# WHAT IT PRINTS:
#   Progress every 100 frames, then a final count of unique birds.

import cv2
import argparse

from datasets.video_reader import VideoReader      # opens the video, yields frames
from models.detector       import BirdDetector     # YOLOv8 wrapper
from utils.counter         import BirdCounter      # tracks unique birds


def run(video_path, model_weights="yolov8n.pt", confidence=0.4, show=False):
    """
    Full detection + counting run on a single video file.

    video_path    : path to the .mp4 file
    model_weights : YOLO weights (downloads automatically if not found)
    confidence    : minimum detection score to count as a bird
    show          : display video with boxes in a pop-up window
    """

    # Load the video and the detector
    reader   = VideoReader(video_path)
    detector = BirdDetector(weights=model_weights, confidence=confidence)
    counter  = BirdCounter()

    frame_number = 0

    print("\nStarting detection...\n")

    for frame, timestamp in reader:

        # Find all birds in this frame
        birds = detector.find_birds(frame)

        # Update the unique-bird counter
        counter.update(birds)

        # Optionally draw green boxes and show the video in a window
        if show:
            for b in birds:
                cv2.rectangle(frame, (b["x1"], b["y1"]), (b["x2"], b["y2"]),
                              color=(0, 255, 0), thickness=2)
                cv2.putText(frame, f"bird {b['confidence']:.2f}",
                            (b["x1"], b["y1"] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            # Show running count in the top-left corner
            cv2.putText(frame, f"Unique birds so far: {counter.total_count}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            cv2.imshow("Bird Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Print progress every 100 frames
        if frame_number % 100 == 0:
            print(f"  Frame {frame_number}  |  unique birds so far: {counter.total_count}")

        frame_number += 1

    cv2.destroyAllWindows()

    total = counter.finish()
    print(f"\n── Result ───────────────────────────────")
    print(f"  Frames processed  : {frame_number}")
    print(f"  Unique birds found: {total}")
    return total


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count birds in a video")
    parser.add_argument("--video",       required=True,              help="Path to the .mp4 video file")
    parser.add_argument("--model",       default="yolov8n.pt",       help="YOLOv8 weights (default: yolov8n.pt)")
    parser.add_argument("--confidence",  default=0.4,  type=float,   help="Detection confidence threshold")
    parser.add_argument("--show",        action="store_true",        help="Show live video with boxes")
    args = parser.parse_args()

    run(args.video, args.model, args.confidence, args.show)
