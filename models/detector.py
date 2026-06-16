# models/detector.py
#
# WHAT THIS FILE DOES:
#   Wraps YOLOv8 so the rest of the project can find birds in a frame
#   with one simple function call: detector.find_birds(frame)
#
# OUTPUT per detection:
#   { "x1", "y1", "x2", "y2", "confidence" }
#   x1,y1 = top-left corner of the box in pixels
#   x2,y2 = bottom-right corner of the box in pixels
#   confidence = 0.0 to 1.0, how sure the model is this is a bird

from ultralytics import YOLO

# In the COCO dataset (what yolov8n.pt was trained on), class 14 = "bird"
BIRD_CLASS_ID = 14


class BirdDetector:
    """
    Finds birds in a single image frame.

    Usage:
        detector = BirdDetector()             # loads yolov8n.pt automatically
        birds = detector.find_birds(frame)    # frame is a numpy array from opencv
        for b in birds:
            print(b["x1"], b["y1"], b["x2"], b["y2"])
    """

    def __init__(self, weights="yolov8n.pt", confidence=0.4):
        """
        weights    : path to model weights file
                     "yolov8n.pt" downloads ~6MB automatically on first run
        confidence : only return detections above this score (0.0 to 1.0)
        """
        print(f"Loading detector: {weights}")
        self.model      = YOLO(weights)
        self.confidence = confidence

    def find_birds(self, frame):
        """
        Run the model on one frame and return a list of bird detections.

        frame   : numpy array shape (H, W, 3) — a BGR image from opencv
        returns : list of dicts, one per detected bird
        """

        # Run YOLO. classes=[14] means only look for birds, skip everything else.
        results = self.model(frame, classes=[BIRD_CLASS_ID],
                             conf=self.confidence, verbose=False)

        birds = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            conf = round(float(box.conf[0]), 3)
            birds.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "confidence": conf})

        return birds
