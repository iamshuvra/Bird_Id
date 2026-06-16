# models/classifier.py
#
# WHAT THIS FILE DOES:
#   Takes a cropped bird image and returns the species name.
#
# CURRENT STATUS:
#   This is a stub — it always returns "Unidentifiable".
#   We need to train a classifier first (Week 4).
#   When training is done, pass the model path to SpeciesClassifier()
#   and uncomment the classification code in classify().

import os
from ultralytics import YOLO


class SpeciesClassifier:
    """
    Identifies the species of a cropped bird image.

    Usage (right now):
        classifier = SpeciesClassifier()          # no model yet
        species = classifier.classify(crop)       # returns "Unidentifiable"

    Usage (after Week 4):
        classifier = SpeciesClassifier("models/species.pt")
        species = classifier.classify(crop)       # returns "Herring Gull" etc.
    """

    def __init__(self, weights=None, confidence=0.6):
        """
        weights    : path to trained species classifier .pt file
                     Leave as None until you have a trained model.
        confidence : minimum score to name a species; below this → "Unidentifiable"
        """
        self.model      = None
        self.confidence = confidence

        if weights and os.path.exists(weights):
            self.model = YOLO(weights)
            print(f"Species classifier loaded: {weights}")
        else:
            print("Species classifier: no model loaded (returning Unidentifiable)")

    def classify(self, crop):
        """
        crop : numpy array (H x W x 3) — just the bird, already cropped from the frame

        Returns a string like "Double-crested Cormorant" or "Unidentifiable"
        """

        if self.model is None or crop is None:
            return "Unidentifiable"

        # ── Uncomment this block in Week 4 once the model is trained ─────────
        # results  = self.model(crop, verbose=False)
        # top_idx  = results[0].probs.top1              # index of best class
        # top_conf = results[0].probs.top1conf.item()   # score 0.0-1.0
        # if top_conf >= self.confidence:
        #     return results[0].names[top_idx]
        # ─────────────────────────────────────────────────────────────────────

        return "Unidentifiable"
