# utils/counter.py
#
# WHAT THIS FILE DOES:
#   Counts how many unique birds appear in a video.
#
# THE PROBLEM:
#   YOLOv8 finds birds in EVERY frame.
#   If a bird flies past in 40 frames, we get 40 detections of the same bird.
#   This file solves that: if the same bird appears in many frames, count it once.
#
# HOW IT WORKS (nearest-neighbour tracking):
#   - Each frame, for every detected bird box, we look at all the boxes we saw
#     in the previous frame.
#   - If a box is within MAX_DISTANCE pixels of an existing box → same bird.
#   - If a box is too far away from all previous boxes → new bird, start tracking.
#   - If a bird has not appeared for MAX_MISSING frames → it left; count it once.

MAX_DISTANCE = 100   # pixels — how close two boxes must be to be the same bird
MAX_MISSING  = 15    # frames — how long a bird can disappear before we count it


def _center(x1, y1, x2, y2):
    """Returns the center point (cx, cy) of a bounding box."""
    return (x1 + x2) / 2, (y1 + y2) / 2


def _dist(cx1, cy1, cx2, cy2):
    """Returns straight-line pixel distance between two points."""
    return ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5


class BirdCounter:
    """
    Tracks unique birds across video frames and counts them.

    Usage:
        counter = BirdCounter()
        for each frame:
            boxes = detector.find_birds(frame)   # list of {x1,y1,x2,y2} dicts
            counter.update(boxes)
        total = counter.finish()
        print(f"Total birds: {total}")
    """

    def __init__(self):
        self.active_tracks = []   # birds currently being tracked
        self.total_count   = 0    # birds counted so far (finished tracks)

    def update(self, boxes):
        """
        Called once per frame with all detected bird boxes.

        boxes : list of dicts with keys x1, y1, x2, y2
        """

        # Track which active tracks got matched this frame
        matched = [False] * len(self.active_tracks)

        for b in boxes:
            cx, cy = _center(b["x1"], b["y1"], b["x2"], b["y2"])

            # Find the closest existing track to this box
            best_i    = None
            best_dist = float("inf")
            for i, track in enumerate(self.active_tracks):
                d = _dist(cx, cy, track["cx"], track["cy"])
                if d < best_dist:
                    best_dist = d
                    best_i    = i

            if best_i is not None and best_dist < MAX_DISTANCE:
                # Same bird — update its position and reset missing counter
                self.active_tracks[best_i]["cx"]      = cx
                self.active_tracks[best_i]["cy"]      = cy
                self.active_tracks[best_i]["missing"] = 0
                matched[best_i] = True
            else:
                # New bird — start a new track for it
                self.active_tracks.append({"cx": cx, "cy": cy, "missing": 0})
                matched.append(True)

        # Increment missing counter for tracks not seen this frame
        for i, track in enumerate(self.active_tracks):
            if not matched[i]:
                track["missing"] += 1

        # Retire tracks that have been missing too long → count them
        still_active = []
        for track in self.active_tracks:
            if track["missing"] > MAX_MISSING:
                self.total_count += 1
            else:
                still_active.append(track)
        self.active_tracks = still_active

    def finish(self):
        """
        Call after the last frame to close any still-open tracks.
        Returns the final total bird count.
        """
        self.total_count += len(self.active_tracks)
        self.active_tracks = []
        return self.total_count
