# Bird Identification – Offshore Wind Turbine Monitor

Detects, counts, and identifies birds in fixed-camera video.

Status: **Week 1 – scaffold complete, detection ready**

---

## Two goals

1. **Count** — how many unique birds appear in a video
2. **Identify** — what species each bird is

---

## Project layout

```
Bird Identification/
  data/
    raw_clips/     put your .mp4 video files here
    crops/         bird crop images saved here when using --save_crops

  datasets/
    video_reader.py    opens a video file, yields (frame, timestamp) pairs

  models/
    detector.py        YOLOv8 wrapper — finds birds in a single frame
    classifier.py      species identifier — crops bird and names species
                       (stub until Week 4: always returns "Unidentifiable")

  utils/
    counter.py         tracks unique birds across frames (avoids counting
                       the same bird many times)

  detect.py            COUNT birds in a video    ← run this
  identify.py          IDENTIFY species in a video ← run this
  requirements.txt
```

---

## Quick start

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Count birds in a video (downloads yolov8n.pt automatically on first run)
python detect.py --video data/raw_clips/my_clip.mp4

# 4. Watch live with bounding boxes
python detect.py --video data/raw_clips/my_clip.mp4 --show

# 5. Identify species + save a crop image per detection
python identify.py --video data/raw_clips/my_clip.mp4 --save_crops
```

---

## Model used

YOLOv8n (nano) from Ultralytics — pre-trained on COCO, knows class 14 = "bird".
Download happens automatically on first run (~6 MB).

For better accuracy on turbine footage: train on FBD-SV-2024.
See: https://github.com/Ziwei89/FBD-SV-2024_github
