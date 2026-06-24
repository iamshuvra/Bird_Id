# Bird Identification – Offshore Wind Turbine Monitor


I used this github repository and their dataset to run the code: [Github Repository](https://github.com/Ziwei89/FBOD) | [Dataset](https://github.com/Ziwei89/FBD-SV-2024_github)

## Prepare the Working Environment

First step is to active the environment: run 'source venv/bin/activate' when starting the project.

If you don't have an environement, create one and install all the files using cmd 'pip install requirements.txt'
In this file, we have all the required libraries mentioned to run the code. 


## Data Preprocessing
We have to do some data preprocessing before jumping into training. Run these three steps in order.

### Step 1 — Extract video frames
**Script:** `datasets/scripts/split_video_frames_for_object_detection.py`

Converts all `.mp4` videos into individual `.jpg` frames using OpenCV.

```bash
python datasets/scripts/split_video_frames_for_object_detection.py \
    --data_root_path=./data/FBD-SV-2024/
```

Output frames are saved to `data/FBD-SV-2024/images/train/` and `data/FBD-SV-2024/images/val/` with names like `bird_93_000006.jpg`.

### Step 2 — Generate annotation index txt files
**Script:** `datasets/continuous_image_annotation_frames_padding.py`

This script only reads the XML labels and produces a text index file — it does **not** convert videos or images. It must be run after Step 1 because it checks whether the required image files exist on disk.

```bash
python datasets/continuous_image_annotation_frames_padding.py \
    --data_root_path=./data/FBD-SV-2024/ \
    --input_img_num=5
```

**Output files** (written to `datasets/dataloader/`):
- `img_label_five_continuous_difficulty_train_raw.txt`
- `img_label_five_continuous_difficulty_val_raw.txt`

**Format of each line:**
```
bird_93_000006.jpg  134,559,164,584,0,0.625
      ↑                  ↑           ↑  ↑
 first frame of      bounding box  class confidence
 the 5-frame window  from MIDDLE   id    score
 (frame 000006)      frame(000008)
```

- `bird_93` = video 93 in the dataset
- `000006` = first frame of the 5-frame window `[6, 7, 8, 9, 10]`
- `134,559,164,584` = bounding box `(xmin, ymin, xmax, ymax)` of the bird in the **middle frame** (frame `000008`)
- `0` = class ID (`bird` is the only class, index 0)
- `0.625` = confidence score derived from the XML `<difficult>` tag

**Key parameter — `input_img_num=5`:** The model is temporal — it receives a window of 5 consecutive frames to detect the bird in the middle frame. Surrounding frames provide motion context, which helps detect tiny fast-moving birds that are hard to spot in a single frame.

**Difficulty → confidence score mapping:**

| `<difficult>` in XML | Confidence score |
|---|---|
| 0 (easy) | 0.875 |
| 1 (general) | 0.625 |
| 2 (hard) | 0.375 |
| 3 (very hard) | 0.125 |

Higher difficulty = lower confidence weight, so hard samples contribute less to the training loss.

**Padding:** If the 5-frame window runs off the start or end of a video, the dataloader inserts black (zero) frames in place of the missing ones at runtime. This script verifies that the frames that should exist actually do before writing a line.

### Step 3 — Shuffle the annotation txt files
**Script:** `datasets/dataloader/shuffle_txt_lines.py`

The raw txt files from Step 2 are ordered — all frames from one video appear together. Training on ordered data biases the model, so we shuffle before training.

```bash
python datasets/dataloader/shuffle_txt_lines.py \
    --input_img_num=5
```

**Output files** (written to `datasets/dataloader/`):
- `img_label_five_continuous_difficulty_train.txt`
- `img_label_five_continuous_difficulty_val.txt`

These final shuffled txt files are what the training script reads.





