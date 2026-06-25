# Bird Identification - Offshore Wind Turbine Monitor


I used this github repository and their dataset to run the code: [Github Repository](https://github.com/Ziwei89/FBOD) | [Dataset](https://github.com/Ziwei89/FBD-SV-2024_github)

## Prepare the Working Environment

First step is to active the environment: run 'source venv/bin/activate' when starting the project.

If you don't have an environement, create one and install all the files using cmd 'pip install requirements.txt'
In this file, we have all the required libraries mentioned to run the code. 
```
torch>=2.1.0
torchvision>=0.16.0
opencv-python>=4.8.0
numpy>=1.24.0
tqdm>=4.65.0
matplotlib>=3.7.0
Pillow>=10.0.0
pycocotools>=2.0.7
imgaug>=0.4.0
```

## Data Preprocessing
We have to do some data preprocessing before jumping into training. Run these three steps in order.

### Step 1 - Extract video frames
**Script:** `datasets/scripts/split_video_frames_for_object_detection.py`

Converts all `.mp4` videos into individual `.jpg` frames using OpenCV.

```bash
python datasets/scripts/split_video_frames_for_object_detection.py \
    --data_root_path=./data/FBD-SV-2024/
```

Output frames are saved to `data/FBD-SV-2024/images/train/` and `data/FBD-SV-2024/images/val/` with names like `bird_93_000006.jpg`.

### Step 2 - Generate annotation index txt files
**Script:** `datasets/continuous_image_annotation_frames_padding.py`

This script only reads the XML labels and produces a text index file - it does **not** convert videos or images. It must be run after Step 1 because it checks whether the required image files exist on disk.

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

**Key parameter - `input_img_num=5`:** The model is temporal - it receives a window of 5 consecutive frames to detect the bird in the middle frame. Surrounding frames provide motion context, which helps detect tiny fast-moving birds that are hard to spot in a single frame.

**Difficulty → confidence score mapping:**

| `<difficult>` in XML | Confidence score |
|---|---|
| 0 (easy) | 0.875 |
| 1 (general) | 0.625 |
| 2 (hard) | 0.375 |
| 3 (very hard) | 0.125 |

Higher difficulty = lower confidence weight, so hard samples contribute less to the training loss.

**Padding:** If the 5-frame window runs off the start or end of a video, the dataloader inserts black (zero) frames in place of the missing ones at runtime. This script verifies that the frames that should exist actually do before writing a line.

### Step 3 - Shuffle the annotation txt files
**Script:** `datasets/dataloader/shuffle_txt_lines.py`

The raw txt files from Step 2 are ordered - all frames from one video appear together. Training on ordered data biases the model, so we shuffle before training.

```bash
python datasets/dataloader/shuffle_txt_lines.py \
    --input_img_num=5
```

**Output files** (written to `datasets/dataloader/`):
- `img_label_five_continuous_difficulty_train.txt`
- `img_label_five_continuous_difficulty_val.txt`

These final shuffled txt files are what the training script reads.

## Training

**Script:** `train.py`

Trains the FBOD temporal detection model. Run from the project root:

```bash
source venv/bin/activate
python train.py \
    --data_root_path=./data/FBD-SV-2024/ \
    --input_img_num=5 \
    --Batch_size=4 \
    --num_workers=2 \
    --end_Epoch=100 \
    --Add_name=run1
```

**Key parameters:**

| Parameter | Default | Notes |
|---|---|---|
| `--Batch_size` | 8 | Lower to 4 if GPU runs out of memory |
| `--num_workers` | 2 | Use 0-2 on WSL; higher values cause multiprocessing overhead |
| `--model_input_size` | `384_672` | H×W fed to model. Use `256_448` to reduce compute by ~55% |
| `--end_Epoch` | 100 | AP@50 metric only starts being measured from epoch 30 |
| `--Add_name` | `0816_1` | Label appended to the log folder name |

**Checkpoints** are saved to `logs/five/<config>/`:
- `last_checkpoint.pth` - saved after every epoch, previous one deleted (resume-friendly)
- `best_AP50_<score>_epoch<N>.pth` - saved whenever AP@50 improves
- `FB_object_detect_model.pth` - always the best model so far

Re-run the same command - training auto-detects `last_checkpoint.pth` and continues from the last completed epoch.

**GPU note** The model uses mixed precision (AMP/FP16) automatically when a GPU is available, which gives ~2-3× speedup on RTX cards. On an RTX 4050 Laptop GPU (6 GB VRAM) expect roughly 1.5-2.5 s/iter.

