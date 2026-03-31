# ultraprompt

**Ultraprompt** is a lightweight toolchain for creating **class-annotated instance segmentation data for YOLO** using **SAM3**.  
It lets you interactively add positive/negative points and class-tagged boxes, run SAM3 to generate instance masks, and export **YOLO segmentation (polygons) labels**. Use the GUI for fast labeling or the Python API in notebooks for batch workflows. SAM3 now allows for Promptable Concept Segmentation which streamlines manual labeling tasks significantly!

## Features

- 🔹 **SAM3 via Ultralytics** for box or text prompted conceptual segmentation
- 🔹 **Points (±) + boxes** as hints, per-box class assignment for SAM2 compatibility
- 🔹 **YOLO-Seg export** (`labels/*.txt` with normalized polygons)  
- 🔹 **Notebook-friendly core API** (no GUI required)  
---

## Quickstart (with `uv`)

> **Requirements**
>
> - Python **3.11** (project pins to 3.11)
> - [uv](https://github.com/astral-sh/uv) installed (e.g., `pipx install uv`)
> - (Optional) A compatible GPU + CUDA if you want accelerated SAM2

### 1) Get the code
Clone or download the repo and set up a virtual environment with UV
```bash
cd ultraprompt
uv venv
uv sync
uv pip install git+https://github.com/ultralytics/CLIP.git
```
or use TLS certificates if needed for corporate networks
```bash
uv venv --native-tls 
```

### 2) Activate it:
#### **Windows** (PowerShell)
```bash
.venv\Scripts\activate
```
#### **macOS/Linux**
```bash
source .venv/bin/activate
```

### 3) Launch the GUI
```bash
ultraprompt-gui
```
### Using the GUI
You can import custom class labels by loading a .txt file with class names on each line, for exmaple: create a file called classes.txt containing
```php-template
dog
cat
frog
...
```
The GUI will make these classes available for YOLO polygon labels from SAM2 box prompts.

### 4) Download the SAM3 or SAM2 model weights
You'll need to have your model weights somewhere, download SAM2 weights from here:
https://docs.ultralytics.com/models/sam-2/#how-to-use-sam-2-versatility-in-image-and-video-segmentation

and download SAM3 weights from here (request access is required): https://huggingface.co/facebook/sam3

### Using the core API (no GUI)
```python
from ultraprompt.core.sam_yolo_annotation import UltraSAM3, load_image_rgb

# Initialize SAM3 wrapper
sam = UltraSAM3()

# Load Ultralytics SAM3 weights
# device can be "auto", "cpu", or "cuda"
sam.load("path/to/sam3.pt", device="auto")

# Load and bind image
img = load_image_rgb("path/to/image.jpg")
sam.bind_image(img, image_path="path/to/image.jpg")

# --------------------------------------------------
# Visual prompting (points and/or boxes)
# --------------------------------------------------

# Point prompts
points = [[320, 180], [420, 220]]    # (x, y)
labels = [1, 0]                      # 1 = foreground, 0 = background

# Box prompts
boxes = [[100, 120, 300, 340]]       # (x0, y0, x1, y1)

# Run visual SAM3 inference
masks = sam.infer_visual(
    points=points,
    labels=labels,
    boxes=boxes,
    multimask_output=True,
)

print(f"Visual mode: {len(masks)} mask(s) returned")

# Each mask is a boolean (H, W) NumPy array
# masks[i][y, x] == True indicates foreground


# --------------------------------------------------
# Concept / semantic prompting (text and/or boxes)
# --------------------------------------------------

# Text concepts
concepts = ["person", "bus"]

# Optional: exemplar boxes that define the concept visually
exemplar_boxes = [
    [100, 120, 300, 340],
    [420, 160, 560, 420],
]

# Run concept segmentation
concept_masks = sam.infer_concept(
    text=concepts,          # list of strings or None
    exemplars=exemplar_boxes,  # list of boxes or None
)

print(f"Concept mode: {len(concept_masks)} mask(s) returned")

# --------------------------------------------------
# Segment-everything (no prompts)
# --------------------------------------------------

all_masks = sam.segment_everything(img, top_n=20)
print(f"Segment-everything: {len(all_masks)} mask(s)")
```
## Keeping SAM2 backwards compatibility with an alias.
```python
# UltraSAM2 is kept as an alias for backward compatibility
from ultraprompt.core.sam_yolo_annotation import UltraSAM2
sam = UltraSAM2()  # same as UltraSAM3
```


### Export format (YOLO-Seg)
One line per instance with normalized polygon coordinates
``` php-template
<class_id> x1 y1 x2 y2 ... xn yn
```
🔹class_id is 0..N-1 from your classes.txt (one class name per line).

🔹Coordinates are normalized to [0,1] by image width/height.













---
---

## YOLO Training Tab

This fork extends ultraprompt with a **YOLO Training Tab**, a
second tab in the GUI that takes your SAM3-annotated labels and
trains a YOLO instance segmentation model, all without leaving
the application.

The intended workflow is:
```
SAM3 Annotate tab  →  export YOLO labels
YOLO Training tab  →  build dataset → train → evaluate
```

### Additional dependencies

The YOLO Training Tab requires a few extra packages not in the
base install:
```
uv add pandas matplotlib ultralytics
```

### YOLO Training Tab features

* 🔹 **Dataset Builder**: point to your images folder and labels
  folder, set train/val/test split ratios, and the tab builds the
  full YOLO dataset folder structure and writes `data.yaml`
  automatically — auto-filling the path in the training config
* 🔹 **Device selection** : auto-detects CPU or GPU on startup;
  works on Windows PC (CPU) and Linux with CUDA GPU
* 🔹 **Model selection** : choose from yolo26n/s/m/l/x or yolov8
  variants; weights are auto-downloaded from Ultralytics on first
  use, or browse to a custom `.pt` file
* 🔹 **Hyperparameter fields** : all key training parameters
  pre-filled with defaults optimised by Ray Tune (150 trials,
  A100 GPU, RHEED segmentation dataset, 4 classes)
* 🔹 **Live training log** : coloured terminal streams YOLO output
  in real time showing epoch progress, losses, and metrics
* 🔹 **Live plots** : loss curves and mAP curves update every 30s
  during training without blocking the UI
* 🔹 **Confusion matrix** : displayed automatically in a dedicated
  tab when training completes
* 🔹 **Metrics table** : mAP50, mAP50-95, precision, and recall
  for both box and mask, updated each poll interval
* 🔹 **Load existing results** : browse to any past `results.csv`
  to view plots and metrics without retraining

### Platform notes

- **Windows**: `workers` is automatically set to `0` to prevent
  a Python multiprocessing crash. If training crashes immediately,
  try unchecking **Mixed precision AMP** in the Options section.
- **Linux / Mac**: `workers` defaults to `4` for faster data
  loading. Adjust in the hyperparameter fields if needed.
- **GPU**: select your GPU from the Device dropdown. If you get
  an out-of-memory error, reduce the batch size.
- **CPU**: works on any machine for testing, but training will
  be slow. Use a small image size (640) and batch size (2) for
  quick tests.

### Viewing results from past training runs

The **Load Existing results.csv** button lets you view plots and
metrics from any previously completed training run without
retraining:

1. Click **Load Existing results.csv**
2. Browse to the `results.csv` file inside your training output
   folder, for example:

   runs_final/yolo_run_1/results.csv

3. The Loss Curves, mAP Curves, and Metrics table are populated
   instantly
4. If a `confusion_matrix_normalized.png` exists in the same
   folder it is loaded automatically into the Confusion Matrix tab


### Default hyperparameters

Defaults come from Ray Tune Bayesian optimisation (150 trials,
yolo26n-seg, Reflection high energy electron difrraction (RHEED) dataset):

| Parameter  | Default  | Description              |
|------------|----------|--------------------------|
| epochs     | 200      | training epochs          |
| imgsz      | 1280     | input image size         |
| batch      | 8        | batch size (reduce if OOM)|
| cls        | 0.391    | class loss weight        |
| conf       | 0.276    | confidence threshold     |
| dropout    | 0.248    | regularization           |
| mask_ratio | 4        | mask resolution divisor  |
| lr0        | 0.000503 | initial learning rate    |
| box        | 11.779   | box loss weight          |

These achieved mAP50-95(M) = 0.3485 versus a manual baseline
of 0.1761 on the RHEED dataset.
### Typical usage

1. Use the **SAM3 Annotate** tab to label your images and export
   YOLO `.txt` label files
2. Switch to the **YOLO Training** tab
3. In **Dataset Builder**: select your images folder, labels
   folder, and output folder then click **Build Dataset**
4. The `data.yaml` path is auto-filled in the training config
5. Select your model, adjust hyperparameters if needed
6. Click **Start Training** then watch the live log and plots
7. When complete, best weights path is shown in a popup


