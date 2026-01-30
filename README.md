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









