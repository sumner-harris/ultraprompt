# ultraprompt

**Ultraprompt** is a lightweight toolchain for creating **class-annotated instance segmentation data for YOLO** using **SAM2**.  
It lets you interactively add positive/negative points and class-tagged boxes, run SAM2 to generate instance masks, and export **YOLO segmentation (polygons) labels**. Use the GUI for fast labeling or the Python API in notebooks for batch workflows.

## Features

- 🔹 **SAM2 via Ultralytics** for prompted & automatic segmentation  
- 🔹 **Points (±) + boxes** as hints, per-box class assignment  
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

### Using the core API (no GUI)
```python
from ultraprompt.core.sam_yolo_annotation import UltraSAM2, load_image_rgb

sam2 = UltraSAM2()
sam2.load("path/to/sam2_weights.pt", device="auto")  # "cpu" or "cuda" also ok

img = load_image_rgb("path/to/image.jpg")
sam2.bind_image(img)

# Prompts (examples)
points = [[320, 180], [420, 220]]   # x,y
labels = [1, 0]                      # 1=foreground, 0=background
boxes  = [[100, 120, 300, 340]]      # x0,y0,x1,y1

masks = sam2.infer(points=points, labels=labels, boxes=boxes)  # list of boolean (H,W) arrays
print(f"{len(masks)} mask(s)")
```

### Export format (YOLO-Seg)
One line per instance with normalized polygon coordinates
``` php-template
<class_id> x1 y1 x2 y2 ... xn yn
```
🔹class_id is 0..N-1 from your classes.txt (one class name per line).

🔹Coordinates are normalized to [0,1] by image width/height.



