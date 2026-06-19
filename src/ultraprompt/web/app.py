
from __future__ import annotations

import base64
import csv
import io
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field

from ultraprompt.core.convert_scientific_tiffs import convert_if_required, needs_uint8_conversion
from ultraprompt.core.image_preprocessing import ImagePreprocessSettings, preprocess_image_rgb
from ultraprompt.core.sam_yolo_annotation import UltraSAM3, load_image_rgb, mask_to_polygons, write_yolo_seg

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
FILTERS = [
    {"label": "None", "value": "none"},
    {"label": "Gaussian", "value": "gaussian"},
    {"label": "Median", "value": "median"},
    {"label": "Bilateral", "value": "bilateral"},
    {"label": "DoG Fine", "value": "dog_fine"},
    {"label": "DoG Medium", "value": "dog_medium"},
    {"label": "DoG Coarse", "value": "dog_coarse"},
    {"label": "Sharpen", "value": "sharpen"},
]


class OpenFolderRequest(BaseModel):
    image_dir: str
    out_dir: Optional[str] = None


class ClassesRequest(BaseModel):
    classes: Optional[list[str]] = None
    classes_path: Optional[str] = None


class LoadWeightsRequest(BaseModel):
    weights: str
    device: str = "auto"


class PointPrompt(BaseModel):
    x: float
    y: float
    label: int


class BoxPrompt(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float
    class_id: Optional[int] = 0


class SegmentRequest(BaseModel):
    index: int
    pixel_scale: int = 1
    filter_name: str = "none"
    mode: str
    class_id: int = 0
    concept_text: Optional[str] = ""
    points: list[PointPrompt] = Field(default_factory=list)
    boxes: list[BoxPrompt] = Field(default_factory=list)


class SaveYoloRequest(BaseModel):
    index: int
    pixel_scale: int = 1
    filter_name: str = "none"
    label_png: str


class NewFolderRequest(BaseModel):
    parent_dir: str
    name: str


class BuildDatasetRequest(BaseModel):
    image_dir: str
    labels_dir: str
    output_dir: str
    task: str = "segment"
    classes: list[str] = Field(default_factory=lambda: ["object"])
    train_pct: int = 80
    val_pct: int = 10
    use_test_split: bool = True
    seed: int = 0


class YoloTrainRequest(BaseModel):
    data_yaml: str
    output_dir: str
    run_name: str
    task: str = "segment"
    model: str
    auto_weights: bool = True
    custom_weights: Optional[str] = None
    device: str = "cpu"
    epochs: int = 100
    imgsz: int = 640
    batch: int = 4
    patience: int = 20
    cls: float = 0.5
    conf: float = 0.25
    dropout: float = 0.0
    mask_ratio: int = 4
    auto_optimize: bool = True
    optimizer: str = "AdamW"
    lr0: float = 0.01
    box: float = 7.5
    fliplr: float = 0.5
    flipud: float = 0.0
    scale: float = 0.5
    translate: float = 0.1
    mosaic: float = 0.0
    mixup: float = 0.0
    workers: int = 0
    seed: int = 0
    save: bool = True
    plots: bool = True
    amp: bool = True
    verbose: bool = True
    save_period: bool = False


class LoadResultsRequest(BaseModel):
    results_csv: str


class YoloInferRequest(BaseModel):
    image_dir: str
    output_dir: Optional[str] = None
    task: str = "segment"
    model: str
    auto_weights: bool = True
    custom_weights: Optional[str] = None
    device: str = "cpu"
    conf: float = 0.25
    imgsz: int = 1280
    scope: str = "all"
    image_name: Optional[str] = None
    save_overlays: bool = True
    save_predictions: bool = True


class YoloInferInitRequest(BaseModel):
    task: str = "segment"
    model: str
    auto_weights: bool = True
    custom_weights: Optional[str] = None
    device: str = "cpu"


@dataclass
class InferRecord:
    name: str
    base_png: Path
    overlay_png: Path
    summary: str
    task: str
    boxes: int
    masks: int
    classes: list[str] = field(default_factory=list)
    saved_overlay: Optional[str] = None
    saved_prediction: Optional[str] = None
    saved_boxes: Optional[str] = None


@dataclass
class AppState:
    image_dir: Optional[Path] = None
    out_dir: Optional[Path] = None
    image_paths: list[Path] = field(default_factory=list)
    converted_image_sources: dict[Path, Path] = field(default_factory=dict)
    classes: list[str] = field(default_factory=lambda: ["object"])
    sam: UltraSAM3 = field(default_factory=UltraSAM3)
    sam_weights: Optional[Path] = None
    sam_device: str = "auto"
    train_proc: Optional[subprocess.Popen] = None
    train_status: str = "idle"
    train_error: Optional[str] = None
    train_log_path: Optional[Path] = None
    train_results_path: Optional[Path] = None
    infer_results: list[InferRecord] = field(default_factory=list)
    infer_task: str = "segment"
    infer_image_dir: Optional[Path] = None
    infer_output_dir: Optional[Path] = None
    infer_model_key: Optional[tuple[str, str, str]] = None
    infer_model: Any = None


STATE = AppState()
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
app = FastAPI(title="Ultraprompt Web")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _norm_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _runtime_root() -> Path:
    candidates: list[Path] = []
    if STATE.out_dir is not None:
        candidates.append(STATE.out_dir / ".ultraprompt_runtime")
    if STATE.image_dir is not None:
        candidates.append(STATE.image_dir / ".ultraprompt_runtime")
    env_tmp = os.environ.get("TMPDIR")
    if env_tmp:
        candidates.append(Path(env_tmp).expanduser())
    candidates.append(BASE_DIR / "_runtime")
    seen: set[str] = set()
    for cand in candidates:
        key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        try:
            parent = cand if cand.exists() else cand.parent
            usage = shutil.disk_usage(parent)
            if usage.free < 64 * 1024 * 1024:
                continue
            cand.mkdir(parents=True, exist_ok=True)
            probe = cand / ".write_test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            os.environ["TMPDIR"] = str(cand)
            return cand
        except Exception:
            continue
    raise RuntimeError("No writable runtime directory available. Set TMPDIR or choose a writable image/output folder.")


def _runtime_dir(name: str) -> Path:
    out = _runtime_root() / name
    out.mkdir(parents=True, exist_ok=True)
    return out


def _record(index: int) -> tuple[Path, Path]:
    if index < 0 or index >= len(STATE.image_paths):
        raise HTTPException(status_code=404, detail=f"Image index out of range: {index}")
    display = STATE.image_paths[index]
    source = STATE.converted_image_sources.get(display, display)
    return display, source


def _record_stem(index: int) -> str:
    display, source = _record(index)
    return source.stem if source else display.stem


def _settings(pixel_scale: int, filter_name: str) -> ImagePreprocessSettings:
    return ImagePreprocessSettings(pixel_scale=max(1, int(pixel_scale)), filter_name=(filter_name or "none"))


def _preprocessed(index: int, pixel_scale: int, filter_name: str) -> dict[str, Any]:
    display, source = _record(index)
    base = load_image_rgb(source)
    settings = _settings(pixel_scale, filter_name)
    view = preprocess_image_rgb(base, settings)
    source_h, source_w = base.shape[:2]
    view_h, view_w = view.shape[:2]
    cache_dir = _runtime_dir("views")
    key = f"{source.stem}_px{settings.pixel_scale}_{settings.filter_name}_{view_w}x{view_h}_{source.stat().st_mtime_ns}.png"
    png_path = cache_dir / key
    if not png_path.exists():
        Image.fromarray(view).save(png_path)
    return {
        "display_path": display,
        "source_path": source,
        "image": view,
        "png_path": png_path,
        "source_size": [int(source_w), int(source_h)],
        "view_size": [int(view_w), int(view_h)],
        "pixel_scale": settings.pixel_scale,
        "filter_name": settings.filter_name,
    }


def _sam_image_path(index: int, pixel_scale: int, filter_name: str) -> Path:
    return _preprocessed(index, pixel_scale, filter_name)["png_path"]


def _class_names() -> list[str]:
    return STATE.classes or ["object"]


def _state_payload() -> dict[str, Any]:
    return {
        "image_dir": str(STATE.image_dir) if STATE.image_dir else "",
        "out_dir": str(STATE.out_dir) if STATE.out_dir else "",
        "count": len(STATE.image_paths),
        "images": [p.name for p in STATE.image_paths],
        "classes": _class_names(),
        "filters": FILTERS,
    }


def _mask_b64(mask: np.ndarray) -> str:
    arr = (np.asarray(mask).astype(np.uint8) * 255)
    buf = io.BytesIO()
    Image.fromarray(arr, mode="L").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _png_data_url(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8), mode="L").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _decode_label_png(data_url: str) -> np.ndarray:
    if "," in data_url:
        _, payload = data_url.split(",", 1)
    else:
        payload = data_url
    raw = base64.b64decode(payload)
    arr = np.array(Image.open(io.BytesIO(raw)).convert("L"), dtype=np.uint8)
    return arr.astype(np.int16) - 1


def _resize_mask_nearest(arr: np.ndarray, width: int, height: int) -> np.ndarray:
    if arr.shape == (height, width):
        return arr
    im = Image.fromarray(arr)
    return np.array(im.resize((width, height), resample=Image.Resampling.NEAREST))


def _saved_mask_path(index: int) -> Path:
    if STATE.out_dir is None:
        raise HTTPException(status_code=400, detail="Output directory not set.")
    return STATE.out_dir / "masks" / f"{_record_stem(index)}.png"


def _saved_label_path(index: int) -> Path:
    if STATE.out_dir is None:
        raise HTTPException(status_code=400, detail="Output directory not set.")
    return STATE.out_dir / "labels" / f"{_record_stem(index)}.txt"


def _record_has_saved_annotation(index: int) -> bool:
    if STATE.out_dir is None:
        return False
    try:
        return _saved_mask_path(index).exists() or _saved_label_path(index).exists()
    except Exception:
        return False


def _render_saved_polygons(label_path: Path, source_w: int, source_h: int) -> np.ndarray:
    canvas = Image.new("L", (source_w, source_h), 0)
    draw = ImageDraw.Draw(canvas)
    if not label_path.exists():
        return np.zeros((source_h, source_w), dtype=np.uint8)
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw.strip().split()
        if len(parts) < 7 or len(parts[1:]) % 2 != 0:
            continue
        cls = _safe_int(parts[0], 0)
        pts = []
        coords = [float(v) for v in parts[1:]]
        for x, y in zip(coords[0::2], coords[1::2]):
            pts.append((max(0, min(source_w - 1, x * source_w)), max(0, min(source_h - 1, y * source_h))))
        if len(pts) >= 3:
            draw.polygon(pts, fill=cls + 1)
    return np.array(canvas, dtype=np.uint8)


def _saved_label_map_for_view(index: int, pixel_scale: int, filter_name: str) -> tuple[np.ndarray, str | None]:
    info = _preprocessed(index, pixel_scale, filter_name)
    view_w, view_h = info["view_size"]
    source_w, source_h = info["source_size"]
    out = np.zeros((view_h, view_w), dtype=np.uint8)
    if STATE.out_dir is None:
        return out, None
    mask_path = _saved_mask_path(index)
    if mask_path.exists():
        native = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
        if native.shape != (source_h, source_w):
            native = _resize_mask_nearest(native, source_w, source_h)
        mapped = np.zeros_like(native, dtype=np.uint8)
        valid = native != 255
        mapped[valid] = native[valid] + 1
        mapped = _resize_mask_nearest(mapped, view_w, view_h)
        return mapped, "semantic mask"
    label_path = _saved_label_path(index)
    if label_path.exists():
        native = _render_saved_polygons(label_path, source_w, source_h)
        native = _resize_mask_nearest(native, view_w, view_h)
        return native, "YOLO polygons"
    return out, None


def _label_map_data_url(label_map: np.ndarray) -> str:
    return _png_data_url(label_map)


def _semantic_mask_from_label_map(label_map: np.ndarray) -> np.ndarray:
    semantic = np.full(label_map.shape, 255, dtype=np.uint8)
    valid = label_map >= 0
    semantic[valid] = np.clip(label_map[valid], 0, 254).astype(np.uint8)
    return semantic


def _browse_roots() -> list[dict[str, str]]:
    roots: list[tuple[str, Path]] = []
    home = Path.home()
    roots.append(("Home", home))
    roots.append(("Workspace", BASE_DIR.parent.parent.parent))
    if STATE.image_dir is not None:
        roots.append(("Images", STATE.image_dir))
    if STATE.out_dir is not None:
        roots.append(("Output", STATE.out_dir))
    roots.append(("Root", Path("/")))
    seen: set[str] = set()
    out: list[dict[str, str]] = []
    for name, path in roots:
        p = str(path.resolve())
        if p in seen or not path.exists():
            continue
        seen.add(p)
        out.append({"name": name, "path": p})
    return out


def _safe_browse_path(path: Optional[str]) -> Path:
    if path:
        return _norm_path(path)
    if STATE.image_dir is not None:
        return STATE.image_dir
    return Path.home().resolve()


def _file_visible(path: Path) -> bool:
    return not path.name.startswith(".")


def _entry_selectable(path: Path, purpose: str) -> bool:
    if purpose in {"image_dir", "out_dir"}:
        return path.is_dir()
    suffix = path.suffix.lower()
    if purpose == "classes":
        return path.is_file() and suffix == ".txt"
    if purpose == "weights":
        return path.is_file() and suffix in {".pt", ".pth"}
    if purpose == "yaml":
        return path.is_file() and suffix in {".yaml", ".yml"}
    if purpose == "csv":
        return path.is_file() and suffix == ".csv"
    return path.is_dir()


def _yolo_task_name(task: str) -> str:
    return "semantic" if task == "semantic" else "segment"


def _platform_label() -> str:
    return f"{platform.system()} {platform.release()}"


def _safe_workers(workers: int) -> int:
    if platform.system().lower().startswith("win"):
        return 0
    return max(0, int(workers))


def _reset_yolo_split_dir(root: Path) -> None:
    if root.exists():
        for child in root.iterdir():
            if child.name.startswith(".nfs"):
                continue
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
    root.mkdir(parents=True, exist_ok=True)


def _write_yolo_training_image(src: Path, dst: Path) -> bool:
    img = load_image_rgb(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(dst)
    return src.suffix.lower() in {".tif", ".tiff"}


def _load_yaml_data(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _dataset_root_from_yaml(data_yaml: Path) -> Path:
    data = _load_yaml_data(data_yaml)
    path = data.get("path")
    if not path:
        raise HTTPException(status_code=400, detail=f"Missing path in {data_yaml}")
    return _norm_path(path)


def _clear_yolo_dataset_caches(dataset_root: Path) -> None:
    for cache in dataset_root.rglob("*.cache"):
        cache.unlink(missing_ok=True)


def _validate_segment_dataset_data(data_yaml: Path) -> None:
    data = _load_yaml_data(data_yaml)
    nc = int(data.get("nc", 0))
    if nc <= 0:
        raise HTTPException(status_code=400, detail="Segment dataset has no classes configured.")
    root = _dataset_root_from_yaml(data_yaml)
    _clear_yolo_dataset_caches(root)
    errors: list[str] = []
    for split in ["train", "val", "test"]:
        rel = data.get(split)
        if not rel:
            continue
        split_dir = root / rel
        labels_dir = split_dir.as_posix().replace("/images/", "/labels/")
        for txt in Path(labels_dir).glob("*.txt"):
            for line in txt.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if not parts:
                    continue
                cls = _safe_int(parts[0], -1)
                if cls < 0 or cls >= nc:
                    errors.append(f"{txt.name}: class id {cls} exceeds 0-{nc-1}")
                    break
    if errors:
        raise HTTPException(status_code=400, detail="Segment dataset class mismatch. " + "; ".join(errors[:20]))


def _validate_semantic_dataset_data(data_yaml: Path) -> None:
    data = _load_yaml_data(data_yaml)
    nc = int(data.get("nc", 0))
    if nc <= 0:
        raise HTTPException(status_code=400, detail="Semantic dataset has no classes configured.")
    root = _dataset_root_from_yaml(data_yaml)
    _clear_yolo_dataset_caches(root)
    errors: list[str] = []
    for split in ["train", "val", "test"]:
        rel = data.get(split)
        if not rel:
            continue
        split_dir = root / rel
        masks_dir = split_dir.as_posix().replace("/images/", "/masks/")
        for mask_path in Path(masks_dir).glob("*.png"):
            arr = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
            classes = sorted(int(v) for v in np.unique(arr) if int(v) != 255)
            if any(v < 0 or v >= nc for v in classes):
                errors.append(f"{mask_path.name}: class ids {classes} exceed 0-{nc-1} with 255 as ignore")
    if errors:
        raise HTTPException(status_code=400, detail="Semantic dataset class mismatch. " + "; ".join(errors[:20]))


def _validate_yolo_dataset(data_yaml: Path, task: str) -> None:
    if task == "semantic":
        _validate_semantic_dataset_data(data_yaml)
    else:
        _validate_segment_dataset_data(data_yaml)


def _read_results(results_csv: Path) -> dict[str, Any]:
    if not results_csv.exists():
        return {"exists": False}
    with open(results_csv, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {"exists": True, "path": str(results_csv), "rows": 0, "metrics": {}, "series": {}}
    series: dict[str, list[float]] = {}
    metrics: dict[str, float] = {}
    for row in rows:
        for key, value in row.items():
            if value in (None, ""):
                continue
            try:
                num = float(value)
            except Exception:
                continue
            series.setdefault(key, []).append(num)
    for key, values in series.items():
        if values:
            metrics[key] = values[-1]
    return {"exists": True, "path": str(results_csv), "rows": len(rows), "metrics": metrics, "series": series}


def _resolve_yolo_weights(task: str, model: str, auto_weights: bool, custom_weights: Optional[str]) -> str:
    if auto_weights:
        return model
    if not custom_weights:
        raise HTTPException(status_code=400, detail="Custom weights path is required when auto weights is off.")
    path = _norm_path(custom_weights)
    if not path.exists():
        raise HTTPException(status_code=400, detail=f"Weights not found: {path}")
    return str(path)


def _save_overlay_png(arr_bgr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(arr_bgr)
    rgb = arr[..., ::-1] if arr.ndim == 3 and arr.shape[-1] == 3 else arr
    Image.fromarray(rgb.astype(np.uint8)).save(path)


def _save_segment_outputs(result: Any, out_dir: Path, stem: str) -> tuple[Optional[str], Optional[str]]:
    labels_dir = out_dir / "labels"
    boxes_dir = out_dir / "boxes"
    labels_dir.mkdir(parents=True, exist_ok=True)
    boxes_dir.mkdir(parents=True, exist_ok=True)
    txt_path = labels_dir / f"{stem}.txt"
    txt_path.unlink(missing_ok=True)
    result.save_txt(txt_path)
    csv_path = boxes_dir / f"{stem}.csv"
    wrote_boxes = False
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x0", "y0", "x1", "y1", "conf", "class_id", "class_name"])
        if result.boxes is not None:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                xyxy = box.xyxy[0].tolist()
                conf = float(box.conf[0]) if hasattr(box, "conf") else 0.0
                cls_id = int(box.cls[0]) if hasattr(box, "cls") else 0
                writer.writerow([*xyxy, conf, cls_id, result.names.get(cls_id, str(cls_id))])
                wrote_boxes = True
    txt_out = str(txt_path) if txt_path.exists() and txt_path.stat().st_size > 0 else None
    if txt_out is None:
        txt_path.unlink(missing_ok=True)
    csv_out = str(csv_path) if wrote_boxes else None
    if csv_out is None:
        csv_path.unlink(missing_ok=True)
    return txt_out, csv_out


def _save_semantic_output(result: Any, out_dir: Path, stem: str) -> Optional[str]:
    if result.semantic_mask is None:
        return None
    masks_dir = out_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    mask_path = masks_dir / f"{stem}.png"
    sem = result.semantic_mask.data
    if hasattr(sem, "cpu"):
        sem = sem.cpu().numpy()
    Image.fromarray(np.asarray(sem).astype(np.uint8), mode="L").save(mask_path)
    return str(mask_path)


def _infer_source_paths(image_dir: Path) -> list[Path]:
    return [p for p in sorted(image_dir.iterdir()) if p.is_file() and p.suffix.lower() in IMG_EXTS]


def _infer_source_record(image_dir: Path, image_name: str) -> Path:
    image_path = image_dir / image_name
    if not image_path.is_file() or image_path.suffix.lower() not in IMG_EXTS:
        raise HTTPException(status_code=404, detail=f"Inference source image not found: {image_name}")
    return image_path


def _infer_model_cache_key(task: str, weights: str, device: str) -> tuple[str, str, str]:
    return (_yolo_task_name(task), str(weights), str(device))


def _load_infer_model(task: str, model: str, auto_weights: bool, custom_weights: Optional[str], device: str):
    weights = _resolve_yolo_weights(task, model, auto_weights, custom_weights)
    key = _infer_model_cache_key(task, weights, device)
    if STATE.infer_model is not None and STATE.infer_model_key == key:
        return STATE.infer_model, weights, False
    try:
        from ultralytics import YOLO
        loaded = YOLO(weights)
        STATE.infer_model = loaded
        STATE.infer_model_key = key
        return loaded, weights, True
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _runtime_infer_source_png(image_path: Path) -> Path:
    out = _runtime_dir('infer_source') / f"{image_path.stem}.png"
    if not out.exists() or out.stat().st_mtime_ns < image_path.stat().st_mtime_ns:
        Image.fromarray(load_image_rgb(image_path)).save(out)
    return out


def _infer_record(result: Any, image_path: Path, task: str, runtime_dir: Path, save_dir: Optional[Path], save_overlays: bool, save_predictions: bool) -> InferRecord:
    base_dir = runtime_dir / "base"
    overlay_dir = runtime_dir / "overlay"
    base_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    base_png = base_dir / f"{image_path.stem}.png"
    overlay_png = overlay_dir / f"{image_path.stem}.png"
    Image.fromarray(load_image_rgb(image_path)).save(base_png)
    overlay = result.plot(conf=True, labels=True, boxes=True, masks=True)
    _save_overlay_png(overlay, overlay_png)

    saved_overlay = None
    saved_prediction = None
    saved_boxes = None
    if save_dir is not None:
        if save_overlays:
            out_overlay = save_dir / "overlays" / f"{image_path.stem}.png"
            _save_overlay_png(overlay, out_overlay)
            saved_overlay = str(out_overlay)
        if save_predictions:
            if task == "semantic":
                saved_prediction = _save_semantic_output(result, save_dir, image_path.stem)
            else:
                saved_prediction, saved_boxes = _save_segment_outputs(result, save_dir, image_path.stem)

    classes: list[str] = []
    boxes_count = 0
    masks_count = 0
    if task == "semantic":
        if result.semantic_mask is not None:
            sem = result.semantic_mask.data
            if hasattr(sem, "cpu"):
                sem = sem.cpu().numpy()
            ids = [int(v) for v in np.unique(sem) if int(v) != 255]
            classes = [result.names.get(v, str(v)) for v in ids]
            masks_count = len(ids)
    else:
        if result.boxes is not None:
            boxes = result.boxes.cpu().numpy()
            boxes_count = len(boxes)
            ids = sorted({int(v) for v in boxes.cls.tolist()}) if hasattr(boxes, "cls") else []
            classes = [result.names.get(v, str(v)) for v in ids]
        if result.masks is not None:
            masks_count = len(result.masks)

    summary = (result.verbose() or "").strip().strip(",") or f"{task} inference"
    return InferRecord(
        name=image_path.name,
        base_png=base_png,
        overlay_png=overlay_png,
        summary=summary,
        task=task,
        boxes=boxes_count,
        masks=masks_count,
        classes=classes,
        saved_overlay=saved_overlay,
        saved_prediction=saved_prediction,
        saved_boxes=saved_boxes,
    )


def _append_train_log(text: str) -> None:
    if STATE.train_log_path is None:
        return
    with open(STATE.train_log_path, "a", encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")


@app.get("/")
def root() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/state")
def api_state() -> dict[str, Any]:
    return _state_payload()


@app.post("/api/open-folder")
def api_open_folder(req: OpenFolderRequest) -> dict[str, Any]:
    image_dir = _norm_path(req.image_dir)
    if not image_dir.is_dir():
        raise HTTPException(status_code=400, detail=f"Image directory not found: {image_dir}")
    raw_paths = [p for p in sorted(image_dir.iterdir()) if p.is_file() and p.suffix.lower() in IMG_EXTS]
    if not raw_paths:
        raise HTTPException(status_code=400, detail=f"No supported image files found in: {image_dir}")

    out_dir = _norm_path(req.out_dir) if req.out_dir else None
    STATE.image_dir = image_dir
    STATE.out_dir = out_dir
    STATE.image_paths = []
    STATE.converted_image_sources.clear()

    converted = 0
    conversion_errors: list[str] = []
    conversion_dir = image_dir.parent / f"{image_dir.name}_uint8"
    for src in raw_paths:
        try:
            display, did_convert = convert_if_required(src, conversion_dir)
        except Exception as e:
            display = src
            did_convert = False
            conversion_errors.append(f"{src.name}: {e}")
        STATE.image_paths.append(display)
        if did_convert:
            STATE.converted_image_sources[display] = src
            converted += 1

    payload = _state_payload()
    payload["converted"] = converted
    payload["conversion_errors"] = conversion_errors
    payload["existing_outputs"] = sum(1 for i in range(len(STATE.image_paths)) if _record_has_saved_annotation(i))
    return payload


@app.post("/api/classes")
def api_classes(req: ClassesRequest) -> dict[str, Any]:
    names: list[str]
    if req.classes_path:
        path = _norm_path(req.classes_path)
        if not path.exists():
            raise HTTPException(status_code=400, detail=f"classes.txt not found: {path}")
        names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        names = [line.strip() for line in (req.classes or []) if line.strip()]
    if not names:
        raise HTTPException(status_code=400, detail="No class names provided.")
    STATE.classes = names
    return _state_payload()


@app.post("/api/load-weights")
def api_load_weights(req: LoadWeightsRequest) -> dict[str, Any]:
    weights = _norm_path(req.weights)
    if not weights.exists():
        raise HTTPException(status_code=400, detail=f"Weights not found: {weights}")
    try:
        STATE.sam.load(weights, device=req.device)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    STATE.sam_weights = weights
    STATE.sam_device = req.device
    return {"ok": True, "weights": str(weights), "device": req.device}


@app.get("/api/image/{index}/info")
def api_image_info(index: int, pixel_scale: int = Query(1), filter_name: str = Query("none")) -> dict[str, Any]:
    info = _preprocessed(index, pixel_scale, filter_name)
    return {
        "name": info["source_path"].name,
        "display_name": info["display_path"].name,
        "source_size": info["source_size"],
        "view_size": info["view_size"],
        "pixel_scale": info["pixel_scale"],
        "filter_name": info["filter_name"],
    }


@app.get("/api/image/{index}/png")
def api_image_png(index: int, pixel_scale: int = Query(1), filter_name: str = Query("none")) -> FileResponse:
    return FileResponse(_preprocessed(index, pixel_scale, filter_name)["png_path"])


@app.get("/api/image/{index}/saved-label")
def api_saved_label(index: int, pixel_scale: int = Query(1), filter_name: str = Query("none")) -> dict[str, Any]:
    label_map, source = _saved_label_map_for_view(index, pixel_scale, filter_name)
    return {"exists": bool(source), "source": source, "label_png": _label_map_data_url(label_map)}


@app.post("/api/segment")
def api_segment(req: SegmentRequest) -> dict[str, Any]:
    if STATE.sam_weights is None:
        raise HTTPException(status_code=400, detail="Load SAM weights first.")
    info = _preprocessed(req.index, req.pixel_scale, req.filter_name)
    image = info["image"]
    image_path = _sam_image_path(req.index, req.pixel_scale, req.filter_name)
    try:
        STATE.sam.bind_image(image, image_path=image_path)
        if req.mode == "points":
            pts = [[float(p.x), float(p.y)] for p in req.points]
            labs = [int(p.label) for p in req.points]
            masks = STATE.sam.infer_visual(points=pts or None, labels=labs or None, boxes=None, multimask_output=True)
        elif req.mode == "boxes":
            boxes = [[float(b.x0), float(b.y0), float(b.x1), float(b.y1)] for b in req.boxes]
            masks = STATE.sam.infer_visual(points=None, labels=None, boxes=boxes, multimask_output=True)
        elif req.mode == "concept":
            boxes = [[float(b.x0), float(b.y0), float(b.x1), float(b.y1)] for b in req.boxes]
            text = [part.strip() for part in (req.concept_text or "").split(",") if part.strip()]
            masks = STATE.sam.infer_concept(text=text or None, exemplars=boxes or None)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported mode: {req.mode}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {
        "masks": [{"png": _mask_b64(mask), "class_id": req.class_id} for mask in masks],
        "count": len(masks),
    }


@app.post("/api/save-yolo")
def api_save_yolo(req: SaveYoloRequest) -> dict[str, Any]:
    if STATE.out_dir is None:
        raise HTTPException(status_code=400, detail="Set an output directory first.")
    info = _preprocessed(req.index, req.pixel_scale, req.filter_name)
    source_w, source_h = info["source_size"]
    label_map = _decode_label_png(req.label_png)
    label_map = _resize_mask_nearest(label_map.astype(np.int16), source_w, source_h)

    masks_dir = STATE.out_dir / "masks"
    labels_dir = STATE.out_dir / "labels"
    masks_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    stem = _record_stem(req.index)
    semantic = _semantic_mask_from_label_map(label_map)
    mask_path = masks_dir / f"{stem}.png"
    Image.fromarray(semantic, mode="L").save(mask_path)

    polygons: list[np.ndarray] = []
    class_ids: list[int] = []
    for cls_id in sorted(int(v) for v in np.unique(label_map) if int(v) >= 0):
        class_mask = (label_map == cls_id).astype(np.uint8)
        polys = mask_to_polygons(class_mask, simplify_eps=1.5, min_area=4.0)
        for poly in polys:
            polygons.append(poly)
            class_ids.append(cls_id)
    label_path = labels_dir / f"{stem}.txt"
    write_yolo_seg(label_path, polygons, class_ids, source_w, source_h)
    return {"ok": True, "polygons": len(polygons), "label_path": str(label_path), "mask_path": str(mask_path)}


@app.get("/api/browse")
def api_browse(path: Optional[str] = None, purpose: str = Query("image_dir")) -> dict[str, Any]:
    current = _safe_browse_path(path)
    if current.is_file():
        current = current.parent
    if not current.exists():
        current = current.parent if current.parent.exists() else Path.home().resolve()
    entries = []
    try:
        children = sorted([p for p in current.iterdir() if _file_visible(p)], key=lambda p: (not p.is_dir(), p.name.lower()))
    except PermissionError:
        children = []
    for child in children:
        entries.append({
            "name": child.name,
            "path": str(child.resolve()),
            "is_dir": child.is_dir(),
            "selectable": _entry_selectable(child, purpose),
        })
    return {
        "path": str(current.resolve()),
        "parent": str(current.parent.resolve()) if current.parent != current else None,
        "roots": _browse_roots(),
        "entries": entries,
        "current_selectable": _entry_selectable(current, purpose),
    }


@app.post("/api/mkdir")
def api_mkdir(req: NewFolderRequest) -> dict[str, Any]:
    parent = _norm_path(req.parent_dir)
    if not parent.is_dir():
        raise HTTPException(status_code=400, detail=f"Parent directory not found: {parent}")
    name = req.name.strip()
    if not name or any(ch in name for ch in ("/", "\\")):
        raise HTTPException(status_code=400, detail="Invalid folder name.")
    path = parent / name
    path.mkdir(parents=False, exist_ok=True)
    return {"path": str(path), "name": name}


@app.get("/api/yolo/devices")
def api_yolo_devices() -> dict[str, Any]:
    devices = ["cpu"]
    try:
        import torch
        if torch.cuda.is_available():
            for idx in range(torch.cuda.device_count()):
                devices.append(f"{idx} cuda:{torch.cuda.get_device_name(idx)}")
    except Exception:
        pass
    return {"devices": devices, "platform": _platform_label()}


@app.post("/api/yolo/build-dataset")
def api_yolo_build_dataset(req: BuildDatasetRequest) -> dict[str, Any]:
    image_dir = _norm_path(req.image_dir)
    labels_dir = _norm_path(req.labels_dir)
    output_dir = _norm_path(req.output_dir)
    if not image_dir.is_dir():
        raise HTTPException(status_code=400, detail=f"Image directory not found: {image_dir}")
    if not labels_dir.is_dir():
        raise HTTPException(status_code=400, detail=f"Annotation directory not found: {labels_dir}")
    classes = [c.strip() for c in req.classes if c.strip()]
    if not classes:
        raise HTTPException(status_code=400, detail="Provide at least one class name.")
    if req.use_test_split:
        if req.train_pct + req.val_pct >= 100:
            raise HTTPException(status_code=400, detail="Train % + Val % must be less than 100 when test split is enabled.")
    else:
        if req.train_pct + req.val_pct != 100:
            raise HTTPException(status_code=400, detail="Train % + Val % must equal 100 when test split is disabled.")

    task = _yolo_task_name(req.task)
    dataset_root = output_dir / ("semantic" if task == "semantic" else "segment")
    _reset_yolo_split_dir(dataset_root)

    image_paths = _infer_source_paths(image_dir)
    pairs: list[tuple[Path, Path]] = []
    for image_path in image_paths:
        anno = labels_dir / f"{image_path.stem}.{'png' if task == 'semantic' else 'txt'}"
        if anno.exists():
            pairs.append((image_path, anno))
    if not pairs:
        suffix = ".png" if task == "semantic" else ".txt"
        raise HTTPException(status_code=400, detail=f"No {suffix} annotation files found in: {labels_dir}")

    rng = random.Random(req.seed)
    rng.shuffle(pairs)
    total = len(pairs)
    train_n = round(total * req.train_pct / 100)
    val_n = round(total * req.val_pct / 100)
    if req.use_test_split:
        test_n = max(0, total - train_n - val_n)
    else:
        test_n = 0
        if train_n + val_n != total:
            val_n = total - train_n
    splits = {
        "train": pairs[:train_n],
        "val": pairs[train_n:train_n + val_n],
    }
    if req.use_test_split:
        splits["test"] = pairs[train_n + val_n:train_n + val_n + test_n]

    normalized_images = 0
    for split, items in splits.items():
        img_out = dataset_root / "images" / split
        anno_out = dataset_root / ("masks" if task == "semantic" else "labels") / split
        img_out.mkdir(parents=True, exist_ok=True)
        anno_out.mkdir(parents=True, exist_ok=True)
        for image_path, anno_path in items:
            if _write_yolo_training_image(image_path, img_out / f"{image_path.stem}.png"):
                normalized_images += 1
            shutil.copy2(anno_path, anno_out / f"{image_path.stem}{anno_path.suffix.lower()}")

    data = {
        "path": str(dataset_root),
        "train": "images/train",
        "val": "images/val",
        "nc": len(classes),
        "names": classes,
    }
    if req.use_test_split:
        data["test"] = "images/test"
    if task == "semantic":
        data["masks_dir"] = "masks"
    yaml_path = dataset_root / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return {
        "ok": True,
        "task": task,
        "yaml_path": str(yaml_path),
        "total": total,
        "counts": {"train": len(splits.get("train", [])), "val": len(splits.get("val", [])), "test": len(splits.get("test", []))},
        "normalized_images": normalized_images,
        "use_test_split": req.use_test_split,
    }


@app.post("/api/yolo/start")
def api_yolo_start(req: YoloTrainRequest) -> dict[str, Any]:
    if STATE.train_proc and STATE.train_proc.poll() is None:
        raise HTTPException(status_code=400, detail="Training is already running.")
    data_yaml = _norm_path(req.data_yaml)
    if not data_yaml.exists():
        raise HTTPException(status_code=400, detail=f"data.yaml not found: {data_yaml}")
    _validate_yolo_dataset(data_yaml, _yolo_task_name(req.task))

    output_dir = _norm_path(req.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    weights = _resolve_yolo_weights(req.task, req.model, req.auto_weights, req.custom_weights)

    runtime = _runtime_dir("train")
    run_id = int(time.time())
    cfg_path = runtime / f"train_{run_id}.json"
    script_path = runtime / f"train_{run_id}.py"
    log_path = runtime / f"train_{run_id}.log"
    results_path = output_dir / req.run_name / "results.csv"
    config = req.model_dump()
    config["data_yaml"] = str(data_yaml)
    config["output_dir"] = str(output_dir)
    config["weights"] = weights
    cfg_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    script = f"""
import json
from pathlib import Path
from ultralytics import YOLO
cfg = json.loads(Path(r'{cfg_path}').read_text())
model = YOLO(cfg['weights'])
kwargs = dict(
    data=cfg['data_yaml'],
    project=cfg['output_dir'],
    name=cfg['run_name'],
    device=cfg['device'],
    epochs=cfg['epochs'],
    imgsz=cfg['imgsz'],
    batch=cfg['batch'],
    patience=cfg['patience'],
    workers=max(0, int(cfg['workers'])),
    seed=cfg['seed'],
    amp=cfg['amp'],
    save=cfg['save'],
    plots=cfg['plots'],
    verbose=cfg['verbose'],
    save_period=(1 if cfg['save_period'] else -1),
    fliplr=cfg['fliplr'],
    flipud=cfg['flipud'],
    scale=cfg['scale'],
    translate=cfg['translate'],
    mosaic=cfg['mosaic'],
    mixup=cfg['mixup'],
)
if cfg['auto_optimize']:
    kwargs['optimizer'] = 'auto'
else:
    kwargs['optimizer'] = cfg['optimizer']
    kwargs['lr0'] = cfg['lr0']
if cfg['task'] != 'semantic':
    kwargs['cls'] = cfg['cls']
    kwargs['conf'] = cfg['conf']
    kwargs['dropout'] = cfg['dropout']
    kwargs['mask_ratio'] = cfg['mask_ratio']
    kwargs['box'] = cfg['box']
model.train(**kwargs)
"""
    script_path.write_text(script, encoding="utf-8")
    log_file = open(log_path, "w", encoding="utf-8")
    env = os.environ.copy()
    env.setdefault("TMPDIR", str(_runtime_root()))
    proc = subprocess.Popen([sys.executable, str(script_path)], stdout=log_file, stderr=subprocess.STDOUT, cwd=str(runtime), env=env)

    STATE.train_proc = proc
    STATE.train_status = "running"
    STATE.train_error = None
    STATE.train_log_path = log_path
    STATE.train_results_path = results_path
    return {"ok": True, "status": "running", "log_path": str(log_path)}


@app.post("/api/yolo/stop")
def api_yolo_stop() -> dict[str, Any]:
    if STATE.train_proc and STATE.train_proc.poll() is None:
        STATE.train_proc.terminate()
        try:
            STATE.train_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            STATE.train_proc.kill()
        STATE.train_status = "stopped"
    else:
        STATE.train_status = "idle"
    return {"ok": True, "status": STATE.train_status}


@app.post("/api/yolo/load-results")
def api_yolo_load_results(req: LoadResultsRequest) -> dict[str, Any]:
    path = _norm_path(req.results_csv)
    if not path.exists():
        raise HTTPException(status_code=400, detail=f"results.csv not found: {path}")
    STATE.train_results_path = path
    return {"ok": True, "path": str(path)}


@app.get("/api/yolo/status")
def api_yolo_status(offset: int = Query(0)) -> dict[str, Any]:
    running = STATE.train_proc is not None and STATE.train_proc.poll() is None
    if running:
        STATE.train_status = "running"
    elif STATE.train_proc is not None:
        code = STATE.train_proc.poll()
        STATE.train_status = "completed" if code == 0 else "failed"
        if code not in (None, 0) and STATE.train_error is None:
            STATE.train_error = f"Process exited with code {code}"

    lines: list[str] = []
    log_offset = offset
    if STATE.train_log_path and STATE.train_log_path.exists():
        all_lines = STATE.train_log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        lines = all_lines[offset:]
        log_offset = len(all_lines)

    results = _read_results(STATE.train_results_path) if STATE.train_results_path else {"exists": False}
    epoch = f"Epoch: {results.get('rows', '-') if results.get('exists') else '-'}"
    confusion_matrix = None
    if STATE.train_results_path is not None:
        cm = STATE.train_results_path.parent / "confusion_matrix.png"
        if cm.exists():
            confusion_matrix = f"/api/yolo/confusion-matrix?path={cm}"
    return {
        "running": running,
        "status": STATE.train_status,
        "error": STATE.train_error,
        "log": lines,
        "log_offset": log_offset,
        "results": results,
        "epoch": epoch,
        "confusion_matrix": confusion_matrix,
    }


@app.get("/api/yolo/confusion-matrix")
def api_yolo_confusion_matrix(path: str) -> FileResponse:
    cm = _norm_path(path)
    if not cm.exists():
        raise HTTPException(status_code=404, detail=f"Confusion matrix not found: {cm}")
    return FileResponse(cm)


@app.get("/api/yolo/infer/state")
def api_yolo_infer_state() -> dict[str, Any]:
    return {
        "count": len(STATE.infer_results),
        "images": [rec.name for rec in STATE.infer_results],
        "task": STATE.infer_task,
        "image_dir": str(STATE.infer_image_dir) if STATE.infer_image_dir else "",
        "output_dir": str(STATE.infer_output_dir) if STATE.infer_output_dir else "",
        "status": "ready" if STATE.infer_results else "idle",
        "model_ready": STATE.infer_model is not None,
    }


@app.post("/api/yolo/infer/init")
def api_yolo_infer_init(req: YoloInferInitRequest) -> dict[str, Any]:
    model, weights, loaded = _load_infer_model(req.task, req.model, req.auto_weights, req.custom_weights, req.device)
    return {
        "ok": True,
        "task": _yolo_task_name(req.task),
        "weights": weights,
        "device": req.device,
        "loaded": loaded,
        "status": "ready",
    }


@app.get("/api/yolo/infer/scan")
def api_yolo_infer_scan(image_dir: str) -> dict[str, Any]:
    root = _norm_path(image_dir)
    if not root.is_dir():
        raise HTTPException(status_code=400, detail=f"Image directory not found: {root}")
    images = [p.name for p in _infer_source_paths(root)]
    return {"image_dir": str(root), "images": images, "count": len(images)}


@app.get("/api/yolo/infer/source/png")
def api_yolo_infer_source_png(image_dir: str, image_name: str) -> FileResponse:
    image_path = _infer_source_record(_norm_path(image_dir), image_name)
    return FileResponse(_runtime_infer_source_png(image_path))


@app.get("/api/yolo/infer/source/meta")
def api_yolo_infer_source_meta(image_dir: str, image_name: str) -> dict[str, Any]:
    image_path = _infer_source_record(_norm_path(image_dir), image_name)
    rgb = load_image_rgb(image_path)
    h, w = rgb.shape[:2]
    return {
        "name": image_path.name,
        "summary": "Source image",
        "task": _yolo_task_name(STATE.infer_task),
        "boxes": 0,
        "masks": 0,
        "classes": [],
        "source_size": [w, h],
    }


@app.post("/api/yolo/infer/run")
def api_yolo_infer_run(req: YoloInferRequest) -> dict[str, Any]:
    image_dir = _norm_path(req.image_dir)
    if not image_dir.is_dir():
        raise HTTPException(status_code=400, detail=f"Image directory not found: {image_dir}")
    all_images = _infer_source_paths(image_dir)
    if req.scope == "current":
        if not req.image_name:
            raise HTTPException(status_code=400, detail="Select an image when using current-image inference.")
        images = [p for p in all_images if p.name == req.image_name]
    else:
        images = all_images
    if not images:
        raise HTTPException(status_code=400, detail="No images selected for inference.")

    output_dir = _norm_path(req.output_dir) if req.output_dir else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    try:
        model, _weights, _loaded = _load_infer_model(req.task, req.model, req.auto_weights, req.custom_weights, req.device)
        env_tmp = _runtime_root()
        os.environ.setdefault("TMPDIR", str(env_tmp))
        results = model.predict(source=[str(p) for p in images], conf=req.conf, imgsz=req.imgsz, device=req.device, verbose=False, save=False, retina_masks=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    runtime = _runtime_dir("infer") / str(int(time.time()))
    runtime.mkdir(parents=True, exist_ok=True)
    infer_records: list[InferRecord] = []
    for image_path, result in zip(images, results):
        infer_records.append(_infer_record(result, image_path, _yolo_task_name(req.task), runtime, output_dir, req.save_overlays, req.save_predictions))

    STATE.infer_results = infer_records
    STATE.infer_task = _yolo_task_name(req.task)
    STATE.infer_image_dir = image_dir
    STATE.infer_output_dir = output_dir
    return {
        "count": len(infer_records),
        "images": [rec.name for rec in infer_records],
        "task": STATE.infer_task,
        "image_dir": str(image_dir),
        "output_dir": str(output_dir) if output_dir else "",
    }


@app.get("/api/yolo/infer/image/{index}/png")
def api_yolo_infer_image_png(index: int) -> FileResponse:
    if index < 0 or index >= len(STATE.infer_results):
        raise HTTPException(status_code=404, detail=f"Inference image index out of range: {index}")
    return FileResponse(STATE.infer_results[index].base_png)


@app.get("/api/yolo/infer/image/{index}/overlay")
def api_yolo_infer_image_overlay(index: int) -> FileResponse:
    if index < 0 or index >= len(STATE.infer_results):
        raise HTTPException(status_code=404, detail=f"Inference image index out of range: {index}")
    return FileResponse(STATE.infer_results[index].overlay_png)


@app.get("/api/yolo/infer/image/{index}/meta")
def api_yolo_infer_image_meta(index: int) -> dict[str, Any]:
    if index < 0 or index >= len(STATE.infer_results):
        raise HTTPException(status_code=404, detail=f"Inference image index out of range: {index}")
    rec = STATE.infer_results[index]
    return {
        "name": rec.name,
        "summary": rec.summary,
        "task": rec.task,
        "boxes": rec.boxes,
        "masks": rec.masks,
        "classes": rec.classes,
        "saved_overlay": rec.saved_overlay,
        "saved_prediction": rec.saved_prediction,
        "saved_boxes": rec.saved_boxes,
    }


def main() -> None:
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Ultraprompt web app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--image-dir", default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    if args.image_dir:
        try:
            api_open_folder(OpenFolderRequest(image_dir=args.image_dir, out_dir=args.out_dir))
        except Exception:
            pass
    elif args.out_dir:
        STATE.out_dir = _norm_path(args.out_dir)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
