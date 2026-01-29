# src/ultraprompt/core/sam_yolo_annotation.py
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

try:
    from PIL import Image
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False

from ultralytics import SAM
try:
    # SAM 3 semantic (concept) segmentation predictor
    from ultralytics.models.sam import SAM3SemanticPredictor
    _HAS_SAM3_SEMANTIC = True
except Exception:
    SAM3SemanticPredictor = None  # type: ignore
    _HAS_SAM3_SEMANTIC = False



# -------------------- Utilities --------------------

def effective_device(pref: Optional[str]) -> str:
    """Map 'auto'/'cuda'/'cpu'/None to a real device string."""
    if pref is None or str(pref).lower() == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    pref = str(pref).lower()
    if pref == "cuda":
        return "cuda"
    if pref == "cpu":
        return "cpu"
    return "cpu"


def load_image_rgb(path: Path | str) -> np.ndarray:
    """Read image as HxWx3 RGB uint8."""
    p = str(path)
    if _HAS_CV2:
        im = cv2.imread(p, cv2.IMREAD_COLOR)
        if im is None:
            raise RuntimeError(f"cv2 failed to read: {p}")
        return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    if _HAS_PIL:
        return np.array(Image.open(p).convert("RGB"))
    raise RuntimeError("Need OpenCV or PIL to read images.")


def _best_mask_from_results(res_list) -> Optional[np.ndarray]:
    """
    Ultralytics returns a list of Results; we pick the largest-area mask from the first item.
    Returns boolean mask (H,W) or None.
    """
    if not res_list:
        return None
    res = res_list[0]
    if getattr(res, "masks", None) is None:
        return None
    mm = res.masks.data  # tensor (N,H,W) float
    m = mm.cpu().numpy()
    if m.ndim == 3 and m.shape[0] > 1:
        areas = m.reshape(m.shape[0], -1).sum(1)
        m = m[int(np.argmax(areas))]
    else:
        m = m[0]
    return (m > 0.5).astype(bool)



def _all_masks_from_results(res_list) -> List[np.ndarray]:
    """Return all masks from the first Results item as boolean (H,W) arrays."""
    if not res_list:
        return []
    res = res_list[0]
    if getattr(res, "masks", None) is None:
        return []
    mm = res.masks.data  # tensor (N,H,W) float
    m = mm.cpu().numpy()
    if m.ndim != 3:
        return []
    return [(m[i] > 0.5).astype(bool) for i in range(m.shape[0])]

def colorize_masks_rgba(masks: List[np.ndarray], alpha: float = 0.45) -> Optional[np.ndarray]:
    """Return an RGBA overlay from boolean masks."""
    if not masks:
        return None
    H, W = masks[0].shape
    out = np.zeros((H, W, 4), dtype=np.uint8)
    rng = np.random.default_rng(42)
    cols = rng.integers(0, 255, size=(len(masks), 3), dtype=np.uint8)
    for i, m in enumerate(masks):
        if m is None:
            continue
        idx = m.astype(bool)
        out[idx, :3] = cols[i]
        out[idx, 3] = int(alpha * 255)
    return out


def mask_to_polygon(mask: np.ndarray, simplify_eps: float = 2.0) -> Optional[np.ndarray]:
    """Largest external contour as polygon (Nx2) in pixel coords."""
    if not _HAS_CV2:
        raise RuntimeError("OpenCV required for mask_to_polygon.")
    H, W = mask.shape
    m8 = (mask.astype(np.uint8) * 255)
    cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    if simplify_eps and simplify_eps > 0:
        cnt = cv2.approxPolyDP(cnt, epsilon=simplify_eps, closed=True)
    poly = cnt.reshape(-1, 2)
    if poly.shape[0] < 3:
        return None
    poly[:, 0] = np.clip(poly[:, 0], 0, W - 1)
    poly[:, 1] = np.clip(poly[:, 1], 0, H - 1)
    return poly.astype(np.float32)


def write_yolo_seg(label_path: Path,
                   polygons: List[np.ndarray],
                   class_ids: List[int],
                   img_w: int,
                   img_h: int) -> None:
    """YOLO-Seg format writer."""
    lines = []
    for cls_id, poly in zip(class_ids, polygons):
        if poly is None or len(poly) < 3:
            continue
        norm = poly.copy()
        norm[:, 0] = norm[:, 0] / float(img_w)
        norm[:, 1] = norm[:, 1] / float(img_h)
        flat = " ".join(f"{v:.6f}" for xy in norm for v in xy)
        lines.append(f"{int(cls_id)} {flat}")
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))


# -------------------- UltraSAM2 wrapper --------------------


# -------------------- UltraSAM3 wrapper --------------------

class UltraSAM3:
    """
    Thin wrapper around Ultralytics' SAM interface that supports:

      1) Visual segmentation (SAM 2-style): points/boxes/masks -> single instance
         - Implemented via `ultralytics.SAM` using `sam3.pt` (or other SAM weights).

      2) Concept segmentation (SAM 3 PCS): text prompts and/or exemplar boxes -> *all* instances
         - Implemented via `ultralytics.models.sam.SAM3SemanticPredictor`.

    Notes:
      - Visual prompting works by calling the SAM model directly each time (no `.predictor` usage).
      - Concept prompting uses the SAM3SemanticPredictor, and (for now) expects an image *path*
        for `set_image()`. The GUI always has a file path, so this keeps things simple.

    Usage:
        sam = UltraSAM3()
        sam.load("sam3.pt", device="auto")

        img = load_image_rgb("image.png")
        sam.bind_image(img, image_path="image.png")

        # Visual prompts (points/boxes)
        masks = sam.infer_visual(points=[[x,y], ...], labels=[1,0,...], boxes=[[x0,y0,x1,y1], ...])

        # Concept prompts (text and/or exemplar boxes)
        masks = sam.infer_concept(text=["person", "bus"], exemplars=[[x0,y0,x1,y1]])
    """

    def __init__(self) -> None:
        # Visual model (SAM interface) - supports SAM2 + SAM3 PVS
        self.model: Optional[SAM] = None
        # Semantic predictor (SAM3 PCS)
        self.semantic: Optional["SAM3SemanticPredictor"] = None  # type: ignore[name-defined]
        self._device: str = "cpu"

        self._last_image: Optional[np.ndarray] = None
        self._last_image_path: Optional[str] = None

        # Keep the predictor overrides around so we can rebuild if device changes
        self._semantic_overrides: Optional[dict] = None

    @property
    def device(self) -> str:
        return self._device

    def load(
        self,
        weights: Path | str,
        device: str | None = "auto",
        *,
        semantic_conf: float = 0.25,
        semantic_half: bool = True,
        verbose: bool = False,
    ) -> None:
        """Load SAM weights and (optionally) initialize SAM3 semantic predictor."""
        self._device = effective_device(device)

        # Visual prompting model (works for sam3.pt too)
        self.model = SAM(str(weights))
        try:
            self.model.to(self._device)  # no-op on some builds; safe to try
        except Exception:
            pass

        # Concept segmentation predictor (SAM3 only). Safe to skip if deps unavailable.
        self.semantic = None
        self._semantic_overrides = None
        if _HAS_SAM3_SEMANTIC:
            # Ultralytics docs recommend SAM3SemanticPredictor(overrides=...)
            self._semantic_overrides = dict(
                conf=float(semantic_conf),
                task="segment",
                mode="predict",
                model=str(weights),
                half=bool(semantic_half),
                verbose=bool(verbose),
                device=self._device,
                save=False
            )
            try:
                self.semantic = SAM3SemanticPredictor(overrides=self._semantic_overrides)  # type: ignore[call-arg]
            except Exception:
                # Keep visual segmentation working even if semantic init fails.
                print("SAM3SemanticPredictor init failed:", e)
                self.semantic = None

    def _ensure_ready(self) -> None:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load(weights_path) first.")

    def bind_image(self, image_rgb: np.ndarray, image_path: Path | str | None = None) -> None:
        """Cache the image for subsequent calls. Provide `image_path` to enable concept segmentation."""
        self._ensure_ready()
        self._last_image = image_rgb
        self._last_image_path = str(image_path) if image_path is not None else None

        # For semantic predictor, we set the image once.
        if self.semantic is not None and self._last_image_path is not None:
            try:
                self.semantic.set_image(self._last_image_path)
            except Exception:
                # Don't hard-fail; caller may still use visual segmentation.
                pass

    # ---------- Visual segmentation (SAM2-style prompts) ----------
    def infer_visual(
        self,
        points: Optional[Sequence[Sequence[float]]] = None,
        labels: Optional[Sequence[int]] = None,
        boxes: Optional[Sequence[Sequence[float]]] = None,
        multimask_output: bool = True,  # kept for compatibility, not passed through
    ) -> List[np.ndarray]:
        """
        Run SAM visual prompting. Returns a list of boolean masks.

        Behavior:
          - If `boxes` are provided: returns ONE mask per box. `points/labels` are global hints.
          - Else if only `points/labels` are provided: returns ONE mask for the point prompts.
          - Else: returns [] (use segment_everything() instead).
        """
        self._ensure_ready()
        if self._last_image is None:
            raise RuntimeError("No image bound. Call bind_image(image_rgb) first.")

        if points is None and labels is not None:
            raise ValueError("labels provided without points.")
        if points is not None and labels is None:
            raise ValueError("points provided without labels.")

        pts_b = [list(points)] if points else None
        labs_b = [list(labels)] if labels else None

        masks_out: List[np.ndarray] = []

        if boxes and len(boxes) > 0:
            # one call per box; pass only supported kwargs
            for (x0, y0, x1, y1) in boxes:
                kwargs = dict(device=self._device, verbose=False,
                              bboxes=[[float(x0), float(y0), float(x1), float(y1)]])
                if pts_b is not None:
                    kwargs["points"] = pts_b
                if labs_b is not None:
                    kwargs["labels"] = labs_b
                res_list = self.model(self._last_image, **kwargs)  # type: ignore[misc]
                m = _best_mask_from_results(res_list)
                if m is not None:
                    masks_out.append(m)
            return masks_out

        if points and labels:
            kwargs = dict(device=self._device, verbose=False)
            if pts_b is not None:
                kwargs["points"] = pts_b
            if labs_b is not None:
                kwargs["labels"] = labs_b
            res_list = self.model(self._last_image, **kwargs)  # type: ignore[misc]
            m = _best_mask_from_results(res_list)
            if m is not None:
                masks_out.append(m)
            return masks_out

        return []

    def segment_everything(self, image_rgb: np.ndarray | None = None, top_n: int = 20) -> List[np.ndarray]:
        """Segment everything in the current (or provided) image using the SAM interface."""
        self._ensure_ready()
        img = image_rgb if image_rgb is not None else self._last_image
        if img is None:
            raise RuntimeError("No image provided/bound for segment_everything().")

        res = self.model(img, device=self._device, verbose=False)[0]  # type: ignore[misc]
        masks_out: List[np.ndarray] = []
        if res.masks is not None:
            mm = res.masks.data.cpu().numpy()  # (N,H,W)
            areas = mm.reshape(mm.shape[0], -1).sum(1)
            keep = np.argsort(-areas)[:min(top_n, mm.shape[0])]
            for i in keep:
                masks_out.append((mm[i] > 0.5).astype(bool))
        return masks_out

    # ---------- Concept segmentation (SAM3 PCS prompts) ----------
    def infer_concept(
        self,
        *,
        text: Optional[Sequence[str]] = None,
        exemplars: Optional[Sequence[Sequence[float]]] = None,
    ) -> List[np.ndarray]:
        """Run SAM3 Promptable Concept Segmentation (PCS).

        Args:
          text: list of noun phrases (e.g. ["person", "bus"]).
          exemplars: list of exemplar bboxes [[x0,y0,x1,y1], ...] to find similar instances.

        Returns:
          List of boolean masks, one per detected instance.
        """

        if self.semantic is None:
            raise RuntimeError(
                "SAM3SemanticPredictor is not available. "
                "Ensure ultralytics>=8.3.237 and required deps (e.g., Ultralytics CLIP) are installed."
            )
        if not self._last_image_path:
            raise RuntimeError(
                f"Concept segmentation requires a non-empty image_path. Got: {self._last_image_path!r}. "
                "Call bind_image(..., image_path=str(path))."
            )

        # Ensure the predictor has the current image set (safe to call repeatedly).
        try:
            self.semantic.set_image(self._last_image_path)
        except Exception:
            pass

        kwargs = {}
        if text:
            kwargs["text"] = list(text)
        if exemplars:
            kwargs["bboxes"] = [[float(x0), float(y0), float(x1), float(y1)] for (x0, y0, x1, y1) in exemplars]

        if not kwargs:
            return []

        res_list = self.semantic(**kwargs)
        return _all_masks_from_results(res_list)


# Backwards-compat alias (older GUI/imports)
UltraSAM2 = UltraSAM3
