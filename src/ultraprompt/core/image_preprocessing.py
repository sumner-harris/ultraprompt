from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ImagePreprocessSettings:
    pixel_scale: int = 1
    filter_name: str = "none"


def _require_cv2():
    try:
        import cv2
    except Exception as e:
        raise RuntimeError("Image preprocessing requires OpenCV (cv2).") from e
    return cv2


def resize_for_pixel_scale(image: np.ndarray, pixel_scale: int) -> np.ndarray:
    pixel_scale = max(1, int(pixel_scale))
    if pixel_scale == 1:
        return np.ascontiguousarray(image)

    cv2 = _require_cv2()
    h, w = image.shape[:2]
    new_w = max(1, int(round(w / pixel_scale)))
    new_h = max(1, int(round(h / pixel_scale)))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _dog(image: np.ndarray, sigma_small: float, sigma_large: float) -> np.ndarray:
    cv2 = _require_cv2()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    small = cv2.GaussianBlur(gray, (0, 0), sigma_small)
    large = cv2.GaussianBlur(gray, (0, 0), sigma_large)
    dog = small - large
    lo, hi = np.percentile(dog, [0.5, 99.5])
    if hi <= lo:
        lo, hi = float(dog.min()), float(dog.max())
    if hi <= lo:
        out = np.zeros_like(gray, dtype=np.uint8)
    else:
        out = np.clip((dog - lo) / (hi - lo), 0, 1)
        out = (out * 255).astype(np.uint8)
    return np.stack([out, out, out], axis=-1)


def apply_filter(image: np.ndarray, filter_name: str) -> np.ndarray:
    name = (filter_name or "none").lower()
    if name == "none":
        return np.ascontiguousarray(image)

    cv2 = _require_cv2()
    if name == "gaussian":
        return cv2.GaussianBlur(image, (0, 0), 1.0)
    if name == "median":
        return cv2.medianBlur(image, 3)
    if name == "bilateral":
        return cv2.bilateralFilter(image, 7, 40, 40)
    if name == "dog_fine":
        return _dog(image, 1.0, 2.0)
    if name == "dog_medium":
        return _dog(image, 1.5, 4.0)
    if name == "dog_coarse":
        return _dog(image, 2.0, 8.0)
    if name == "sharpen":
        blur = cv2.GaussianBlur(image, (0, 0), 1.5)
        return cv2.addWeighted(image, 1.5, blur, -0.5, 0)

    raise ValueError(f"Unknown image filter: {filter_name}")


def preprocess_image_rgb(image: np.ndarray, settings: ImagePreprocessSettings) -> np.ndarray:
    out = resize_for_pixel_scale(image, settings.pixel_scale)
    out = apply_filter(out, settings.filter_name)
    if out.dtype != np.uint8:
        out = np.clip(out, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(out)


def scale_polygon(poly: np.ndarray, sx: float, sy: float) -> np.ndarray:
    out = np.asarray(poly, dtype=np.float32).copy()
    if out.size == 0:
        return out
    out[:, 0] *= float(sx)
    out[:, 1] *= float(sy)
    return out
