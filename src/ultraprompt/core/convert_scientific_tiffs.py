from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

TIFF_EXTS = {".tif", ".tiff"}


def _first_display_plane(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)

    while arr.ndim > 3:
        arr = arr[0]

    if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
        arr = np.moveaxis(arr, 0, -1)

    if arr.ndim == 3 and arr.shape[-1] not in (3, 4):
        arr = arr[0]

    if arr.ndim == 3 and arr.shape[-1] >= 4:
        arr = arr[..., :3]

    if arr.ndim not in (2, 3):
        raise ValueError(f"unsupported TIFF shape: {arr.shape}")

    return arr


def normalize_to_uint8(arr: np.ndarray, low: float = 0.5, high: float = 99.5) -> np.ndarray:
    arr = _first_display_plane(arr).astype(np.float32, copy=False)
    finite = np.isfinite(arr)

    if not finite.any():
        return np.zeros(arr.shape, dtype=np.uint8)

    vals = arr[finite]
    lo, hi = np.percentile(vals, [low, high])

    if hi <= lo:
        lo, hi = float(vals.min()), float(vals.max())

    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)

    out = np.clip((arr - lo) / (hi - lo), 0, 1)
    return (out * 255).astype(np.uint8)


def iter_tiffs(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    files = sorted(input_path.glob("*.tif")) + sorted(input_path.glob("*.tiff"))
    return [p for p in files if p.is_file()]


def needs_uint8_conversion(path: Path | str) -> bool:
    path = Path(path)
    if path.suffix.lower() not in TIFF_EXTS:
        return False

    import tifffile

    with tifffile.TiffFile(path) as tif:
        if not tif.pages:
            return True
        return np.dtype(tif.pages[0].dtype) != np.dtype(np.uint8)


def converted_path_for(src: Path, output_dir: Path) -> Path:
    return output_dir / f"{src.stem}.png"


def convert_file(src: Path, dst: Path, low: float = 0.5, high: float = 99.5, rgb: bool = False) -> None:
    import tifffile

    arr = tifffile.imread(src)
    out = normalize_to_uint8(arr, low=low, high=high)

    if rgb and out.ndim == 2:
        out = np.stack([out, out, out], axis=-1)

    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out).save(dst)


def convert_if_required(
    src: Path,
    output_dir: Path,
    low: float = 0.5,
    high: float = 99.5,
    rgb: bool = False,
    force: bool = False,
) -> tuple[Path, bool]:
    if not needs_uint8_conversion(src):
        return src, False

    dst = converted_path_for(src, output_dir)
    if force or not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
        convert_file(src, dst, low=low, high=high, rgb=rgb)
    return dst, True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert 16/32-bit scientific TIFFs to normalized 8-bit PNGs."
    )
    parser.add_argument("input", type=Path, help="TIFF file or folder containing .tif/.tiff files")
    parser.add_argument("--output", "-o", type=Path, required=True, help="output folder")
    parser.add_argument("--low", type=float, default=0.5, help="low percentile for contrast scaling")
    parser.add_argument("--high", type=float, default=99.5, help="high percentile for contrast scaling")
    parser.add_argument("--rgb", action="store_true", help="write grayscale inputs as RGB PNGs")
    parser.add_argument("--force", action="store_true", help="recreate outputs even when they are up to date")
    args = parser.parse_args()

    files = iter_tiffs(args.input)
    if not files:
        raise SystemExit(f"no TIFF files found: {args.input}")

    converted = 0
    skipped = 0
    for src in files:
        if needs_uint8_conversion(src):
            dst = converted_path_for(src, args.output)
            if args.force or not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
                convert_file(src, dst, low=args.low, high=args.high, rgb=args.rgb)
                print(f"wrote {dst}")
            else:
                print(f"kept {dst}")
            converted += 1
        else:
            print(f"skipped {src} (already uint8)")
            skipped += 1

    print(f"converted {converted} file(s); skipped {skipped} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
