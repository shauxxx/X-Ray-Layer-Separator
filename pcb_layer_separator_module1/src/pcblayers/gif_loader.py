"""Load GIFs or image sequences as canonical grayscale float32 frames.

Prefer a sequence of 16-bit PNG files whenever the original projections are
available. GIF frames are palette-based and cannot preserve the full dynamic
range of a typical 16-bit projection.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageSequence

logger = logging.getLogger(__name__)


def _natural_sort_key(path: Path) -> list[tuple[int, object]]:
    """Sort names naturally (frame_2.png before frame_10.png)."""
    # Tag numeric and textual chunks so arbitrary filenames remain comparable.
    return [
        (0, int(part)) if part.isdigit() else (1, part.casefold())
        for part in re.split(r"(\d+)", path.name)
    ]


def _scale_integer_image(arr: np.ndarray, *, source_mode: str) -> np.ndarray:
    """Scale a supported integer image format without per-frame stretching."""
    if arr.dtype == np.uint8:
        values = arr.astype(np.float32) / np.float32(255.0)
    elif arr.dtype == np.uint16:
        values = arr.astype(np.float32) / np.float32(65535.0)
    elif np.issubdtype(arr.dtype, np.signedinteger):
        # Pillow may expose a 16-bit grayscale PNG as mode I / int32.
        # Accept the 16-bit unsigned value range, but do not guess how an
        # arbitrary 32-bit signed scientific image should be scaled.
        if source_mode == "I" and arr.size and int(arr.min()) >= 0 and int(arr.max()) <= 65535:
            values = arr.astype(np.float32) / np.float32(65535.0)
        else:
            raise ValueError(
                f"Unsupported integer pixel range {arr.min()}..{arr.max()} "
                f"for mode {source_mode!r}; expected 8-bit or 16-bit grayscale data"
            )
    elif np.issubdtype(arr.dtype, np.unsignedinteger):
        scale = np.float32(np.iinfo(arr.dtype).max)
        values = arr.astype(np.float32) / scale
    else:
        raise TypeError(f"Unsupported image dtype: {arr.dtype}")

    if not np.isfinite(values).all():
        raise ValueError("Image contains NaN or infinite pixel values")
    # Guard against unexpected decoder data or floating-point edge cases.
    np.clip(values, 0.0, 1.0, out=values)
    return np.ascontiguousarray(values, dtype=np.float32)


def _pil_to_gray_float(frame: Image.Image) -> np.ndarray:
    """Convert one Pillow image to a 2-D float32 grayscale array in [0, 1].

    8-bit and 16-bit grayscale pixels are scaled by their format's fixed
    maximum. Floating-point mode ``F`` is accepted only when its values are
    already finite and in [0, 1]; silently stretching scientific data per
    frame would destroy comparability between projections.
    """
    if frame.width <= 0 or frame.height <= 0:
        raise ValueError(f"Invalid image dimensions: {frame.size}")

    mode = frame.mode
    if mode in {"1", "L", "I;16", "I;16L", "I;16B", "I"}:
        arr = np.asarray(frame)
        # Pillow mode "1" commonly yields bool, which represents 0 or 1.
        if arr.dtype == np.bool_:
            return np.ascontiguousarray(arr.astype(np.float32))
        return _scale_integer_image(arr, source_mode=mode)

    if mode == "F":
        values = np.asarray(frame, dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("Floating-point image contains NaN or infinity")
        lo = float(values.min())
        hi = float(values.max())
        if lo < 0.0 or hi > 1.0:
            raise ValueError(
                f"Pillow mode F contains values outside [0, 1] ({lo}..{hi}). "
                "Choose and document a shared normalization scale before loading."
            )
        return np.ascontiguousarray(values, dtype=np.float32)

    # Composite transparency over black before grayscale conversion. This is
    # important for palette GIFs/PNGs whose transparent pixels should not
    # turn into arbitrary palette colors.
    rgba = frame.convert("RGBA")
    black_background = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
    gray = Image.alpha_composite(black_background, rgba).convert("L")
    return _scale_integer_image(np.asarray(gray), source_mode="L")


def _ensure_same_shape(frames: Iterable[np.ndarray], *, source: Path) -> None:
    expected: tuple[int, int] | None = None
    for index, frame in enumerate(frames):
        if frame.ndim != 2:
            raise ValueError(f"Frame {index} in {source} is not grayscale: {frame.shape}")
        if expected is None:
            expected = frame.shape
        elif frame.shape != expected:
            raise ValueError(
                f"Inconsistent frame shapes in {source}: frame 0 is {expected}, "
                f"frame {index} is {frame.shape}"
            )


def load_gif(path: str | Path) -> list[np.ndarray]:
    """Load every frame of a GIF as ``float32`` grayscale data in [0, 1].

    Raises:
        FileNotFoundError: the path does not exist or is not a file.
        ValueError: the input is not a GIF, contains no frames, or has invalid
            frame data.
        OSError: Pillow cannot decode the file.
    """
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"GIF file not found: {path}")

    frames: list[np.ndarray] = []
    durations_ms: list[int] = []
    with Image.open(path) as im:
        if im.format != "GIF":
            raise ValueError(f"Expected a GIF file, got {im.format!r}: {path}")

        declared_count = int(getattr(im, "n_frames", 1))
        for frame in ImageSequence.Iterator(im):
            frames.append(_pil_to_gray_float(frame))
            duration = frame.info.get("duration", 0)
            try:
                durations_ms.append(max(0, int(duration or 0)))
            except (TypeError, ValueError):
                durations_ms.append(0)

    if not frames:
        raise ValueError(f"No frames could be decoded from GIF: {path}")
    _ensure_same_shape(frames, source=path)

    total_duration_ms = sum(durations_ms)
    effective_fps = (
        len(frames) * 1000.0 / total_duration_ms
        if total_duration_ms > 0
        else None
    )
    logger.info(
        "Loaded GIF '%s': %d/%d frames, size=%s, duration=%.1f ms, effective_fps=%s",
        path.name,
        len(frames),
        declared_count,
        frames[0].shape,
        float(total_duration_ms),
        f"{effective_fps:.3f}" if effective_fps is not None else "n/a",
    )
    return frames


def load_png_sequence(
    directory: str | Path,
    pattern: str = "*.png",
) -> list[np.ndarray]:
    """Load images matching ``pattern`` from a folder in natural filename order.

    16-bit grayscale PNG is supported and scaled by 65535, preserving the
    full available intensity scale. All images must have identical dimensions.
    """
    directory = Path(directory).expanduser()
    if not directory.is_dir():
        raise NotADirectoryError(f"Image sequence directory not found: {directory}")
    if not pattern or not isinstance(pattern, str):
        raise ValueError("pattern must be a non-empty glob pattern string")

    files = sorted(
        (p for p in directory.glob(pattern) if p.is_file()),
        key=_natural_sort_key,
    )
    if not files:
        raise FileNotFoundError(
            f"No files matching pattern {pattern!r} in directory {directory}"
        )

    frames: list[np.ndarray] = []
    expected_shape: tuple[int, int] | None = None
    for index, file_path in enumerate(files):
        with Image.open(file_path) as im:
            frame = _pil_to_gray_float(im)

        if frame.ndim != 2:
            raise ValueError(f"Image is not grayscale after conversion: {file_path}")
        if expected_shape is None:
            expected_shape = frame.shape
        elif frame.shape != expected_shape:
            raise ValueError(
                f"Inconsistent image sizes: {files[0].name} is {expected_shape}, "
                f"but {file_path.name} is {frame.shape}"
            )
        frames.append(frame)

    logger.info(
        "Loaded image sequence '%s': %d files, size=%s, pattern=%r",
        directory.name,
        len(frames),
        frames[0].shape,
        pattern,
    )
    return frames


def load_sequence(
    path: str | Path,
    pattern: str = "*.png",
) -> list[np.ndarray]:
    """Load a GIF, a single PNG image, or a directory of image frames.

    A single PNG is returned as a one-frame list. This is useful for loading
    stitched/overview images for inspection, but layer-separation algorithms
    generally need the full tomosynthesis sequence instead.
    """
    path = Path(path).expanduser()
    if path.is_file():
        suffix = path.suffix.lower()
        if suffix == ".gif":
            return load_gif(path)
        if suffix == ".png":
            with Image.open(path) as im:
                if im.format != "PNG":
                    raise ValueError(f"Expected a PNG image: {path}")
                frame = _pil_to_gray_float(im)
            logger.info(
                "Loaded single PNG '%s': size=%s, dtype=%s",
                path.name,
                frame.shape,
                frame.dtype,
            )
            return [frame]
        raise ValueError(
            f"Unsupported input file {path}; expected a .gif file, a .png image, "
            "or a directory of images"
        )
    if path.is_dir():
        return load_png_sequence(path, pattern=pattern)
    raise FileNotFoundError(f"Input path does not exist: {path}")
