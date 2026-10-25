"""Validation, intensity normalization, and debug export for image frames."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_NORMALIZE_MODES = {"none", "global", "per_frame", "percentile"}


def validate_frames(frames: Sequence[np.ndarray]) -> dict[str, Any]:
    """Validate the agreed frame contract and return summary statistics.

    The function validates *every* frame, not just the first one. It rejects
    NaN/Inf values because those can silently corrupt later numerical work.
    """
    if len(frames) == 0:
        raise ValueError("Frame list is empty")

    expected_shape: tuple[int, int] | None = None
    global_min = math.inf
    global_max = -math.inf

    for index, frame in enumerate(frames):
        if not isinstance(frame, np.ndarray):
            raise TypeError(f"Frame {index} must be a numpy.ndarray, got {type(frame).__name__}")
        if frame.ndim != 2:
            raise ValueError(f"Frame {index} must have shape (H, W), got {frame.shape}")
        if frame.shape[0] == 0 or frame.shape[1] == 0:
            raise ValueError(f"Frame {index} has an empty dimension: {frame.shape}")
        if frame.dtype != np.float32:
            raise TypeError(f"Frame {index} must have dtype float32, got {frame.dtype}")
        if expected_shape is None:
            expected_shape = frame.shape
        elif frame.shape != expected_shape:
            raise ValueError(
                f"Inconsistent frame shapes: expected {expected_shape}, "
                f"frame {index} has {frame.shape}"
            )
        if not np.isfinite(frame).all():
            raise ValueError(f"Frame {index} contains NaN or infinite values")

        local_min = float(frame.min())
        local_max = float(frame.max())
        global_min = min(global_min, local_min)
        global_max = max(global_max, local_max)

    if global_min < 0.0 or global_max > 1.0:
        raise ValueError(
            f"Pixel values must be in [0, 1], got min={global_min}, max={global_max}"
        )

    stats: dict[str, Any] = {
        "n_frames": len(frames),
        "shape": expected_shape,
        "dtype": "float32",
        "global_min": global_min,
        "global_max": global_max,
    }
    logger.info("Frame contract validated: %s", stats)
    return stats


def normalize(
    frames: Sequence[np.ndarray],
    mode: str = "none",
) -> list[np.ndarray]:
    """Optionally normalize a canonical frame sequence.

    ``none`` leaves values unchanged. ``global`` uses a shared min/max across
    all frames. ``percentile`` clips using shared 1st/99th percentiles before
    min-max scaling. ``per_frame`` normalizes each image independently; this
    can destroy relative brightness and should generally NOT be used for layer
    separation unless the downstream team explicitly wants that behavior.
    """
    if mode not in _NORMALIZE_MODES:
        raise ValueError(
            f"Unknown normalization mode {mode!r}; choose one of {sorted(_NORMALIZE_MODES)}"
        )
    validate_frames(frames)

    if mode == "none":
        return list(frames)

    if mode == "per_frame":
        out: list[np.ndarray] = []
        for frame in frames:
            lo = float(frame.min())
            hi = float(frame.max())
            if hi <= lo:
                out.append(np.zeros_like(frame, dtype=np.float32))
            else:
                out.append(((frame - lo) / (hi - lo)).astype(np.float32, copy=False))
        logger.warning(
            "Applied per-frame normalization; relative intensity between frames may be lost"
        )
        return out

    # Avoid np.stack(frames), which creates another full-size copy of the data.
    if mode == "global":
        lo = min(float(frame.min()) for frame in frames)
        hi = max(float(frame.max()) for frame in frames)
    else:  # percentile
        all_values = np.concatenate([frame.reshape(-1) for frame in frames])
        lo, hi = (float(v) for v in np.percentile(all_values, [1.0, 99.0]))
        del all_values

    if hi <= lo:
        logger.info("Normalization skipped: all pixel values are constant")
        return [np.zeros_like(frame, dtype=np.float32) for frame in frames]

    out = [
        np.clip((frame - lo) / (hi - lo), 0.0, 1.0).astype(np.float32, copy=False)
        for frame in frames
    ]
    logger.info("Applied %s normalization with lo=%.6g, hi=%.6g", mode, lo, hi)
    return out


def _to_uint8_unchecked(frame: np.ndarray) -> np.ndarray:
    """Convert a previously validated grayscale frame without another full scan."""
    return np.rint(np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8)


def to_uint8(frame: np.ndarray) -> np.ndarray:
    """Convert one grayscale frame from [0, 1] to uint8 [0, 255].

    Values are clipped deliberately so this is also useful for previewing
    intermediate arrays. Non-finite values are rejected rather than hidden.
    """
    if not isinstance(frame, np.ndarray):
        raise TypeError("frame must be a numpy.ndarray")
    if frame.ndim != 2:
        raise ValueError(f"frame must have shape (H, W), got {frame.shape}")
    if not np.issubdtype(frame.dtype, np.number):
        raise TypeError(f"frame must contain numeric values, got {frame.dtype}")
    if not np.isfinite(frame).all():
        raise ValueError("frame contains NaN or infinite values")
    return _to_uint8_unchecked(frame)


def save_frames(
    frames: Sequence[np.ndarray],
    out_dir: str | Path,
    prefix: str = "frame",
    *,
    validate: bool = True,
) -> list[Path]:
    """Save numbered debug PNGs. Set ``validate=False`` only after validation."""
    if validate:
        validate_frames(frames)
    elif len(frames) == 0:
        raise ValueError("Frame list is empty")
    if not prefix or Path(prefix).name != prefix or prefix in {".", ".."}:
        raise ValueError("prefix must be a non-empty filename prefix, not a path")

    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    for index, frame in enumerate(frames):
        path = out_dir / f"{prefix}_{index:04d}.png"
        Image.fromarray(_to_uint8_unchecked(frame)).save(path, format="PNG")
        paths.append(path)

    logger.info("Saved %d PNG frames to %s", len(paths), out_dir)
    return paths


def save_preview_gif(
    frames: Sequence[np.ndarray],
    out_path: str | Path,
    fps: float = 10.0,
    *,
    validate: bool = True,
) -> Path:
    """Save a GIF preview. Set ``validate=False`` only after validation."""
    if validate:
        validate_frames(frames)
    elif len(frames) == 0:
        raise ValueError("Frame list is empty")
    if not math.isfinite(fps) or fps <= 0 or fps > 100:
        raise ValueError(f"fps must be finite and in (0, 100] for GIF timing resolution, got {fps!r}")

    out_path = Path(out_path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # GIF delays are stored in 10 ms increments; round to the nearest centisecond.
    duration_ms = max(10, int(math.floor(100.0 / fps + 0.5)) * 10)
    images = [Image.fromarray(_to_uint8_unchecked(frame)).convert("P") for frame in frames]
    images[0].save(
        out_path,
        format="GIF",
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    logger.info("Saved preview GIF to %s (%.3g fps requested)", out_path, fps)
    return out_path
