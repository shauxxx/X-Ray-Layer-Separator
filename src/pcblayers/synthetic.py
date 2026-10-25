"""Generate deterministic synthetic grayscale frames for parallel development."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

from .preprocessing import to_uint8, validate_frames


def _validate_synthetic_args(
    n_frames: int,
    size: int,
    n_spots: int,
    spot_sigma: float,
) -> None:
    if isinstance(n_frames, bool) or not isinstance(n_frames, (int, np.integer)) or n_frames < 1:
        raise ValueError("n_frames must be an integer >= 1")
    if isinstance(size, bool) or not isinstance(size, (int, np.integer)) or size < 12:
        raise ValueError("size must be an integer >= 12")
    if isinstance(n_spots, bool) or not isinstance(n_spots, (int, np.integer)) or n_spots < 0:
        raise ValueError("n_spots must be an integer >= 0")
    if isinstance(spot_sigma, bool) or not isinstance(spot_sigma, (int, float, np.number)):
        raise ValueError("spot_sigma must be a finite number > 0")
    if not math.isfinite(float(spot_sigma)) or spot_sigma <= 0:
        raise ValueError("spot_sigma must be a finite number > 0")


def make_synthetic_frames(
    n_frames: int = 20,
    size: int = 128,
    n_spots: int = 5,
    spot_sigma: float = 3.0,
    seed: int = 0,
) -> list[np.ndarray]:
    """Return X-ray-like grayscale frames in the agreed float32 [0, 1] format.

    The sequence contains a static noisy background and moving Gaussian spots.
    It is intended for integration tests and parallel development, not as a
    physical model of a real PCB or CT acquisition process.
    """
    _validate_synthetic_args(n_frames, size, n_spots, spot_sigma)
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]

    background = 0.15 + 0.10 * (xx / size) + 0.05 * (yy / size)
    background = np.clip(
        background + rng.normal(0.0, 0.02, (size, size)), 0.0, 1.0
    ).astype(np.float32)

    lower = max(5.0, size * 0.2)
    upper = min(float(size - 5), size * 0.8)
    positions = rng.uniform(lower, upper, size=(n_spots, 2))
    velocities = rng.normal(0.0, 1.5, size=(n_spots, 2))
    denominator = 2.0 * float(spot_sigma) ** 2

    frames: list[np.ndarray] = []
    for _ in range(n_frames):
        image = background.copy()
        for y, x in positions:
            distance_squared = (yy - y) ** 2 + (xx - x) ** 2
            image += (0.5 * np.exp(-distance_squared / denominator)).astype(np.float32)
        frames.append(np.clip(image, 0.0, 1.0).astype(np.float32, copy=False))

        positions += velocities
        low_hits = positions < 5.0
        high_hits = positions > size - 5.0
        velocities[low_hits | high_hits] *= -1.0
        positions = np.clip(positions, 5.0, size - 5.0)

    validate_frames(frames)
    return frames


def save_synthetic_gif(
    path: str | Path,
    n_frames: int = 20,
    size: int = 128,
    fps: float = 10.0,
    seed: int = 0,
) -> Path:
    """Generate synthetic frames and save them as an animated GIF."""
    if not math.isfinite(fps) or fps <= 0 or fps > 100:
        raise ValueError(f"fps must be finite and in (0, 100] for GIF timing resolution, got {fps!r}")

    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = make_synthetic_frames(n_frames=n_frames, size=size, seed=seed)
    images = [Image.fromarray(to_uint8(frame)).convert("P") for frame in frames]
    images[0].save(
        path,
        format="GIF",
        save_all=True,
        append_images=images[1:],
        duration=max(10, int(math.floor(100.0 / fps + 0.5)) * 10),
        loop=0,
        optimize=False,
    )
    return path
