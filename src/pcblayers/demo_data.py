"""Синтетическая двухслойная «плата» с известным правильным ответом.

Нужна для тестов и для запуска программы без реальных снимков.

Модель кадра (закон Бера-Ламберта): после перехода к поглощению
``-ln(I)`` вклады слоёв складываются::

    y_k = bottom + shift(top, d_k),   d_k = r * (cos(theta_k), sin(theta_k))

Нижний слой неподвижен, верхний движется по окружности радиуса ``r`` пикселей.
Кадры отдаются в формате общего контракта: ``float32``, ``(H, W)``, ``[0, 1]``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from .preprocessing import to_uint8


@dataclass(frozen=True)
class TwoLayerScene:
    """Кадры и эталон: что должна восстановить программа."""

    frames: list[np.ndarray]      # N кадров float32 (H, W) в [0, 1]
    bottom: np.ndarray            # эталон нижнего слоя (поглощение, больше = плотнее)
    top: np.ndarray               # эталон верхнего слоя в координатах кадра 0
    shifts: np.ndarray            # (N, 2, 2): [кадр][0=нижний, 1=верхний][dx, dy]


def _pcb_pattern(rng: np.random.Generator, size: int, n_traces: int, n_pads: int) -> np.ndarray:
    """Рисует «дорожки» и «контактные площадки» с поглощением 0..~0.6."""
    canvas = np.zeros((size, size), dtype=np.float32)
    for _ in range(n_traces):
        thickness = int(rng.integers(2, 4))
        start = int(rng.integers(8, size - 40))
        length = int(rng.integers(size // 5, size // 2))
        fixed = int(rng.integers(8, size - 8 - thickness))
        if rng.random() < 0.5:
            canvas[fixed:fixed + thickness, start:start + length] += 0.30
        else:
            canvas[start:start + length, fixed:fixed + thickness] += 0.30
    yy, xx = np.mgrid[0:size, 0:size]
    for _ in range(n_pads):
        cy, cx = rng.uniform(10, size - 10, size=2)
        radius = float(rng.uniform(2.5, 5.0))
        canvas += 0.35 * (((yy - cy) ** 2 + (xx - cx) ** 2) <= radius**2)
    canvas = ndimage.gaussian_filter(canvas, 0.8)
    smooth = ndimage.gaussian_filter(rng.normal(0.0, 1.0, (size, size)), 6.0)
    canvas += 0.02 * smooth / max(float(smooth.std()), 1e-6)
    return np.clip(canvas, 0.0, 0.7).astype(np.float32)


def make_two_layer_scene(
    n_frames: int = 24,
    size: int = 192,
    radius_px: float = 8.0,
    step_deg: float = 15.0,
    noise: float = 0.003,
    seed: int = 0,
) -> TwoLayerScene:
    """Строит последовательность кадров: нижний слой стоит, верхний ходит по кругу."""
    if n_frames < 4:
        raise ValueError("n_frames must be >= 4")
    if size < 64:
        raise ValueError("size must be >= 64")
    rng = np.random.default_rng(seed)
    bottom = _pcb_pattern(rng, size, n_traces=14, n_pads=10)
    top = _pcb_pattern(rng, size, n_traces=14, n_pads=10)

    frames: list[np.ndarray] = []
    shifts = np.zeros((n_frames, 2, 2), dtype=np.float64)
    # Кадр 0 - опорный: смещения отсчитываются от него (там верхний слой стоит в (r, 0)).
    dx0, dy0 = radius_px, 0.0
    for k in range(n_frames):
        theta = math.radians(k * step_deg)
        dx, dy = radius_px * math.cos(theta), radius_px * math.sin(theta)
        shifts[k, 1] = (dx - dx0, dy - dy0)
        moved_top = ndimage.shift(top, (dy, dx), order=1, mode="nearest")
        attenuation = bottom + moved_top
        attenuation = attenuation + rng.normal(0.0, noise, attenuation.shape)
        frames.append(np.clip(np.exp(-attenuation), 0.0, 1.0).astype(np.float32))
    top_ref = ndimage.shift(top, (dy0, dx0), order=1, mode="nearest").astype(np.float32)
    return TwoLayerScene(frames=frames, bottom=bottom, top=top_ref, shifts=shifts)


def save_scene_gif(path: str | Path, scene: TwoLayerScene, fps: float = 10.0) -> Path:
    """Сохраняет кадры сцены как GIF (8 бит, как у реальных входных данных)."""
    if not math.isfinite(fps) or fps <= 0 or fps > 100:
        raise ValueError(f"fps must be in (0, 100], got {fps!r}")
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(to_uint8(f)).convert("P") for f in scene.frames]
    images[0].save(
        path,
        format="GIF",
        save_all=True,
        append_images=images[1:],
        duration=max(10, int(round(1000.0 / fps / 10.0)) * 10),
        loop=0,
        optimize=False,
    )
    return path


def make_demo_gif(path: str | Path, **kwargs) -> Path:
    """Короткая обёртка: сгенерировать сцену и записать GIF."""
    fps = kwargs.pop("fps", 10.0)
    return save_scene_gif(path, make_two_layer_scene(**kwargs), fps=fps)
