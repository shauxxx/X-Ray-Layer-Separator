"""Сравнение результата с исходными кадрами: метрики и картинка для глаз."""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from .builtin_separation import attenuation, shift_image


def _edges(image: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(image.astype(np.float32), (0, 0), 1.5)
    edges = cv2.Laplacian(blurred, cv2.CV_32F)
    return edges - edges.mean()


def edge_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Корреляция границ двух слоёв. Около 0 - слои не «просачиваются» друг в друга."""
    ea, eb = _edges(a), _edges(b)
    denom = float(np.sqrt((ea**2).sum() * (eb**2).sum())) + 1e-12
    return float((ea * eb).sum() / denom)


def sharpness(image: np.ndarray) -> float:
    """Энергия градиента (Tenengrad): выше = резче."""
    gx = cv2.Sobel(image.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    return float(np.mean(gx * gx + gy * gy))


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum())) + 1e-12
    return float((a * b).sum() / denom)


def aligned_mean(
    frames: Sequence[np.ndarray],
    shifts_xy: np.ndarray,
    index: np.ndarray,
) -> np.ndarray:
    """Среднее поглощение по кадрам, выровненным по движению одного слоя."""
    acc = np.zeros(frames[0].shape, dtype=np.float32)
    for k in index:
        acc += shift_image(attenuation(frames[k]), -shifts_xy[k, 0], -shifts_xy[k, 1])
    return acc / max(len(index), 1)


def compare_with_frames(
    frames: Sequence[np.ndarray],
    layers: tuple[np.ndarray, np.ndarray],
    shifts: np.ndarray,
    reliable: np.ndarray,
) -> dict[str, float]:
    """Считает метрики качества разделения относительно исходных кадров.

    * ``edge_corr_layers``: чем ближе к 0, тем меньше один слой виден в другом.
    * ``sharpness_layer_*``: резкость каждого слоя (справочно: удобно сравнивать
      разные запуски и алгоритмы между собой на одних и тех же данных).
    * ``corr_with_aligned_mean_layer_*``: насколько слой похож на среднее кадров,
      выровненных по его движению (быстрая проверка здравого смысла: у
      нормального результата заметно выше нуля).
    """
    index = np.flatnonzero(reliable)
    layer_1, layer_2 = layers
    mean_1 = aligned_mean(frames, shifts[:, 0, :], index)
    mean_2 = aligned_mean(frames, shifts[:, 1, :], index)
    return {
        "n_frames": float(len(frames)),
        "n_reliable_frames": float(len(index)),
        "edge_corr_layers": edge_correlation(layer_1, layer_2),
        "sharpness_layer_1": sharpness(layer_1),
        "sharpness_layer_2": sharpness(layer_2),
        "corr_with_aligned_mean_layer_1": _pearson(_edges(layer_1), _edges(mean_1)),
        "corr_with_aligned_mean_layer_2": _pearson(_edges(layer_2), _edges(mean_2)),
    }


def _to_gray8(image: np.ndarray, dense_is_dark: bool) -> np.ndarray:
    lo, hi = np.percentile(image, [0.5, 99.5])
    x = np.clip((image - lo) / max(float(hi - lo), 1e-9), 0.0, 1.0)
    if dense_is_dark:
        x = 1.0 - x
    return np.rint(x * 255.0).astype(np.uint8)


def make_montage(
    frames: Sequence[np.ndarray],
    layers: tuple[np.ndarray, np.ndarray],
    dense_is_dark: bool = True,
) -> np.ndarray:
    """Склейка слева направо: исходный кадр 0 | среднее кадров | layer_1 | layer_2 (uint8)."""
    first = np.rint(np.clip(frames[0], 0.0, 1.0) * 255.0).astype(np.uint8)
    mean = np.mean([attenuation(f) for f in frames], axis=0)
    panels = [
        first,
        _to_gray8(mean, dense_is_dark),
        _to_gray8(layers[0], dense_is_dark),
        _to_gray8(layers[1], dense_is_dark),
    ]
    divider = np.full((first.shape[0], 4), 128, dtype=np.uint8)
    out: list[np.ndarray] = []
    for i, panel in enumerate(panels):
        if i:
            out.append(divider)
        out.append(panel)
    return np.hstack(out)
