"""Встроенное разделение слоёв: запасной вариант, пока нет модуля 3.

Метод простой и устойчивый (чередующаяся оценка с медианой), без геометрии лога.

После перехода к поглощению ``y = -ln(I)`` вклады слоёв складываются::

    y_k = B + shift(T, d_top_k)      (нижний слой B стоит, верхний T движется)

Сдвиги ``d`` дал модуль 2 (``motion_analysis.estimate_shifts``). Дальше:

1. Кадры, выровненные по слою, показывают его резким, а другой слой
   размазывается между кадрами. Медиана по кадрам оставляет только то,
   что в выровненных кадрах неподвижно, то есть сам слой.
2. Оценку одного слоя вычитаем из кадров и повторяем для другого.
   Несколько итераций убирают «призрак» второго слоя.

Контракт функции совпадает с тем, что main.py ждёт от модуля 3
(``pcblayers.layer_separation.separate_layers``).
"""

from __future__ import annotations

import logging
from typing import Sequence

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_EPS = 1e-4  # нижняя граница яркости перед логарифмом


def attenuation(frame: np.ndarray) -> np.ndarray:
    """``-ln(I)``: после этого вклады слоёв складываются."""
    return -np.log(np.maximum(frame, _EPS)).astype(np.float32)


def shift_image(image: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Сдвиг содержимого на (dx, dy) пикселей с субпиксельной интерполяцией."""
    h, w = image.shape
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(
        image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
    )


def _median_of_aligned(
    stack: Sequence[np.ndarray],
    shifts: np.ndarray,
    index: np.ndarray,
) -> np.ndarray:
    """Медиана кадров, выровненных обратно на ``-shifts`` (по выбранным кадрам)."""
    aligned = np.empty((len(index),) + stack[0].shape, dtype=np.float32)
    for slot, k in enumerate(index):
        aligned[slot] = shift_image(stack[slot], -shifts[k, 0], -shifts[k, 1])
    return np.median(aligned, axis=0).astype(np.float32)


def separate_layers(
    frames: Sequence[np.ndarray],
    shifts: np.ndarray,
    reliable: np.ndarray | None = None,
    iterations: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Разделяет кадры на неподвижный и движущийся слои.

    Args:
        frames: кадры контракта модуля 1 (float32, (H, W), [0, 1]).
        shifts: ``(N, 2, 2)`` из ``estimate_shifts``: [кадр][0=нижний, 1=верхний][dx, dy].
        reliable: булева маска ``(N,)`` кадров, которым можно верить. ``None`` - все.
        iterations: число проходов чередующейся оценки.

    Returns:
        ``(layer_1, layer_2)``: ``float32 (H, W)``. Значения - поглощение с
        вычтенным постоянным уровнем (больше = плотнее). ``layer_1`` - нижний
        (неподвижный) слой, ``layer_2`` - верхний (движущийся). Оба слоя
        приведены к координатам кадра 0.
    """
    n = len(frames)
    shifts = np.asarray(shifts, dtype=np.float64)
    if shifts.shape != (n, 2, 2):
        raise ValueError(f"shifts must have shape ({n}, 2, 2), got {shifts.shape}")
    if reliable is None:
        reliable = np.ones(n, dtype=bool)
    reliable = np.asarray(reliable, dtype=bool)
    if reliable.shape != (n,):
        raise ValueError(f"reliable must have shape ({n},), got {reliable.shape}")
    index = np.flatnonzero(reliable)
    if len(index) < 3:
        raise ValueError(
            f"Too few reliable frames ({len(index)}); need at least 3 to separate layers"
        )
    if iterations < 1:
        raise ValueError("iterations must be >= 1")

    y = [attenuation(frames[k]) for k in index]
    bottom_shifts = shifts[:, 0, :]
    top_shifts = shifts[:, 1, :]

    top = np.zeros_like(y[0])
    bottom = np.zeros_like(y[0])
    for it in range(iterations):
        # Нижний слой: вычитаем текущую оценку верхнего и выравниваем по нижнему.
        residual = [
            yk - shift_image(top, top_shifts[k, 0], top_shifts[k, 1])
            for yk, k in zip(y, index)
        ]
        bottom = _median_of_aligned(residual, bottom_shifts, index)
        # Верхний слой: вычитаем нижний и выравниваем по верхнему.
        residual = [
            yk - shift_image(bottom, bottom_shifts[k, 0], bottom_shifts[k, 1])
            for yk, k in zip(y, index)
        ]
        top = _median_of_aligned(residual, top_shifts, index)
        logger.debug("builtin separation iteration %d/%d done", it + 1, iterations)

    # Постоянный уровень между слоями неразличим - убираем его из обоих.
    bottom = (bottom - np.median(bottom)).astype(np.float32)
    top = (top - np.median(top)).astype(np.float32)
    return bottom, top
