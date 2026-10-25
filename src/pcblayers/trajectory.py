"""Проверка траектории смещений: отбраковка выбросов по окружности.

При томосинтезе источник ходит по окружности, поэтому смещение каждого слоя
относительно опорного кадра тоже лежит на окружности (радиус пропорционален
высоте слоя). Оценки модуля 2 по отдельным кадрам иногда уходят в сторону
(слабый пик корреляции, совпавший сосед). Окружность, подогнанная по
большинству кадров методом RANSAC, позволяет такие кадры найти.

Неподвижный слой (все смещения почти нулевые) считается вырожденной
окружностью и проверку проходит целиком.
"""

from __future__ import annotations

import numpy as np


def _circle_through(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray):
    """Окружность по трём точкам: (cx, cy, r) или None, если точки на одной прямой."""
    ax, ay = p1
    bx, by = p2
    cx, cy = p3
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-9:
        return None
    ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay)
          + (cx**2 + cy**2) * (ay - by)) / d
    uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx)
          + (cx**2 + cy**2) * (bx - ax)) / d
    return float(ux), float(uy), float(np.hypot(ax - ux, ay - uy))


def fit_circle(points: np.ndarray) -> tuple[float, float, float]:
    """Алгебраическая подгонка окружности (метод Каса) по точкам ``(n, 2)``."""
    x, y = points[:, 0], points[:, 1]
    a = np.column_stack([x, y, np.ones_like(x)])
    b = -(x**2 + y**2)
    (p, q, r), *_ = np.linalg.lstsq(a, b, rcond=None)
    cx, cy = -p / 2.0, -q / 2.0
    radius_sq = cx**2 + cy**2 - r
    return float(cx), float(cy), float(np.sqrt(max(radius_sq, 0.0)))


def circle_inliers(
    points: np.ndarray,
    tolerance_px: float = 2.0,
    static_px: float = 1.5,
    iterations: int = 300,
    seed: int = 0,
) -> tuple[np.ndarray, dict[str, float]]:
    """Какие точки лежат на общей окружности.

    Args:
        points: смещения ``(n, 2)`` (dx, dy), кадр 0 = (0, 0).
        tolerance_px: допустимое расстояние от окружности.
        static_px: если все точки ближе к началу координат, слой неподвижен.

    Returns:
        ``(inlier_mask (n,), info)``, где ``info`` содержит ``cx, cy, r, static``.
    """
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    if n == 0:
        return np.zeros(0, dtype=bool), {"cx": 0.0, "cy": 0.0, "r": 0.0, "static": True}

    # Неподвижный слой: подавляющее большинство точек у нуля.
    near_origin = np.hypot(points[:, 0], points[:, 1]) <= static_px
    if near_origin.mean() >= 0.7:
        return near_origin | (np.hypot(points[:, 0], points[:, 1]) <= static_px * 2), {
            "cx": 0.0, "cy": 0.0, "r": 0.0, "static": True}

    if n < 4:
        return np.ones(n, dtype=bool), {"cx": 0.0, "cy": 0.0, "r": 0.0, "static": False}

    rng = np.random.default_rng(seed)
    best_mask = np.zeros(n, dtype=bool)
    best_score = (-1, 0.0)
    max_radius = 4.0 * float(np.abs(points).max() + 1.0)
    for _ in range(iterations):
        idx = rng.choice(n, size=3, replace=False)
        circle = _circle_through(*points[idx])
        if circle is None or circle[2] > max_radius:
            continue
        cx, cy, r = circle
        dist = np.abs(np.hypot(points[:, 0] - cx, points[:, 1] - cy) - r)
        mask = dist <= tolerance_px
        score = (int(mask.sum()), -float(dist[mask].sum()))
        if score > best_score:
            best_score, best_mask = score, mask

    if best_mask.sum() < 3:
        return np.ones(n, dtype=bool), {"cx": 0.0, "cy": 0.0, "r": 0.0, "static": False}

    cx, cy, r = fit_circle(points[best_mask])
    dist = np.abs(np.hypot(points[:, 0] - cx, points[:, 1] - cy) - r)
    mask = dist <= tolerance_px
    if mask.sum() >= 3:
        cx, cy, r = fit_circle(points[mask])
        dist = np.abs(np.hypot(points[:, 0] - cx, points[:, 1] - cy) - r)
        mask = dist <= tolerance_px
    return mask, {"cx": cx, "cy": cy, "r": r, "static": False}
