"""Модуль 2: анализ движения слоёв между кадрами.

Вход:  список кадров (H, W), float, яркость 0..1 (результат модуля 1).
Выход: объект MotionResult:
    shifts      (N, 2, 2)  смещения [кадр][слой][dx, dy]; слой 0 = нижний (неподвижный),
                           слой 1 = верхний (движущийся); NaN, если пик не найден
    strength    (N, 2)     сила пика корреляции (z-оценка) для каждого слоя
    reliable    (N,)       True, если кадру можно доверять
    relative    (N, 2)     смещение верхнего слоя относительно нижнего (dx, dy);
                           в ненадёжных кадрах заполнено по модели траектории

Контракт из плана для участника 3: это `relative` (N, 2) и `reliable` (N,).
Расширенный формат (N, 2, 2) лежит в `shifts`.

Соглашения:
    * Смещение кадра k считается относительно опорного кадра (по умолчанию 0).
    * dx > 0: содержимое кадра k сдвинуто вправо по сравнению с опорным;
      dy > 0: вниз (ось y направлена вниз, как строки массива).
    * Единицы: пиксели кадра, который получил модуль.

Алгоритм:
    1. Для каждого кадра считается фазовая корреляция с опорным кадром
       (на карте градиентов, чтобы работали границы дорожек, а не фон).
    2. На поверхности корреляции ищется несколько пиков с субпиксельным уточнением.
    3. Нижний слой = пик, который во всех кадрах стоит на одном месте (мода по кадрам).
       Верхний слой = самый сильный из остальных пиков.
    4. Траектория верхнего слоя подгоняется эллипсом (RANSAC):
       dx(k) = cx + ax*cos(θ_k) + bx*sin(θ_k), то же для dy.
       Кадры, не попавшие на траекторию, помечаются ненадёжными.
"""
from __future__ import annotations

import warnings
from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy import fft as sfft
from scipy.ndimage import gaussian_filter, shift as nd_shift, sobel


# ----------------------------------------------------------------------------
# Результат
# ----------------------------------------------------------------------------
@dataclass
class MotionResult:
    shifts: np.ndarray            # (N, 2, 2) [кадр][слой][dx, dy], NaN если не найдено
    strength: np.ndarray          # (N, 2) z-оценка пика, NaN для опорного кадра
    reliable: np.ndarray          # (N,) bool
    relative: np.ndarray          # (N, 2) верхний слой минус нижний, с заполнением по модели
    relative_model: np.ndarray | None  # (N, 2) значения подогнанной траектории
    model: dict | None            # параметры траектории (coef, rms_px, n_inliers, ...)
    static_center: np.ndarray | None   # (2,) где стоит пик нижнего слоя
    static_std: float             # разброс нижнего слоя, px (мера его стабильности)
    ref_index: int


# ----------------------------------------------------------------------------
# Подготовка кадров
# ----------------------------------------------------------------------------
def _check_frames(frames) -> list[np.ndarray]:
    if len(frames) < 4:
        raise ValueError(f"нужно минимум 4 кадра, получено {len(frames)}")
    out, shape = [], None
    for i, f in enumerate(frames):
        a = np.asarray(f, dtype=np.float32)
        if a.ndim != 2:
            raise ValueError(f"кадр {i}: ожидался 2D массив (H, W), получено {a.shape}")
        if shape is None:
            shape = a.shape
        elif a.shape != shape:
            raise ValueError(f"кадр {i}: размер {a.shape}, а у кадра 0 {shape}")
        if not np.isfinite(a).all():
            raise ValueError(f"кадр {i}: есть NaN или inf")
        out.append(a)
    return out


def _window(shape) -> np.ndarray:
    return np.outer(np.hanning(shape[0]), np.hanning(shape[1])).astype(np.float32)


def _prep(img: np.ndarray, smooth_sigma: float, win: np.ndarray) -> np.ndarray:
    """Сглаживание -> модуль градиента -> вычитание среднего -> окно Ханна."""
    if smooth_sigma > 0:
        img = gaussian_filter(img, smooth_sigma)
    g = np.hypot(sobel(img, axis=0), sobel(img, axis=1))
    g -= g.mean()
    return g * win


# ----------------------------------------------------------------------------
# Фазовая корреляция и поиск пиков
# ----------------------------------------------------------------------------
def _corr_surface(spec_a: np.ndarray, spec_ref: np.ndarray) -> np.ndarray:
    """Пик стоит в точке (dy, dx), если кадр a = опорный кадр, сдвинутый на (dy, dx)."""
    r = spec_a * np.conj(spec_ref)
    mag = np.abs(r)
    r /= np.maximum(mag, 1e-8 * float(mag.max()) + 1e-30)
    return sfft.ifft2(r, workers=-1).real


def _parabola(cm: float, c0: float, cp: float) -> float:
    d = cm - 2.0 * c0 + cp
    if abs(d) < 1e-20:
        return 0.0
    return float(np.clip(0.5 * (cm - cp) / d, -0.5, 0.5))


def _wrap(v: float, n: int) -> float:
    v = v % n
    return v - n if v > n / 2 else v


def _find_peaks(c: np.ndarray, k: int, radius: int):
    """k сильнейших пиков: список (x, y, z), z = (пик - медиана) / робастный σ."""
    med = float(np.median(c))
    sigma = 1.4826 * float(np.median(np.abs(c - med)))
    if sigma < 1e-12:                       # плоский кадр, корреляции нет
        return []
    h, w = c.shape
    work = c.copy()
    peaks = []
    for _ in range(k):
        iy, ix = divmod(int(np.argmax(work)), w)
        val = float(c[iy, ix])
        dx = _parabola(c[iy, (ix - 1) % w], val, c[iy, (ix + 1) % w])
        dy = _parabola(c[(iy - 1) % h, ix], val, c[(iy + 1) % h, ix])
        peaks.append((_wrap(ix + dx, w), _wrap(iy + dy, h), (val - med) / sigma))
        ys = np.arange(iy - radius, iy + radius + 1) % h
        xs = np.arange(ix - radius, ix + radius + 1) % w
        work[np.ix_(ys, xs)] = -np.inf
    return peaks


def _static_center(cands, cell: float):
    """Нижний слой: место, где пики встречаются почти в каждом кадре (мода)."""
    pts = np.array([(x, y) for fr in cands for (x, y, _) in fr], dtype=float)
    if len(pts) == 0:
        return None
    keys = np.round(pts / cell).astype(int)
    cnt = Counter(map(tuple, keys))

    def score(key):
        return sum(cnt.get((key[0] + i, key[1] + j), 0) for i in (-1, 0, 1) for j in (-1, 0, 1))

    best = np.array(max(cnt, key=score), dtype=float) * cell
    near = pts[np.hypot(pts[:, 0] - best[0], pts[:, 1] - best[1]) <= 2 * cell]
    return np.median(near, axis=0)


# ----------------------------------------------------------------------------
# Траектория верхнего слоя
# ----------------------------------------------------------------------------
def _design(theta: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones_like(theta), np.cos(theta), np.sin(theta)])


def _fit_trajectory(theta, rel, mask, tol, rng, n_trials=300, min_inlier_frac=0.5):
    """RANSAC + МНК: rel(k) = c + a*cosθ + b*sinθ по каждой оси."""
    idx = np.flatnonzero(mask)
    if len(idx) < 5:
        return None
    a_mat = _design(theta)
    best, best_n = None, 0
    for _ in range(n_trials):
        tri = rng.choice(idx, 3, replace=False)
        try:
            coef = np.linalg.solve(a_mat[tri], rel[tri])
        except np.linalg.LinAlgError:
            continue
        r = np.hypot(*(rel[idx] - a_mat[idx] @ coef).T)
        n = int((r <= tol).sum())
        if n > best_n:
            best, best_n = coef, n
    if best is None or best_n < max(5, min_inlier_frac * len(idx)):
        return None
    coef = best
    for _ in range(3):                       # уточнение МНК по инлайерам
        r = np.hypot(*(rel[idx] - a_mat[idx] @ coef).T)
        inl = idx[r <= tol]
        if len(inl) < 4:
            break
        coef, *_ = np.linalg.lstsq(a_mat[inl], rel[inl], rcond=None)
    pred = a_mat @ coef
    res = np.full(len(theta), np.inf)
    res[idx] = np.hypot(*(rel[idx] - pred[idx]).T)
    inlier = res <= tol
    return {
        "coef": coef,                        # (3, 2): строки [const, cos, sin], столбцы [x, y]
        "center": coef[0],
        "pred": pred,
        "residual": res,
        "inlier": inlier,
        "rms_px": float(np.sqrt(np.mean(res[inlier] ** 2))),
        "n_inliers": int(inlier.sum()),
        "amplitude_px": float(np.max(np.hypot(*(pred - coef[0]).T))),
    }


# ----------------------------------------------------------------------------
# Главная функция
# ----------------------------------------------------------------------------
def estimate_shifts(
    frames,
    *,
    ref_index: int = 0,
    angle_step_deg: float = 10.0,
    angles_deg=None,
    smooth_sigma: float = 1.0,
    min_snr: float = 7.0,
    peak_radius: int = 4,
    n_candidates: int = 4,
    static_tol: float = 4.0,
    cluster_cell: float = 3.0,
    fit_tol_px: float = 2.0,
    seed: int = 0,
) -> MotionResult:
    """Оценивает смещения слоёв во всех кадрах относительно опорного.

    frames          список/массив кадров (H, W), float 0..1
    ref_index       номер опорного кадра
    angle_step_deg  шаг угла между кадрами (10° по логу съёмки); angles_deg переопределяет
    smooth_sigma    сглаживание перед градиентом, px (гасит шум и квантование GIF)
    min_snr         минимальная z-оценка пика, чтобы считать его настоящим
    peak_radius     слои ближе этого расстояния (px) неразличимы ("слились")
    static_tol      допуск (px) для пика нижнего слоя вокруг его типичного положения
    fit_tol_px      допуск (px) отклонения кадра от подогнанной траектории
    """
    imgs = _check_frames(frames)
    n = len(imgs)
    if not 0 <= ref_index < n:
        raise ValueError(f"ref_index={ref_index} вне диапазона 0..{n - 1}")
    angles = np.asarray(angles_deg, float) if angles_deg is not None else np.arange(n) * angle_step_deg
    if len(angles) != n:
        raise ValueError("длина angles_deg должна совпадать с числом кадров")
    theta = np.deg2rad(angles)

    # 1. корреляция каждого кадра с опорным, поиск пиков
    win = _window(imgs[0].shape)
    spec_ref = sfft.fft2(_prep(imgs[ref_index], smooth_sigma, win), workers=-1)
    cands = [[] for _ in range(n)]
    for k in range(n):
        if k == ref_index:
            continue
        spec = sfft.fft2(_prep(imgs[k], smooth_sigma, win), workers=-1)
        peaks = _find_peaks(_corr_surface(spec, spec_ref), n_candidates, peak_radius)
        cands[k] = [p for p in peaks if p[2] >= min_snr]

    # 2. где стоит нижний (неподвижный) слой
    shifts = np.full((n, 2, 2), np.nan)
    strength = np.full((n, 2), np.nan)
    shifts[ref_index] = 0.0
    reliable = np.zeros(n, bool)
    reliable[ref_index] = True
    c0 = _static_center(cands, cluster_cell)
    if c0 is None:
        warnings.warn("ни в одном кадре не найдено уверенных пиков корреляции; "
                      "проверьте кадры или уменьшите min_snr", RuntimeWarning)
        rel = np.full((n, 2), np.nan)
        rel[ref_index] = 0.0
        return MotionResult(shifts, strength, reliable, rel, None, None, None, float("nan"), ref_index)

    # 3. раздаём пики по слоям
    merged = np.zeros(n, bool)
    for k in range(n):
        if k == ref_index:
            continue
        near = [p for p in cands[k] if np.hypot(p[0] - c0[0], p[1] - c0[1]) <= static_tol]
        if not near:
            continue                                   # нижний слой не найден
        st = max(near, key=lambda p: p[2])
        shifts[k, 0], strength[k, 0] = st[:2], st[2]
        others = [p for p in cands[k]
                  if p is not st and np.hypot(p[0] - st[0], p[1] - st[1]) >= peak_radius]
        if others:
            tp = max(others, key=lambda p: p[2])
            shifts[k, 1], strength[k, 1] = tp[:2], tp[2]
        else:
            merged[k] = True                           # верхний слой слился с нижним

    have = np.isfinite(shifts[:, 0, 0]) & np.isfinite(shifts[:, 1, 0]) & ~merged
    have[ref_index] = True
    rel = np.full((n, 2), np.nan)
    rel[have] = shifts[have, 1] - shifts[have, 0]

    # 4. траектория верхнего слоя и флаги надёжности
    model = _fit_trajectory(theta, rel, have, fit_tol_px, np.random.default_rng(seed))
    if model is not None:
        reliable = have & model["inlier"]
        relative_model = model["pred"]
    else:
        reliable = have.copy()
        relative_model = None
    reliable[ref_index] = True

    relative = rel.copy()
    if relative_model is not None:
        relative[~reliable] = relative_model[~reliable]
    else:
        relative[~reliable] = np.nan
    relative[ref_index] = 0.0

    use = reliable.copy()
    use[ref_index] = False
    if use.sum() >= 2:
        s = shifts[use, 0]
        static_std = float(np.hypot(s[:, 0].std(), s[:, 1].std()))
    else:
        static_std = float("nan")

    if model is not None:
        model = {k: v for k, v in model.items() if k not in ("pred", "residual", "inlier")}
    return MotionResult(shifts, strength, reliable, relative, relative_model,
                        model, c0, static_std, ref_index)


# ----------------------------------------------------------------------------
# Визуализация
# ----------------------------------------------------------------------------
def plot_shifts(result: MotionResult, path: str | None = None, show: bool = False):
    """Три графика: dx, dy по кадрам; траектория; сила пиков. Сохраняет в path."""
    import matplotlib.pyplot as plt

    n = len(result.reliable)
    ks = np.arange(n)
    rel = np.where(result.reliable[:, None], result.relative, np.nan)
    bad = ~result.reliable

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))

    ax[0].plot(ks, rel[:, 0], "o-", label="верхний слой, dx", ms=4)
    ax[0].plot(ks, rel[:, 1], "s-", label="верхний слой, dy", ms=4)
    if result.relative_model is not None:
        ax[0].plot(ks, result.relative_model[:, 0], ":", c="C0", alpha=.6)
        ax[0].plot(ks, result.relative_model[:, 1], ":", c="C1", alpha=.6)
    ax[0].plot(ks, result.shifts[:, 0, 0], "--", c="gray", label="нижний слой, dx")
    ax[0].plot(ks, result.shifts[:, 0, 1], "--", c="k", label="нижний слой, dy")
    for k in ks[bad]:
        ax[0].axvspan(k - .5, k + .5, color="red", alpha=.15)
    ax[0].set(xlabel="номер кадра", ylabel="смещение, px", title="Смещения по кадрам (красное = ненадёжно)")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=.3)

    ax[1].plot(rel[:, 0], rel[:, 1], "o", ms=4)
    if result.relative_model is not None:
        m = result.relative_model
        ax[1].plot(np.r_[m[:, 0], m[0, 0]], np.r_[m[:, 1], m[0, 1]], ":", c="C1")
    ax[1].invert_yaxis()
    ax[1].set(xlabel="dx, px", ylabel="dy, px", title="Траектория верхнего слоя")
    ax[1].set_aspect("equal", adjustable="datalim")
    ax[1].grid(alpha=.3)

    w = 0.4
    colors = np.where(result.reliable, "C2", "C3")
    ax[2].bar(ks - w / 2, np.nan_to_num(result.strength[:, 0]), w, label="нижний", color="gray")
    ax[2].bar(ks + w / 2, np.nan_to_num(result.strength[:, 1]), w, label="верхний", color=colors)
    ax[2].set(xlabel="номер кадра", ylabel="сила пика (z)", title="Надёжность пиков")
    ax[2].legend(fontsize=8)

    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
    if show:
        plt.show()
    plt.close(fig)


# ----------------------------------------------------------------------------
# Синтетические данные (для тестов и отладки)
# ----------------------------------------------------------------------------
def make_synthetic_frames(n=36, shape=(300, 400), amplitude=15.0, step_deg=10.0,
                          noise=0.0, levels=None, seed=0):
    """Два текстурных слоя: нижний стоит, верхний ходит по окружности радиуса amplitude.

    Возвращает (frames, true_rel): true_rel (N, 2) = смещение верхнего слоя
    относительно кадра 0 (dx, dy); нижний слой не двигается.
    levels: число уровней яркости (например 256 или 64) для имитации GIF.
    """
    rng = np.random.default_rng(seed)

    def texture():
        t = gaussian_filter(rng.random(shape).astype(np.float32), 1.2)
        return (t - t.min()) / (t.max() - t.min())

    bottom, top = texture(), texture()
    ang = np.deg2rad(np.arange(n) * step_deg)
    pos = amplitude * np.column_stack([np.cos(ang), np.sin(ang)])
    frames = []
    for p in pos:
        f = 0.5 * bottom + 0.5 * nd_shift(top, (p[1], p[0]), order=3, mode="wrap")
        if noise > 0:
            f = f + rng.normal(0.0, noise, f.shape)
        f = (f - f.min()) / (f.max() - f.min())
        if levels:
            f = np.round(f * (levels - 1)) / (levels - 1)
        frames.append(f.astype(np.float32))
    return frames, pos - pos[0]


if __name__ == "__main__":          # быстрая демонстрация: python motion_analysis.py
    import os

    frames, true_rel = make_synthetic_frames(noise=0.02, levels=256)
    res = estimate_shifts(frames)
    ok = res.reliable
    err = np.hypot(*(res.relative[ok] - true_rel[ok]).T)
    print(f"надёжных кадров: {ok.sum()} из {len(ok)}")
    print(f"ошибка по надёжным: средняя {err.mean():.2f} px, макс {err.max():.2f} px")
    print(f"разброс нижнего слоя: {res.static_std:.2f} px")
    os.makedirs("output", exist_ok=True)
    plot_shifts(res, "output/shifts_demo.png")
    print("график: output/shifts_demo.png")