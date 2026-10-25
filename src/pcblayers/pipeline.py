"""Единый конвейер: GIF/PNG -> кадры -> смещения -> два слоя -> файлы и отчёт.

Состыковка модулей команды:

* модуль 1: ``gif_loader.load_sequence``, ``preprocessing.validate_frames``;
* модуль 2: ``motion_analysis.estimate_shifts`` -> ``shifts (N, 2, 2)``, ``strength (N, 2)``;
* модуль 3: ``layer_separation.separate_layers`` (когда файл появится в ветке),
  иначе используется встроенный запасной алгоритм ``builtin_separation``.

Контракт для модуля 3 описан в README (раздел «Контракт модуля 3»).
"""

from __future__ import annotations

import importlib
import inspect
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from . import builtin_separation
from .compare import compare_with_frames, make_montage
from .gif_loader import load_sequence
from .motion_analysis import estimate_shifts
from .outputs import plot_shifts, save_layers, save_report, save_shifts_csv
from .preprocessing import normalize, save_frames, validate_frames
from .trajectory import circle_inliers

logger = logging.getLogger(__name__)

BACKENDS = ("auto", "builtin", "module3")
MIN_FRAMES = 4


class PipelineError(RuntimeError):
    """Ошибка, понятная пользователю (плохой вход, нечего разделять и т. п.)."""


@dataclass
class PipelineConfig:
    pattern: str = "*.png"            # маска файлов, если вход - папка с PNG
    normalize: str = "none"           # none | global | per_frame | percentile
    max_size: int = 1024              # размер кадров для разделения слоёв, px (0 = как есть)
    motion_max_size: int = 0          # размер кадров для поиска смещений (0 = как есть)
    backend: str = "auto"             # auto | builtin | module3
    min_peak: float = 0.005           # абсолютный порог надёжности пика корреляции
    rel_peak: float = 0.4             # порог относительно медианы пиков слоя
    trajectory_tol_px: float = 3.0    # допуск отклонения от окружности траектории, px
    bottom_tolerance_px: float = 3.0  # допустимое «дрожание» нижнего слоя
    iterations: int = 4               # итерации встроенного алгоритма
    dense_is_dark: bool = True        # плотный материал тёмный на выходных PNG
    save_debug_frames: bool = False   # сохранить кадры, которые видит конвейер
    make_plots: bool = True           # график смещений
    log_path: Path | None = None      # лог томографа (нужен модулю 3); None = поиск рядом со входом


@dataclass
class PipelineResult:
    layer_1: np.ndarray
    layer_2: np.ndarray
    shifts: np.ndarray
    strength: np.ndarray
    reliable: np.ndarray
    backend_used: str
    metrics: dict[str, float]
    report: dict
    files: dict[str, Path] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Подготовка
# --------------------------------------------------------------------------- #
def resize_frames(frames: list[np.ndarray], max_size: int) -> tuple[list[np.ndarray], float]:
    """Уменьшает кадры так, чтобы длинная сторона была не больше ``max_size``."""
    if max_size <= 0:
        return frames, 1.0
    height, width = frames[0].shape
    longest = max(height, width)
    if longest <= max_size:
        return frames, 1.0
    scale = max_size / longest
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized = [
        cv2.resize(f, size, interpolation=cv2.INTER_AREA).astype(np.float32, copy=False)
        for f in frames
    ]
    logger.info("Frames resized %dx%d -> %dx%d (scale %.3f)", width, height, *size, scale)
    return resized, scale


def reliability_mask(
    strength: np.ndarray,
    min_peak: float = 0.005,
    rel_peak: float = 0.4,
    shifts: np.ndarray | None = None,
    trajectory_tol_px: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Какие кадры можно использовать.

    Оценка смещения слоя ненадёжна, если высота пика фазовой корреляции ниже
    ``max(min_peak, rel_peak * медиана)``. Если переданы ``shifts``, кадр
    дополнительно должен лежать на общей окружности траектории слоя
    (см. ``trajectory.circle_inliers``). Кадр 0 (опорный) надёжен всегда.

    Returns:
        ``(reliable (N,), per_layer (N, 2))``.
    """
    strength = np.asarray(strength, dtype=np.float64)
    if strength.ndim != 2 or strength.shape[1] != 2:
        raise ValueError(f"strength must have shape (N, 2), got {strength.shape}")
    per_layer = np.ones(strength.shape, dtype=bool)
    for layer in range(2):
        tail = strength[1:, layer]
        median = float(np.median(tail)) if tail.size else 0.0
        threshold = max(min_peak, rel_peak * median)
        per_layer[:, layer] = strength[:, layer] >= threshold
        if shifts is not None:
            inliers, _ = circle_inliers(np.asarray(shifts)[:, layer, :], trajectory_tol_px)
            per_layer[:, layer] &= inliers
    per_layer[0] = True
    return per_layer.all(axis=1), per_layer


def bottom_layer_stability(
    shifts: np.ndarray,
    reliable: np.ndarray,
    tolerance_px: float,
) -> dict[str, float | bool]:
    """Проверка допущения «нижний слой стоит на месте»."""
    bottom = shifts[reliable, 0, :]
    magnitude = np.hypot(bottom[:, 0], bottom[:, 1])
    maximum = float(magnitude.max()) if magnitude.size else 0.0
    return {
        "max_bottom_shift_px": maximum,
        "mean_bottom_shift_px": float(magnitude.mean()) if magnitude.size else 0.0,
        "stable": bool(maximum <= tolerance_px),
        "tolerance_px": float(tolerance_px),
    }


# --------------------------------------------------------------------------- #
# Выбор алгоритма разделения
# --------------------------------------------------------------------------- #
Separator = Callable[..., tuple[np.ndarray, np.ndarray]]


def _load_module3() -> Separator | None:
    """Ищет ``pcblayers.layer_separation.separate_layers`` (модуль 3)."""
    try:
        module = importlib.import_module(".layer_separation", package=__package__)
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.endswith("layer_separation"):
            return None
        raise  # модуль 3 есть, но ему не хватает зависимости: пусть это будет видно
    function = getattr(module, "separate_layers", None)
    if not callable(function):
        raise PipelineError(
            "pcblayers/layer_separation.py exists but has no function "
            "separate_layers(frames, shifts=..., reliable=...) -> (layer_1, layer_2)"
        )
    return function


def _call_separator(
    function: Separator,
    frames: list[np.ndarray],
    extras: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    """Вызывает разделитель, передавая только те аргументы, которые он принимает."""
    parameters = inspect.signature(function).parameters
    takes_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())
    kwargs = {k: v for k, v in extras.items() if takes_kwargs or k in parameters}
    return function(frames, **kwargs)


def _validate_layers(result: object, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise PipelineError("separate_layers must return exactly two arrays (layer_1, layer_2)")
    layers = []
    for number, layer in enumerate(result, start=1):
        array = np.asarray(layer, dtype=np.float32)
        if array.shape != shape:
            raise PipelineError(
                f"layer_{number} has shape {array.shape}, expected {shape} (frame size)"
            )
        if not np.isfinite(array).all():
            raise PipelineError(f"layer_{number} contains NaN or infinite values")
        layers.append(array)
    return layers[0], layers[1]


def find_log(input_path: str | Path, explicit: Path | None = None) -> Path | None:
    """Лог томографа: явный путь или единственный ``*.log`` в папке с проекциями."""
    if explicit is not None:
        if not Path(explicit).is_file():
            raise PipelineError(f"Log file not found: {explicit}")
        return Path(explicit)
    folder = Path(input_path)
    if folder.is_dir():
        candidates = sorted(folder.glob("*.log"))
        if len(candidates) == 1:
            return candidates[0]
    return None


def _resolve_backend(name: str) -> tuple[str, Separator | None]:
    if name not in BACKENDS:
        raise PipelineError(f"Unknown backend {name!r}; choose one of {BACKENDS}")
    if name == "builtin":
        return "builtin", None
    function = _load_module3()
    if function is not None:
        return "module3", function
    if name == "module3":
        raise PipelineError(
            "Backend 'module3' requested, but pcblayers/layer_separation.py is not "
            "in the package yet. Merge participant 3's branch or use --backend builtin."
        )
    logger.info("layer_separation.py not found: using the built-in fallback separator")
    return "builtin", None


# --------------------------------------------------------------------------- #
# Конвейер
# --------------------------------------------------------------------------- #
def run_pipeline(
    input_path: str | Path,
    out_dir: str | Path,
    config: PipelineConfig | None = None,
) -> PipelineResult:
    """Выполняет весь конвейер и записывает результаты в ``out_dir``."""
    config = config or PipelineConfig()
    out_dir = Path(out_dir)
    timings: dict[str, float] = {}

    backend_used, separator = _resolve_backend(config.backend)
    log_path = find_log(input_path, config.log_path) if separator is not None else None
    if config.backend == "auto" and separator is not None and log_path is None:
        # Алгоритму модуля 3 нужна геометрия съёмки из лога. Без неё он молча
        # взял бы чужие значения по умолчанию, поэтому в режиме auto не рискуем.
        logger.info("No tomograph log found (use --log): using the built-in separator")
        backend_used, separator = "builtin", None

    # 1. Загрузка и проверка кадров (модуль 1)
    t0 = time.perf_counter()
    frames = load_sequence(input_path, pattern=config.pattern)
    if config.normalize != "none":
        frames = normalize(frames, mode=config.normalize)
    stats = validate_frames(frames)
    if len(frames) < MIN_FRAMES:
        raise PipelineError(
            f"Need at least {MIN_FRAMES} frames to separate layers, got {len(frames)}. "
            "Pass a GIF with the projection sequence or a folder with numbered PNGs."
        )
    original_shape = frames[0].shape
    motion_frames, motion_scale = resize_frames(frames, config.motion_max_size)
    frames, scale = resize_frames(frames, config.max_size)
    shape = frames[0].shape
    timings["load_s"] = time.perf_counter() - t0
    if config.save_debug_frames:
        save_frames(frames, out_dir / "debug_frames", validate=False)

    # 2. Смещения между кадрами (модуль 2).
    # Слои разнесены всего на десятки пикселей, поэтому смещения ищем на полном
    # разрешении (иначе два пика корреляции сливаются), а затем пересчитываем
    # к размеру кадров, на которых идёт разделение.
    t0 = time.perf_counter()
    shifts_motion, strength = estimate_shifts(motion_frames)
    shifts_motion = np.asarray(shifts_motion, dtype=np.float64)
    strength = np.asarray(strength, dtype=np.float64)
    del motion_frames
    if shifts_motion.shape != (len(frames), 2, 2) or strength.shape != (len(frames), 2):
        raise PipelineError(
            f"estimate_shifts returned shapes {shifts_motion.shape} and {strength.shape}; "
            f"expected ({len(frames)}, 2, 2) and ({len(frames)}, 2)"
        )
    reliable, per_layer = reliability_mask(
        strength, config.min_peak, config.rel_peak,
        shifts=shifts_motion, trajectory_tol_px=config.trajectory_tol_px,
    )
    shifts = shifts_motion * (scale / motion_scale)
    stability = bottom_layer_stability(
        shifts, reliable, config.bottom_tolerance_px * (scale / motion_scale)
    )
    timings["motion_s"] = time.perf_counter() - t0
    n_reliable = int(reliable.sum())
    logger.info("Reliable frames: %d of %d", n_reliable, len(frames))
    if n_reliable < 3:
        raise PipelineError(
            f"Only {n_reliable} frames have a reliable shift estimate (need >= 3). "
            "Lower --min-peak, or check that the input is a projection sequence "
            "with a visibly moving upper layer."
        )
    if not stability["stable"]:
        logger.warning(
            "Bottom layer is not static: max shift %.1f px > tolerance %.1f px",
            stability["max_bottom_shift_px"], config.bottom_tolerance_px,
        )

    # 3. Разделение слоёв (модуль 3 или запасной алгоритм)
    t0 = time.perf_counter()
    if separator is None:
        layers = builtin_separation.separate_layers(
            frames, shifts, reliable=reliable, iterations=config.iterations
        )
    else:
        layers = _call_separator(
            separator,
            frames,
            {
                "shifts": shifts,
                "reliable": reliable,
                "strength": strength,
                "log_path": log_path,
                "out_dir": out_dir,
                "scale": scale,
            },
        )
    layer_1, layer_2 = _validate_layers(layers, shape)
    timings["separation_s"] = time.perf_counter() - t0

    # 4. Файлы, сравнение с исходными кадрами, отчёт
    t0 = time.perf_counter()
    files = save_layers((layer_1, layer_2), out_dir, config.dense_is_dark)
    files["shifts_csv"] = save_shifts_csv(shifts, strength, reliable, out_dir / "shifts.csv")
    if config.make_plots:
        plot_path = plot_shifts(shifts, strength, reliable, out_dir / "shifts.png")
        if plot_path is not None:
            files["shifts_plot"] = plot_path
    metrics = compare_with_frames(frames, (layer_1, layer_2), shifts, reliable)
    montage_path = out_dir / "comparison.png"
    montage = make_montage(frames, (layer_1, layer_2), config.dense_is_dark)
    if not cv2.imwrite(str(montage_path), montage):
        raise OSError(f"Could not write image: {montage_path}")
    files["comparison"] = montage_path

    report = {
        "input": str(input_path),
        "backend": backend_used,
        "frames": stats["n_frames"],
        "original_shape_hw": list(original_shape),
        "processed_shape_hw": list(shape),
        "scale": scale,
        "motion_scale": motion_scale,
        "reliable_frames": n_reliable,
        "unreliable_frame_indices": [int(i) for i in np.flatnonzero(~reliable)],
        "bottom_layer": stability,
        "metrics": metrics,
        "timings_s": timings,
        "layer_1": "неподвижная составляющая (нижний слой)",
        "layer_2": "движущаяся составляющая (верхний слой)",
    }
    files["report"] = save_report(report, out_dir / "report.json")
    timings["output_s"] = time.perf_counter() - t0
    report["files"] = {k: str(v) for k, v in files.items()}
    save_report(report, out_dir / "report.json")

    return PipelineResult(
        layer_1=layer_1,
        layer_2=layer_2,
        shifts=shifts,
        strength=strength,
        reliable=reliable,
        backend_used=backend_used,
        metrics=metrics,
        report=report,
        files=files,
    )
