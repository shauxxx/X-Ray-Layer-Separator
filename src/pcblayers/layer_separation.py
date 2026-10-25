"""Разделение слоёв (модуль 3): обёртка над скриптом ``reconstruction.py``.

``reconstruction.py`` - скрипт участника 3 (``new_claudecode.py``), скопированный
без изменений. Он умеет всё нужное (разбор лога, автокалибровка геометрии,
точное разделение слоёв в частотной области), но запускается как отдельная
программа с настройками в константах. Эта обёртка даёт ему функцию, которую
вызывает конвейер (``pipeline.py``), и раздаёт кадры, уже загруженные модулем 1.

Контракт ``separate_layers`` описан в README. Алгоритму нужна геометрия съёмки
из лога томографа (``--log``): шаг поворота, угол, ΔZ, размер пикселя.
Смещения модуля 2 он не использует: положения слоёв подбираются по самим кадрам.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np

from . import reconstruction as tomo

logger = logging.getLogger(__name__)


def separate_layers(
    frames: Sequence[np.ndarray],
    shifts: np.ndarray | None = None,       # не используется: геометрия берётся из лога
    reliable: np.ndarray | None = None,     # не используется
    strength: np.ndarray | None = None,     # не используется
    log_path: str | Path | None = None,
    out_dir: str | Path | None = None,
    scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Разделяет кадры на два слоя по геометрии томосинтеза.

    Args:
        frames: кадры контракта модуля 1 (float32, (H, W), [0, 1]).
        log_path: лог томографа (INI). Без него берутся значения по умолчанию
            из ``reconstruction.parse_geometry``, они могут не подойти вашей установке.
        out_dir: куда писать отладочный график резкости (``module3_debug/``).
        scale: во сколько раз кадры уменьшены относительно снимка из лога.

    Returns:
        ``(layer_1, layer_2)``: ``float32 (H, W)``, больше = плотнее.
        ``layer_1`` - нижний слой (меньший радиус смещения r), ``layer_2`` - верхний.
    """
    n = len(frames)
    if n < 4:
        raise ValueError(f"Need at least 4 frames, got {n}")
    if not scale > 0:
        raise ValueError(f"scale must be > 0, got {scale!r}")

    log = Path(log_path) if log_path else Path("<лог не указан>")
    geometry = tomo.parse_geometry(log, n)
    if not log.is_file():
        logger.warning(
            "Лог съёмки не указан или не найден (%s): используются значения по умолчанию "
            "(шаг %.1f°, угол %.2f°, ΔZ %.3f, пиксель %.1f мкм). Передайте --log.",
            log, geometry["step"], geometry["angle"], geometry["delta_z"],
            geometry["pix_mm"] * 1000,
        )
    # Размер пикселя в логе относится к полному кадру; кадры могли быть уменьшены.
    geometry["pix_mm"] /= scale

    if out_dir is not None:
        tomo.OUTPUT_DIR = Path(out_dir) / "module3_debug"

    attenuation = [tomo.attenuation(f.astype(np.float32, copy=False)) for f in frames]
    log_sense = 1 if geometry["cw"] else -1
    sense, phi0, radii, _ = tomo.calibrate(attenuation, n, geometry, log_sense)
    logger.info("Модуль 3: направление %+d, φ0=%.1f°, радиусы слоёв %s px",
                sense, phi0, [round(r, 2) for r in radii])

    stack = tomo.make_stack(attenuation, 1.0, tomo.SIGMA_BG * scale, 0.0)
    thetas = tomo.thetas_for(n, geometry["step"], sense, phi0)
    spectra = tomo.Spectra(stack, 1.0, max(abs(r) for r in radii), store=False)
    _, layers = spectra.solve(thetas, radii, tomo.REGULARIZATION, images=True)

    # radii отсортированы по возрастанию; «верхний» - тот, у кого r больше
    # (переключатель TOP_IS_LARGER_R из скрипта участника 3).
    if not tomo.TOP_IS_LARGER_R:
        layers = layers[::-1]
    return layers[0].astype(np.float32), layers[-1].astype(np.float32)
