"""Запись результатов: слои, смещения, графики, отчёт."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def layer_to_uint8(layer: np.ndarray, dense_is_dark: bool = True) -> np.ndarray:
    """Растяжение по перцентилям -> 8 бит. Плотный материал тёмный, как на снимках."""
    lo, hi = np.percentile(layer, [0.5, 99.5])
    x = np.clip((layer - lo) / max(float(hi - lo), 1e-9), 0.0, 1.0)
    if dense_is_dark:
        x = 1.0 - x
    return np.rint(x * 255.0).astype(np.uint8)


def layer_to_uint16(layer: np.ndarray, dense_is_dark: bool = True) -> np.ndarray:
    lo, hi = np.percentile(layer, [0.5, 99.5])
    x = np.clip((layer - lo) / max(float(hi - lo), 1e-9), 0.0, 1.0)
    if dense_is_dark:
        x = 1.0 - x
    return np.rint(x * 65535.0).astype(np.uint16)


def _write_png(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write image: {path}")


def save_layers(
    layers: tuple[np.ndarray, np.ndarray],
    out_dir: Path,
    dense_is_dark: bool = True,
) -> dict[str, Path]:
    """Пишет layer_1.png, layer_2.png (8 бит) и *_16bit.png (16 бит)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, Path] = {}
    for number, layer in enumerate(layers, start=1):
        name = f"layer_{number}"
        files[name] = out_dir / f"{name}.png"
        files[f"{name}_16bit"] = out_dir / f"{name}_16bit.png"
        _write_png(files[name], layer_to_uint8(layer, dense_is_dark))
        _write_png(files[f"{name}_16bit"], layer_to_uint16(layer, dense_is_dark))
    return files


def save_shifts_csv(
    shifts: np.ndarray,
    strength: np.ndarray,
    reliable: np.ndarray,
    path: Path,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["frame", "bottom_dx", "bottom_dy", "top_dx", "top_dy",
             "bottom_strength", "top_strength", "reliable"]
        )
        for k in range(len(shifts)):
            writer.writerow(
                [k,
                 f"{shifts[k, 0, 0]:.3f}", f"{shifts[k, 0, 1]:.3f}",
                 f"{shifts[k, 1, 0]:.3f}", f"{shifts[k, 1, 1]:.3f}",
                 f"{strength[k, 0]:.4f}", f"{strength[k, 1]:.4f}", int(reliable[k])]
            )
    return path


def plot_shifts(
    shifts: np.ndarray,
    strength: np.ndarray,
    reliable: np.ndarray,
    path: Path,
) -> Path | None:
    """График смещений: dx, dy по кадрам и траектория верхнего слоя. None без matplotlib."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib is not installed; skipping shifts plot")
        return None

    n = len(shifts)
    frames = np.arange(n)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    for layer, label in ((0, "нижний"), (1, "верхний")):
        ax.plot(frames, shifts[:, layer, 0], marker="o", ms=3, label=f"{label}: dx")
        ax.plot(frames, shifts[:, layer, 1], marker="s", ms=3, ls="--", label=f"{label}: dy")
    ax.scatter(frames[~reliable], np.zeros((~reliable).sum()), c="red", marker="x",
               label="ненадёжные кадры", zorder=5)
    ax.set_xlabel("кадр")
    ax.set_ylabel("смещение, px")
    ax.set_title("Смещения относительно кадра 0")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(shifts[:, 1, 0], shifts[:, 1, 1], "-o", ms=3, label="верхний слой")
    ax.plot(shifts[:, 0, 0], shifts[:, 0, 1], "-o", ms=3, label="нижний слой")
    bad = ~reliable
    ax.scatter(shifts[bad, 1, 0], shifts[bad, 1, 1], c="red", marker="x", zorder=5)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("dx, px")
    ax.set_ylabel("dy, px")
    ax.set_title("Траектория")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.bar(frames - 0.2, strength[:, 0], width=0.4, label="нижний")
    ax.bar(frames + 0.2, strength[:, 1], width=0.4, label="верхний")
    ax.set_xlabel("кадр")
    ax.set_ylabel("высота пика корреляции")
    ax.set_title("Надёжность оценки")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def save_report(report: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
