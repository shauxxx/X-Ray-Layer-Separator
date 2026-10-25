"""Общие фикстуры: синтетическая двухслойная сцена с известным ответом."""

from __future__ import annotations

import numpy as np
import pytest

from pcblayers.demo_data import make_two_layer_scene, save_scene_gif


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12))


@pytest.fixture(scope="session")
def scene():
    return make_two_layer_scene(n_frames=24, size=192, radius_px=8.0, seed=0)


@pytest.fixture(scope="session")
def scene_gif(scene, tmp_path_factory):
    path = tmp_path_factory.mktemp("gif") / "scene.gif"
    return save_scene_gif(path, scene)


@pytest.fixture(scope="session")
def scene_png_dir(scene, tmp_path_factory):
    """Папка с пронумерованными 16-битными PNG, как реальные проекции."""
    import cv2

    directory = tmp_path_factory.mktemp("png")
    for k, frame in enumerate(scene.frames, start=1):
        cv2.imwrite(str(directory / f"{k:06d}.png"), np.rint(frame * 65535).astype(np.uint16))
    return directory
