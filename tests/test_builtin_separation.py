import numpy as np
import pytest

from conftest import pearson
from pcblayers.builtin_separation import separate_layers


def test_separation_with_true_shifts(scene):
    bottom, top = separate_layers(scene.frames, scene.shifts)
    assert bottom.shape == top.shape == scene.frames[0].shape
    assert bottom.dtype == top.dtype == np.float32
    assert pearson(bottom, scene.bottom) > 0.95
    assert pearson(top, scene.top) > 0.95
    # Слои не должны «просачиваться» друг в друга.
    assert abs(pearson(bottom, scene.top)) < 0.3
    assert abs(pearson(top, scene.bottom)) < 0.3


def test_unreliable_frames_are_ignored(scene):
    shifts = scene.shifts.copy()
    reliable = np.ones(len(scene.frames), dtype=bool)
    shifts[5:8] += 40.0  # мусорные смещения...
    reliable[5:8] = False  # ...которые помечены как ненадёжные
    bottom, top = separate_layers(scene.frames, shifts, reliable)
    assert pearson(top, scene.top) > 0.9


def test_input_validation(scene):
    n = len(scene.frames)
    with pytest.raises(ValueError, match="shifts must have shape"):
        separate_layers(scene.frames, np.zeros((n, 2)))
    with pytest.raises(ValueError, match="reliable must have shape"):
        separate_layers(scene.frames, scene.shifts, np.ones(n + 1, dtype=bool))
    few = np.zeros(n, dtype=bool)
    few[:2] = True
    with pytest.raises(ValueError, match="Too few reliable frames"):
        separate_layers(scene.frames, scene.shifts, few)
    with pytest.raises(ValueError, match="iterations"):
        separate_layers(scene.frames, scene.shifts, iterations=0)
