"""Тесты модуля 2 на синтетике с известным ответом.

Запуск: pytest tests/test_motion_analysis.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pcblayers.motion_analysis import estimate_shifts, make_synthetic_frames  # noqa: E402


def _errors(res, true_rel):
    ok = res.reliable.copy()
    ok[res.ref_index] = False
    return np.hypot(*(res.relative[ok] - true_rel[ok]).T)


def test_clean_two_layers():
    frames, true_rel = make_synthetic_frames(amplitude=15.0)
    res = estimate_shifts(frames)
    assert res.shifts.shape == (36, 2, 2)
    assert res.relative.shape == (36, 2)
    assert res.reliable.sum() >= 30
    assert _errors(res, true_rel).max() < 0.5       # проверяет и знак смещения
    assert res.static_std < 0.3                      # нижний слой стабилен


def test_noise_and_gif_quantization():
    frames, true_rel = make_synthetic_frames(amplitude=15.0, noise=0.03, levels=64, seed=1)
    res = estimate_shifts(frames)
    assert res.reliable.sum() >= 28
    assert _errors(res, true_rel).max() < 1.0


def test_relative_filled_by_model_in_unreliable_frames():
    frames, true_rel = make_synthetic_frames(amplitude=15.0)
    res = estimate_shifts(frames)
    assert np.isfinite(res.relative).all()           # нет NaN, даже где кадр ненадёжен
    assert np.hypot(*(res.relative - true_rel).T).max() < 1.0


def test_corrupted_frame_is_flagged():
    frames, true_rel = make_synthetic_frames(amplitude=15.0)
    frames[10] = np.random.default_rng(5).random(frames[10].shape).astype(np.float32)
    res = estimate_shifts(frames)
    assert not res.reliable[10]
    others = res.reliable.copy()
    others[[0, 10]] = False
    assert np.hypot(*(res.relative[others] - true_rel[others]).T).max() < 0.5


def test_small_motion_layers_merge_but_no_crash():
    frames, _ = make_synthetic_frames(amplitude=2.0)
    res = estimate_shifts(frames)                    # слои неразличимы: должно отработать без ошибок
    assert res.shifts.shape == (36, 2, 2)
    assert res.reliable[0]


def test_flat_frames_do_not_crash():
    frames = [np.full((100, 120), 0.5, np.float32) for _ in range(8)]
    with np.errstate(all="ignore"):
        try:
            res = estimate_shifts(frames)
        except Exception as e:                       # noqa: BLE001
            raise AssertionError(f"плоские кадры вызвали исключение: {e!r}")
    assert res.reliable[0] and not res.reliable[1:].any()


def test_input_validation():
    good, _ = make_synthetic_frames(n=6, shape=(64, 64))
    for bad in (
        good[:2],                                                    # мало кадров
        [good[0][:, :-1]] + good[1:],                                # разный размер
        [good[0][..., None]] + good[1:],                             # не 2D
        [np.where(good[0] > .5, np.nan, good[0])] + good[1:],        # NaN
    ):
        try:
            estimate_shifts(bad)
        except ValueError:
            continue
        raise AssertionError("ожидался ValueError")
    try:
        estimate_shifts(good, ref_index=99)
    except ValueError:
        return
    raise AssertionError("ожидался ValueError для ref_index")


if __name__ == "__main__":                           # запуск без pytest
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("OK  ", name)