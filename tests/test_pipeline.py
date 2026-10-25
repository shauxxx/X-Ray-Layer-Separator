import json

import cv2
import numpy as np
import pytest

from conftest import pearson
from pcblayers import pipeline
from pcblayers.pipeline import (
    PipelineConfig,
    PipelineError,
    bottom_layer_stability,
    reliability_mask,
    resize_frames,
    run_pipeline,
)


# ----------------------------- вспомогательное ------------------------------
def test_resize_frames_keeps_aspect_and_dtype():
    frames = [np.random.default_rng(i).random((100, 200)).astype(np.float32) for i in range(3)]
    resized, scale = resize_frames(frames, 100)
    assert scale == 0.5
    assert resized[0].shape == (50, 100) and resized[0].dtype == np.float32
    same, scale = resize_frames(frames, 0)
    assert same is frames and scale == 1.0
    same, scale = resize_frames(frames, 500)
    assert same is frames and scale == 1.0


def test_reliability_mask_thresholds():
    strength = np.array([[1.0, 1.0], [0.30, 0.20], [0.30, 0.01], [0.001, 0.20], [0.30, 0.20]])
    reliable, per_layer = reliability_mask(strength, min_peak=0.005, rel_peak=0.4)
    assert reliable.tolist() == [True, True, False, False, True]
    assert per_layer[0].all()  # опорный кадр надёжен всегда
    with pytest.raises(ValueError):
        reliability_mask(np.zeros((5, 3)))


def test_reliability_mask_rejects_off_trajectory_frames():
    theta = np.radians(np.arange(24) * 15.0)
    top = np.column_stack([8 * np.cos(theta) - 8, 8 * np.sin(theta)])
    shifts = np.zeros((24, 2, 2))
    shifts[:, 1, :] = top
    shifts[10, 1, :] += (30.0, 25.0)  # кадр с «ушедшей» оценкой
    strength = np.full((24, 2), 0.2)
    reliable, _ = reliability_mask(strength, shifts=shifts, trajectory_tol_px=2.0)
    assert not reliable[10]
    assert reliable.sum() >= 22


def test_bottom_layer_stability():
    shifts = np.zeros((6, 2, 2))
    ok = bottom_layer_stability(shifts, np.ones(6, bool), tolerance_px=3.0)
    assert ok["stable"] and ok["max_bottom_shift_px"] == 0.0
    shifts[3, 0] = (10.0, 0.0)
    bad = bottom_layer_stability(shifts, np.ones(6, bool), tolerance_px=3.0)
    assert not bad["stable"] and bad["max_bottom_shift_px"] == 10.0
    # ненадёжный кадр не должен портить проверку
    masked = np.ones(6, bool)
    masked[3] = False
    assert bottom_layer_stability(shifts, masked, 3.0)["stable"]


# --------------------------- выбор алгоритма (модуль 3) ----------------------
def test_unknown_backend_is_rejected(scene_gif, tmp_path):
    with pytest.raises(PipelineError, match="Unknown backend"):
        run_pipeline(scene_gif, tmp_path, PipelineConfig(backend="magic"))


def test_module3_backend_requires_module(scene_gif, tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "_load_module3", lambda: None)
    with pytest.raises(PipelineError, match="layer_separation.py"):
        run_pipeline(scene_gif, tmp_path, PipelineConfig(backend="module3"))


def test_auto_backend_uses_module3_when_present(scene_gif, tmp_path, monkeypatch):
    received = {}

    def fake_separate_layers(frames, shifts, reliable):  # strength не принимает - не получит
        received["n"] = len(frames)
        received["shifts_shape"] = shifts.shape
        received["reliable_sum"] = int(reliable.sum())
        h, w = frames[0].shape
        rng = np.random.default_rng(0)
        return rng.random((h, w), dtype=np.float32), rng.random((h, w), dtype=np.float32)

    monkeypatch.setattr(pipeline, "_load_module3", lambda: fake_separate_layers)
    log = tmp_path / "s.log"
    log.write_text("[Acquisition]\n", encoding="utf-8")
    config = PipelineConfig(backend="auto", make_plots=False, log_path=log)
    result = run_pipeline(scene_gif, tmp_path, config)
    assert result.backend_used == "module3"
    assert received["n"] == 24 and received["shifts_shape"] == (24, 2, 2)
    assert received["reliable_sum"] == int(result.reliable.sum())
    assert (tmp_path / "layer_1.png").is_file() and (tmp_path / "layer_2.png").is_file()


def test_module3_with_kwargs_signature_gets_all_extras(scene_gif, tmp_path, monkeypatch):
    seen = set()

    def fake(frames, **kwargs):
        seen.update(kwargs)
        h, w = frames[0].shape
        return np.zeros((h, w), np.float32), np.ones((h, w), np.float32)

    monkeypatch.setattr(pipeline, "_load_module3", lambda: fake)
    run_pipeline(scene_gif, tmp_path, PipelineConfig(backend="module3", make_plots=False))
    assert seen == {"shifts", "reliable", "strength", "log_path", "out_dir", "scale"}


def test_auto_backend_falls_back_to_builtin_without_log(scene_gif, tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(pipeline, "_load_module3", lambda: (lambda frames: called.append(1)))
    result = run_pipeline(scene_gif, tmp_path, PipelineConfig(backend="auto", make_plots=False))
    assert result.backend_used == "builtin" and not called


def test_find_log(tmp_path):
    assert pipeline.find_log(tmp_path) is None
    (tmp_path / "s.log").write_text("x", encoding="utf-8")
    assert pipeline.find_log(tmp_path) == tmp_path / "s.log"
    (tmp_path / "other.log").write_text("x", encoding="utf-8")
    assert pipeline.find_log(tmp_path) is None  # неоднозначно: нужен явный --log
    assert pipeline.find_log(tmp_path, tmp_path / "other.log") == tmp_path / "other.log"
    with pytest.raises(PipelineError, match="Log file not found"):
        pipeline.find_log(tmp_path, tmp_path / "missing.log")


@pytest.mark.parametrize(
    "bad_result, message",
    [
        ("not a tuple", "exactly two arrays"),
        ((np.zeros((3, 3)), np.zeros((3, 3))), "expected"),
        ((np.full((192, 192), np.nan), np.zeros((192, 192))), "NaN"),
    ],
)
def test_bad_module3_output_is_reported(scene_gif, tmp_path, monkeypatch, bad_result, message):
    monkeypatch.setattr(pipeline, "_load_module3", lambda: (lambda frames: bad_result))
    with pytest.raises(PipelineError, match=message):
        run_pipeline(scene_gif, tmp_path, PipelineConfig(backend="module3", make_plots=False))


# ------------------------------ сквозные тесты -------------------------------
def test_end_to_end_gif(scene, scene_gif, tmp_path):
    result = run_pipeline(scene_gif, tmp_path, PipelineConfig(backend="builtin"))
    assert result.backend_used == "builtin"
    for name in ("layer_1.png", "layer_2.png", "layer_1_16bit.png", "layer_2_16bit.png",
                 "comparison.png", "shifts.csv", "shifts.png", "report.json"):
        assert (tmp_path / name).is_file(), name

    layer_1 = cv2.imread(str(tmp_path / "layer_1.png"), cv2.IMREAD_UNCHANGED)
    layer_16 = cv2.imread(str(tmp_path / "layer_2_16bit.png"), cv2.IMREAD_UNCHANGED)
    assert layer_1.shape == scene.frames[0].shape and layer_1.dtype == np.uint8
    assert layer_16.dtype == np.uint16

    # Результат совпадает с эталоном (GIF теряет точность, поэтому порог мягче).
    assert pearson(result.layer_1, scene.bottom) > 0.9
    assert pearson(result.layer_2, scene.top) > 0.9
    assert abs(pearson(result.layer_1, scene.top)) < 0.3

    # Смещения модуля 2 на надёжных кадрах близки к истинным.
    good = result.reliable
    error = np.abs(result.shifts[good, 1, :] - scene.shifts[good, 1, :]).max()
    assert error < 1.6

    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["backend"] == "builtin" and report["frames"] == 24
    assert report["bottom_layer"]["stable"] is True
    assert report["reliable_frames"] == int(good.sum())
    assert "edge_corr_layers" in report["metrics"] and "files" in report


def test_end_to_end_png_directory_and_debug_frames(scene, scene_png_dir, tmp_path):
    config = PipelineConfig(backend="builtin", save_debug_frames=True, make_plots=False)
    result = run_pipeline(scene_png_dir, tmp_path, config)
    assert pearson(result.layer_2, scene.top) > 0.95  # 16 бит точнее GIF
    assert len(list((tmp_path / "debug_frames").glob("*.png"))) == 24
    assert not (tmp_path / "shifts.png").exists()


def test_resize_keeps_pipeline_working(scene, scene_png_dir, tmp_path):
    config = PipelineConfig(backend="builtin", max_size=96, make_plots=False)
    result = run_pipeline(scene_png_dir, tmp_path, config)
    assert result.layer_1.shape == (96, 96)
    assert result.report["scale"] == 0.5 and result.report["motion_scale"] == 1.0
    # Смещения пересчитаны к размеру разделения: верхний слой ходит по окружности r=4.
    assert 3.0 < np.abs(result.shifts[result.reliable, 1, :]).max() < 9.0
    small = cv2.resize(scene.top, (96, 96), interpolation=cv2.INTER_AREA)
    assert pearson(result.layer_2, small) > 0.9


def test_single_png_is_rejected_with_clear_message(scene, tmp_path):
    path = tmp_path / "single.png"
    cv2.imwrite(str(path), np.rint(scene.frames[0] * 255).astype(np.uint8))
    with pytest.raises(PipelineError, match="at least 4 frames"):
        run_pipeline(path, tmp_path / "out")


def test_missing_input_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_pipeline(tmp_path / "nope.gif", tmp_path / "out")


def test_unmoving_sequence_is_reported(tmp_path):
    """Все кадры одинаковые: слоёв нет, сообщение должно быть понятным."""
    from pcblayers.demo_data import make_two_layer_scene, save_scene_gif

    scene = make_two_layer_scene(n_frames=8)
    frozen = type(scene)(frames=[scene.frames[0]] * 8, bottom=scene.bottom,
                         top=scene.top, shifts=scene.shifts)
    gif = save_scene_gif(tmp_path / "frozen.gif", frozen)
    with pytest.raises((PipelineError, ValueError)):
        run_pipeline(gif, tmp_path / "out", PipelineConfig(make_plots=False))
