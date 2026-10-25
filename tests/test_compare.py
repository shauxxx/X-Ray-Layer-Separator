import numpy as np

from pcblayers.compare import compare_with_frames, edge_correlation, make_montage, sharpness


def test_edge_correlation_identical_and_independent():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(96, 96)).astype(np.float32)
    b = rng.normal(size=(96, 96)).astype(np.float32)
    assert edge_correlation(a, a) > 0.99
    assert abs(edge_correlation(a, b)) < 0.2


def test_sharpness_prefers_sharp_image():
    sharp = np.zeros((64, 64), np.float32)
    sharp[:, 32:] = 1.0
    blurry = np.tile(np.linspace(0, 1, 64, dtype=np.float32), (64, 1))
    assert sharpness(sharp) > sharpness(blurry)


def test_compare_with_frames_reports_good_separation(scene):
    from pcblayers.builtin_separation import separate_layers

    reliable = np.ones(len(scene.frames), dtype=bool)
    layers = separate_layers(scene.frames, scene.shifts, reliable)
    metrics = compare_with_frames(scene.frames, layers, scene.shifts, reliable)
    for key in ("edge_corr_layers", "sharpness_layer_1", "sharpness_layer_2",
                "corr_with_aligned_mean_layer_1", "corr_with_aligned_mean_layer_2"):
        assert key in metrics and np.isfinite(metrics[key])
    assert metrics["n_reliable_frames"] == len(scene.frames)
    assert abs(metrics["edge_corr_layers"]) < 0.2
    # Каждый слой похож на среднее кадров, выровненных по его собственному движению.
    assert metrics["corr_with_aligned_mean_layer_1"] > 0.5
    assert metrics["corr_with_aligned_mean_layer_2"] > 0.5

    # Плохое «разделение» (два одинаковых слоя) должно выглядеть хуже.
    bad = compare_with_frames(scene.frames, (layers[0], layers[0]), scene.shifts, reliable)
    assert bad["edge_corr_layers"] > 0.9


def test_montage_shape(scene):
    from pcblayers.builtin_separation import separate_layers

    layers = separate_layers(scene.frames, scene.shifts)
    montage = make_montage(scene.frames, layers)
    h, w = scene.frames[0].shape
    assert montage.dtype == np.uint8
    assert montage.shape == (h, 4 * w + 3 * 4)
