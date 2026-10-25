import numpy as np

from pcblayers.trajectory import circle_inliers, fit_circle


def _circle(n=36, r=20.0, cx=-20.0, cy=0.0):
    theta = np.radians(np.arange(n) * 10.0)
    return np.column_stack([cx + r * np.cos(theta), cy + r * np.sin(theta)])


def test_fit_circle_recovers_parameters():
    cx, cy, r = fit_circle(_circle())
    assert abs(cx + 20.0) < 1e-6 and abs(cy) < 1e-6 and abs(r - 20.0) < 1e-6


def test_outliers_are_rejected():
    points = _circle()
    points[[4, 9, 20]] += [[25.0, -30.0], [-40.0, 10.0], [15.0, 35.0]]
    mask, info = circle_inliers(points, tolerance_px=2.0)
    assert not info["static"]
    assert not mask[[4, 9, 20]].any()
    assert mask.sum() >= len(points) - 3
    assert abs(info["r"] - 20.0) < 1.0


def test_static_layer_is_all_inliers():
    points = np.zeros((20, 2))
    points[7] = (1.0, -1.0)
    mask, info = circle_inliers(points)
    assert info["static"] and mask.all()


def test_small_inputs_do_not_crash():
    mask, _ = circle_inliers(np.array([[0.0, 0.0], [10.0, 5.0], [-8.0, 3.0]]))
    assert mask.shape == (3,)
    empty, _ = circle_inliers(np.zeros((0, 2)))
    assert empty.shape == (0,)
