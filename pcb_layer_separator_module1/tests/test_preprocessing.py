from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pcblayers.preprocessing import (
    normalize,
    save_frames,
    save_preview_gif,
    to_uint8,
    validate_frames,
)


def test_validate_returns_stats() -> None:
    frames = [np.zeros((8, 8), dtype=np.float32), np.ones((8, 8), dtype=np.float32)]
    stats = validate_frames(frames)
    assert stats["n_frames"] == 2
    assert stats["shape"] == (8, 8)
    assert stats["dtype"] == "float32"
    assert stats["global_min"] == 0.0
    assert stats["global_max"] == 1.0


def test_validate_rejects_empty_frames() -> None:
    with pytest.raises(ValueError, match="empty"):
        validate_frames([])


def test_validate_rejects_bad_dtype() -> None:
    with pytest.raises(TypeError, match="float32"):
        validate_frames([np.zeros((8, 8), dtype=np.uint8)])


def test_validate_checks_every_frame_dtype_and_dimensions() -> None:
    with pytest.raises(TypeError, match="Frame 1"):
        validate_frames([
            np.zeros((8, 8), dtype=np.float32),
            np.zeros((8, 8), dtype=np.float64),
        ])
    with pytest.raises(ValueError, match="Frame 1"):
        validate_frames([
            np.zeros((8, 8), dtype=np.float32),
            np.zeros((8, 8, 1), dtype=np.float32),
        ])


def test_validate_rejects_inconsistent_shapes() -> None:
    with pytest.raises(ValueError, match="Inconsistent frame shapes"):
        validate_frames([
            np.zeros((8, 8), dtype=np.float32),
            np.zeros((9, 9), dtype=np.float32),
        ])


def test_validate_rejects_non_finite_and_out_of_range_values() -> None:
    with pytest.raises(ValueError, match="NaN or infinite"):
        validate_frames([np.array([[np.nan]], dtype=np.float32)])
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        validate_frames([np.array([[-0.1]], dtype=np.float32)])


def test_normalize_global_preserves_relative_scale() -> None:
    a = np.full((4, 4), 0.2, dtype=np.float32)
    b = np.full((4, 4), 0.8, dtype=np.float32)
    out = normalize([a, b], mode="global")
    assert float(out[0].mean()) == pytest.approx(0.0)
    assert float(out[1].mean()) == pytest.approx(1.0)


def test_normalize_none_preserves_values() -> None:
    frame = np.full((4, 4), 0.3, dtype=np.float32)
    result = normalize([frame], mode="none")
    np.testing.assert_array_equal(result[0], frame)


def test_normalize_constant_images_returns_zeros() -> None:
    frame = np.full((4, 4), 0.3, dtype=np.float32)
    out = normalize([frame], mode="global")
    assert np.all(out[0] == 0.0)


def test_normalize_rejects_unknown_mode_even_for_empty_input() -> None:
    with pytest.raises(ValueError, match="Unknown normalization mode"):
        normalize([], mode="mystery")


def test_to_uint8_clips_and_rounds() -> None:
    frame = np.array([[-0.5, 0.5, 1.5]], dtype=np.float32)
    assert to_uint8(frame).tolist() == [[0, 128, 255]]


def test_to_uint8_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="NaN or infinite"):
        to_uint8(np.array([[np.inf]], dtype=np.float32))


def test_save_frames_and_preview_gif(tmp_path: Path) -> None:
    frames = [np.zeros((8, 8), dtype=np.float32), np.ones((8, 8), dtype=np.float32)]
    paths = save_frames(frames, tmp_path / "pngs")
    assert len(paths) == 2
    assert all(path.is_file() for path in paths)
    with Image.open(paths[1]) as image:
        assert image.mode == "L"
        assert image.getpixel((0, 0)) == 255

    preview = save_preview_gif(frames, tmp_path / "preview.gif", fps=5)
    assert preview.is_file()
    with Image.open(preview) as image:
        assert image.format == "GIF"
        assert image.n_frames == 2


def test_save_preview_rejects_invalid_fps(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="fps must be"):
        save_preview_gif([np.zeros((4, 4), dtype=np.float32)], tmp_path / "x.gif", fps=0)


def test_save_preview_rejects_unrepresentable_fps(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="timing resolution"):
        save_preview_gif([np.zeros((4, 4), dtype=np.float32)], tmp_path / "x.gif", fps=120)
