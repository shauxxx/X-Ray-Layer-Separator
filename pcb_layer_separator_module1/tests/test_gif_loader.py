from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pcblayers import load_gif, load_png_sequence, load_sequence, validate_frames
from pcblayers.synthetic import save_synthetic_gif


@pytest.fixture
def synth_gif(tmp_path: Path) -> Path:
    return save_synthetic_gif(
        tmp_path / "test.gif", n_frames=8, size=64, fps=8, seed=42
    )


def assert_contract(frames: list[np.ndarray]) -> None:
    assert frames
    assert all(frame.dtype == np.float32 for frame in frames)
    assert all(frame.ndim == 2 for frame in frames)
    assert len({frame.shape for frame in frames}) == 1
    assert all(np.isfinite(frame).all() for frame in frames)
    assert min(float(frame.min()) for frame in frames) >= 0.0
    assert max(float(frame.max()) for frame in frames) <= 1.0


def test_load_gif_contract(synth_gif: Path) -> None:
    frames = load_gif(synth_gif)
    assert len(frames) == 8
    assert_contract(frames)
    validate_frames(frames)


def test_load_missing_gif(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_gif(tmp_path / "nope.gif")


def test_load_gif_rejects_non_gif(tmp_path: Path) -> None:
    path = tmp_path / "frame.png"
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(path)
    with pytest.raises(ValueError, match="Expected a GIF"):
        load_gif(path)


def test_load_png_sequence_16bit(tmp_path: Path) -> None:
    # Values are deliberately chosen to check scaling against 65535.
    values = [0, 32768, 65535]
    for index, value in enumerate(values):
        arr = np.full((8, 8), value, dtype=np.uint16)
        Image.fromarray(arr).save(tmp_path / f"frame_{index:03d}.png")

    frames = load_png_sequence(tmp_path)
    assert_contract(frames)
    assert float(frames[0].mean()) == pytest.approx(0.0)
    assert float(frames[1].mean()) == pytest.approx(32768 / 65535, abs=1e-6)
    assert float(frames[2].mean()) == pytest.approx(1.0)


def test_png_sequence_uses_natural_sort(tmp_path: Path) -> None:
    for name, value in [("frame_10.png", 10), ("frame_2.png", 2), ("frame_1.png", 1)]:
        Image.fromarray(np.full((4, 4), value, dtype=np.uint8)).save(tmp_path / name)
    frames = load_png_sequence(tmp_path)
    assert [int(round(float(frame.mean() * 255))) for frame in frames] == [1, 2, 10]


def test_png_sequence_rejects_inconsistent_shapes(tmp_path: Path) -> None:
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(tmp_path / "frame_1.png")
    Image.fromarray(np.zeros((9, 8), dtype=np.uint8)).save(tmp_path / "frame_2.png")
    with pytest.raises(ValueError, match="Inconsistent image sizes"):
        load_png_sequence(tmp_path)


def test_png_sequence_rejects_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No files matching"):
        load_png_sequence(tmp_path)


def test_load_sequence_dispatches_gif_png_or_directory(synth_gif: Path, tmp_path: Path) -> None:
    assert len(load_sequence(synth_gif)) == 8

    # Standalone 16-bit PNGs (such as stitched overview images) preserve scale.
    single_png = tmp_path / "stitched.png"
    Image.fromarray(np.full((4, 5), 32768, dtype=np.uint16)).save(single_png)
    single_frames = load_sequence(single_png)
    assert len(single_frames) == 1
    assert single_frames[0].shape == (4, 5)
    assert single_frames[0].dtype == np.float32
    assert float(single_frames[0].mean()) == pytest.approx(32768 / 65535, abs=1e-6)

    sequence_dir = tmp_path / "sequence"
    sequence_dir.mkdir()
    Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(sequence_dir / "frame_1.png")
    Image.fromarray(np.ones((4, 4), dtype=np.uint8)).save(sequence_dir / "frame_2.png")
    assert len(load_sequence(sequence_dir)) == 2


def test_float_png_mode_outside_range_is_rejected(tmp_path: Path) -> None:
    image = Image.fromarray(np.array([[0.0, 2.0]], dtype=np.float32))
    image.save(tmp_path / "float.tif")
    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        load_png_sequence(tmp_path, pattern="*.tif")


def test_natural_sort_key_handles_mixed_filename_prefixes(tmp_path: Path) -> None:
    for name, value in [("2.png", 2), ("a.png", 1)]:
        Image.fromarray(np.full((4, 4), value, dtype=np.uint8)).save(tmp_path / name)
    assert len(load_png_sequence(tmp_path)) == 2
