# Module 1 — Input loading and preprocessing

This package implements the agreed interface between Module 1 and the next
modules in the X-Ray PCB Layer Separator project.

## Data contract

`load_gif(...)`, `load_png_sequence(...)`, and `load_sequence(...)` return a
non-empty `list[numpy.ndarray]`. Each array is grayscale (`(H, W)`), has dtype
`numpy.float32`, contains only finite values in `[0.0, 1.0]`, and has the same
shape as every other frame.

**Use 16-bit PNG sequences when the original projections are available.** GIF
is palette-based and loses the original 16-bit intensity precision. The loader
scales 8-bit images by 255 and 16-bit images by 65535. It deliberately does not
min-max-normalize each image on load, because doing so would destroy relative
brightness between projections.

## Install

Python 3.10 or newer is required.

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

## Use from code

```python
from pcblayers import load_sequence, validate_frames

frames = load_sequence("output/synthetic.gif")  # or a directory of 16-bit PNGs
stats = validate_frames(frames)
print(stats)
```

Public API:

- `load_gif(path)` — load all GIF frames.
- `load_png_sequence(directory, pattern="*.png")` — load image files in natural filename order (`frame_2.png` before `frame_10.png`).
- `load_sequence(path, pattern="*.png")` — load a GIF, a single PNG image (one frame), or a directory of PNG frames. A stitched PNG is useful for inspection; layer separation normally needs the full tomosynthesis sequence.
- `validate_frames(frames)` — check every frame against the data contract and return statistics.
- `normalize(frames, mode=...)` — optional explicit normalization. Default is `"none"`. Use `"per_frame"` only if independently rescaling frames is intended.
- `save_frames(frames, out_dir)` — write PNGs for debugging. Both export functions validate by default; pass `validate=False` only when the same frames have already been validated in your pipeline.
- `save_preview_gif(frames, out_path, fps=10)` — write a lossy visualization GIF. GIF timing has 10 ms precision, so the resulting playback rate can differ slightly from the requested FPS.

## CLI

Generate reproducible synthetic data:

```bash
python -m pcblayers --make-synthetic output/synthetic.gif --synthetic-frames 20 --size 128 --fps 10 --seed 0
```

Load input and save debug PNGs:

```bash
python -m pcblayers output/synthetic.gif --out output/debug --preview-gif --fps 10 -v
```

Load a directory of PNGs (optionally change the filename pattern). For a tomosynthesis dataset, point to the folder containing the numbered projection PNGs, not its parent folder if that parent also contains stitched overview images:

```bash
python -m pcblayers data/projections --pattern "projection_*.png" --out output/debug
```

Run the tests:

```bash
pytest -q
```

Debug files are derived outputs; avoid committing large real datasets or generated
`output/` folders unless the team explicitly wants them version-controlled.


## Tested real-data layout

For the provided investigation archive, load the numbered projection frames from the
`Original/Tomosynthesis .../` directory. It contains 36 grayscale 16-bit PNGs at
2280 × 1812 pixels. The two `Stitch ...png` files are standalone stitched images at
different resolutions; use those individually for inspection, not as one sequence.
