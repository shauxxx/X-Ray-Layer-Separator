# PCB Layer Separation

A part of the PCB reverse engineering project: an application that splits
X-ray tomosynthesis (DTS) frames of a two-sided printed circuit board into two
layers.

```
GIF / PNG frames -> shifts between frames -> two layers -> files + comparison
   (module 1)          (module 2)             (module 3)       (module 4)
```

## Motivation

When a device can't be examined destructively, X-ray and CT scans are often
the only way to look inside a printed circuit board (PCB). Separating the scan
into per-layer images is the first step towards recovering connections between
components and finding hidden connection points.

## How it works

The bottom layer stays in place while the top layer moves from frame to frame
(parallax of the rotating X-ray source). The program measures that motion and
uses it to pull the two layers apart.

## Install

Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
pip install -e .
```

## Run

One command, no data of your own needed (generates a synthetic two-layer GIF):

```bash
pcblayers --demo -o output
```

On your data:

```bash
pcblayers projections.gif -o output           # animated GIF
pcblayers path/to/projections -o output       # folder with numbered PNGs (16-bit is best)
python -m pcblayers.main projections.gif      # same, without installing the script
```

Useful options (`pcblayers --help` lists all):

| Option | Meaning |
|---|---|
| `--max-size 1024` | frame size used for layer separation (longest side, px; `0` = full size) |
| `--motion-max-size 0` | frame size used for motion search (`0` = full size, recommended) |
| `--log s.log` | tomograph log (needed by the module 3 algorithm); by default the only `*.log` in the projections folder is used |
| `--backend auto\|builtin\|module3` | separation algorithm, see below |
| `--pattern "*.png"` | file mask when the input is a folder |
| `--min-peak`, `--rel-peak`, `--trajectory-tol` | what counts as an unreliable frame |
| `--bright-metal` | dense material is bright instead of dark in the output PNGs |
| `--save-frames` | also write the frames the pipeline sees to `OUT/debug_frames` |
| `--no-plots` | skip the shifts plot |

**Which algorithm runs.** `auto` (default) uses the module 3 algorithm when
`layer_separation.py` exists **and** a tomograph log is available, otherwise the
built-in fallback. The module 3 algorithm takes its geometry (rotation step,
angle, ΔZ, pixel size) from the log, so without one it would silently use
defaults for a different machine. Force a choice with `--backend builtin` or
`--backend module3`.

```bash
pcblayers path/to/projections --log path/to/s.log -o output   # module 3 algorithm
pcblayers path/to/projections -o output --backend builtin     # no log needed
```

> Motion search is slow on full-size frames (about 40 s for 36 frames of
> 2280 x 1812) but is what makes the layers separable: on downscaled frames the
> two correlation peaks merge. Use `--motion-max-size` only to trade accuracy for speed.

## Output

Everything is written to the output folder (default `output/`):

| File | Content |
|---|---|
| `layer_1.png`, `layer_2.png` | layer 1 = static component (bottom layer), layer 2 = moving component (top layer), 8-bit |
| `layer_1_16bit.png`, `layer_2_16bit.png` | the same in 16-bit |
| `comparison.png` | source frame 0 \| mean of all frames \| layer 1 \| layer 2 |
| `shifts.csv`, `shifts.png` | measured shifts per frame, reliability of each estimate, plot |
| `report.json` | backend, reliable/unreliable frames, bottom-layer stability, metrics, timings |

The console prints the edge correlation between the layers: the closer to 0,
the less one layer leaks into the other.

Which layer is physically "top" cannot be told from frames alone; if your layers
look swapped, swap the files.

## Input notes

* **GIF is lossy**: it holds at most 256 levels per frame, while the original
  projections are 16-bit. Use a folder of 16-bit PNGs when you have them.
* Needs the whole sequence (at least 4 frames). A single stitched image cannot be separated.
* Frames whose shift estimate is unreliable (weak correlation peak or off the
  circular trajectory) are skipped and listed in `report.json`.

## Module layout

| File | Owner | Role |
|---|---|---|
| `gif_loader.py`, `preprocessing.py`, `synthetic.py` | participant 1 | load GIF/PNG frames, validate, debug export |
| `motion_analysis.py` | participant 2 | `estimate_shifts(frames)` -> shifts `(N, 2, 2)`, strength `(N, 2)` |
| `reconstruction.py` | participant 3 | the separation script (copy of `new_claudecode.py`, unchanged): log parsing, geometry calibration, exact separation in the frequency domain |
| `layer_separation.py` | participant 4 (thin wrapper) | exposes `reconstruction.py` as `separate_layers(...)`; replace it if participant 3 writes their own |
| `main.py`, `pipeline.py`, `compare.py`, `outputs.py`, `trajectory.py`, `builtin_separation.py`, `demo_data.py`, `tests/` | participant 4 | pipeline, CLI, checks, tests, fallback separator, synthetic data |

`python -m pcblayers` still starts participant 1's loader/preview tool.

### Contract for module 3

`main.py` looks for `pcblayers/layer_separation.py` and calls

```python
def separate_layers(frames, shifts=None, reliable=None, strength=None,
                    log_path=None, out_dir=None, scale=1.0):
    """
    frames:   list[np.ndarray]  float32, shape (H, W), values in [0, 1]
    shifts:   np.ndarray (N, 2, 2)  [frame][0=bottom, 1=top][dx, dy] relative to frame 0,
              in pixels of `frames` (frame 0 = (0, 0))
    reliable: np.ndarray (N,) bool  frames the shift estimate can be trusted for
    strength: np.ndarray (N, 2)     correlation peak heights from module 2
    log_path: Path | None           tomograph log (see --log)
    out_dir:  Path                  output folder, for debug files
    scale:    float                 frames were downscaled by this factor relative
                                    to the full-size images the log describes

    returns (layer_1, layer_2): two float32 arrays of shape (H, W), finite,
    "larger value = denser material" (absorption, -ln I), layer_1 = bottom/static,
    layer_2 = top/moving.
    """
```

Only the arguments named in your signature are passed (or all of them with
`**kwargs`). The program writes the PNG files itself. Until the file exists,
`--backend auto` uses the built-in fallback in `builtin_separation.py`.

The algorithm in `reconstruction.py` cannot tell which of the two layers is
physically the top one (it depends on the detector orientation). If they come out
swapped, set `TOP_IS_LARGER_R = False` in `reconstruction.py` or swap the files.

## Tests

```bash
pytest -q
```

Tests run on a synthetic two-layer scene with a known answer
(`pcblayers/demo_data.py`) and cover the pipeline, reliability rules,
trajectory check, comparison metrics and the command line. The module 3 wrapper
is only checked against the contract (shapes, types, files), not for separation
quality: judge that on real projections with `comparison.png`.

## License

MIT
