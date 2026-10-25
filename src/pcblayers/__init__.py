"""Input loading and preprocessing for X-Ray PCB Layer Separator.

Canonical contract passed to Modules 2 and 3
------------------------------------------------
A non-empty ``list[np.ndarray]`` where every frame:
* is a two-dimensional grayscale image with shape ``(height, width)``;
* has dtype ``numpy.float32``;
* contains only finite values in the closed interval [0.0, 1.0];
* has exactly the same shape as every other frame.

Intensity values are scaled according to the input pixel format, not
independently min-max-normalized per frame. This preserves relative brightness
between projections.
"""

from .gif_loader import load_gif, load_png_sequence, load_sequence
from .preprocessing import (
    normalize,
    save_frames,
    save_preview_gif,
    to_uint8,
    validate_frames,
)

__version__ = "0.2.0"

__all__ = [
    "load_gif",
    "load_png_sequence",
    "load_sequence",
    "validate_frames",
    "normalize",
    "to_uint8",
    "save_frames",
    "save_preview_gif",
]
