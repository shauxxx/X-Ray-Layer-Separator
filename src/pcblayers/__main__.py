"""Command-line tools for loading, validating, and previewing frame sequences."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .gif_loader import load_sequence
from .preprocessing import normalize, save_frames, save_preview_gif, validate_frames
from .synthetic import save_synthetic_gif


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pcblayers",
        description="Load GIF/PNG X-ray frames, validate them, and write debug previews.",
    )
    parser.add_argument("input", nargs="?", type=Path, help="GIF file, single PNG image, or image-sequence directory")
    parser.add_argument("--pattern", default="*.png", help="Glob pattern for image-sequence directories")
    parser.add_argument("--out", type=Path, default=Path("output/debug"), help="Directory for PNG debug frames")
    parser.add_argument(
        "--normalize",
        choices=["none", "global", "per_frame", "percentile"],
        default="none",
        help="Optional intensity normalization (default: none)",
    )
    parser.add_argument("--preview-gif", action="store_true", help="Also save a GIF preview next to the debug directory")
    parser.add_argument("--fps", type=float, default=10.0, help="Preview/synthetic GIF frame rate (default: 10)")
    parser.add_argument("--make-synthetic", type=Path, metavar="OUT.gif", help="Generate a synthetic GIF and exit")
    parser.add_argument("--synthetic-frames", type=int, default=20, help="Number of synthetic frames (default: 20)")
    parser.add_argument("--size", type=int, default=128, help="Width and height of synthetic frames (default: 128)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for synthetic data (default: 0)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        if args.make_synthetic:
            output = save_synthetic_gif(
                args.make_synthetic,
                n_frames=args.synthetic_frames,
                size=args.size,
                fps=args.fps,
                seed=args.seed,
            )
            print(f"Synthetic GIF written: {output}")
            return 0

        if args.input is None:
            parser.error("input is required unless --make-synthetic is specified")

        frames = load_sequence(args.input, pattern=args.pattern)
        # The loader already guarantees the contract. Skip a redundant validation
        # for the default path; optional normalizers validate their input.
        if args.normalize != "none":
            frames = normalize(frames, mode=args.normalize)
        stats = validate_frames(frames)
        saved_paths = save_frames(frames, args.out, validate=False)
        if args.preview_gif:
            preview_path = args.out.parent / "preview.gif"
            save_preview_gif(frames, preview_path, fps=args.fps, validate=False)
            print(f"Preview GIF: {preview_path}")

        print(f"Stats: {stats}")
        print(f"Saved {len(saved_paths)} frames to {args.out}")
        return 0
    except (OSError, TypeError, ValueError) as exc:
        if args.verbose:
            logging.exception("Processing failed")
        else:
            logging.error("Processing failed: %s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
