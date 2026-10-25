"""Запуск всего конвейера одной командой.

    python -m pcblayers.main input.gif -o output
    python -m pcblayers.main path/to/projections_dir -o output
    python -m pcblayers.main --demo                     # без своих данных
    pcblayers input.gif                                 # после pip install -e .
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .demo_data import make_demo_gif
from .pipeline import BACKENDS, PipelineConfig, PipelineError, run_pipeline

EXIT_OK = 0
EXIT_ERROR = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pcblayers",
        description=(
            "Разделение рентген-/томографических кадров печатной платы на два слоя: "
            "layer_1.png (неподвижный, нижний) и layer_2.png (движущийся, верхний)."
        ),
    )
    parser.add_argument("input", nargs="?", type=Path,
                        help="GIF с кадрами, одиночный PNG или папка с PNG-проекциями")
    parser.add_argument("-o", "--out", type=Path, default=Path("output"),
                        help="папка для результатов (по умолчанию: output)")
    parser.add_argument("--pattern", default="*.png",
                        help="маска файлов, если вход - папка (по умолчанию: *.png)")
    parser.add_argument("--max-size", type=int, default=1024, metavar="PX",
                        help="размер кадров для разделения слоёв (длинная сторона); "
                             "0 = не уменьшать (по умолчанию: 1024)")
    parser.add_argument("--motion-max-size", type=int, default=0, metavar="PX",
                        help="размер кадров для поиска смещений; 0 = полный размер "
                             "(рекомендуется: на уменьшенных кадрах пики слоёв сливаются)")
    parser.add_argument("--backend", choices=BACKENDS, default="auto",
                        help="алгоритм разделения: auto (модуль 3, если есть, иначе "
                             "встроенный), builtin, module3")
    parser.add_argument("--log", type=Path, metavar="FILE",
                        help="лог томографа (нужен алгоритму модуля 3); по умолчанию ищется "
                             "единственный *.log в папке с проекциями")
    parser.add_argument("--normalize", choices=["none", "global", "per_frame", "percentile"],
                        default="none", help="нормализация яркости кадров (по умолчанию: none)")
    parser.add_argument("--min-peak", type=float, default=0.005,
                        help="абсолютный порог надёжности оценки смещения (по умолчанию: 0.005)")
    parser.add_argument("--trajectory-tol", type=float, default=3.0, metavar="PX",
                        help="допуск отклонения кадра от окружности траектории (по умолчанию: 3)")
    parser.add_argument("--rel-peak", type=float, default=0.4,
                        help="порог относительно медианы пиков слоя (по умолчанию: 0.4)")
    parser.add_argument("--iterations", type=int, default=4,
                        help="итераций встроенного алгоритма (по умолчанию: 4)")
    parser.add_argument("--bright-metal", action="store_true",
                        help="плотный материал светлый на выходных PNG (по умолчанию тёмный)")
    parser.add_argument("--save-frames", action="store_true",
                        help="сохранить кадры, которые видит конвейер, в OUT/debug_frames")
    parser.add_argument("--no-plots", action="store_true", help="не строить график смещений")
    parser.add_argument("--demo", action="store_true",
                        help="сгенерировать синтетический GIF и прогнать на нём конвейер")
    parser.add_argument("--make-demo", type=Path, metavar="OUT.gif",
                        help="только сгенерировать синтетический GIF и выйти")
    parser.add_argument("--demo-frames", type=int, default=24, help="кадров в демо (24)")
    parser.add_argument("--demo-size", type=int, default=192, help="размер демо-кадра, px (192)")
    parser.add_argument("--seed", type=int, default=0, help="seed демо-данных (0)")
    parser.add_argument("-v", "--verbose", action="store_true", help="подробный лог")
    return parser


def _config_from_args(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        pattern=args.pattern,
        normalize=args.normalize,
        max_size=args.max_size,
        motion_max_size=args.motion_max_size,
        backend=args.backend,
        min_peak=args.min_peak,
        trajectory_tol_px=args.trajectory_tol,
        rel_peak=args.rel_peak,
        iterations=args.iterations,
        dense_is_dark=not args.bright_metal,
        save_debug_frames=args.save_frames,
        make_plots=not args.no_plots,
        log_path=args.log,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        if args.make_demo:
            path = make_demo_gif(args.make_demo, n_frames=args.demo_frames,
                                 size=args.demo_size, seed=args.seed)
            print(f"Демо-GIF записан: {path}")
            return EXIT_OK

        if args.demo:
            input_path = make_demo_gif(args.out / "demo.gif", n_frames=args.demo_frames,
                                       size=args.demo_size, seed=args.seed)
            print(f"Демо-GIF записан: {input_path}")
        elif args.input is None:
            parser.error("укажите вход (GIF / PNG / папку) или используйте --demo")
        else:
            input_path = args.input

        result = run_pipeline(input_path, args.out, _config_from_args(args))
    except (OSError, TypeError, ValueError, PipelineError) as exc:
        if args.verbose:
            logging.exception("Processing failed")
        else:
            logging.error("Ошибка: %s", exc)
        return EXIT_ERROR

    rep = result.report
    print(f"Алгоритм: {result.backend_used}; кадров: {rep['frames']} "
          f"(надёжных: {rep['reliable_frames']}); размер: {rep['processed_shape_hw']}")
    if not rep["bottom_layer"]["stable"]:
        print("Внимание: нижний слой заметно движется "
              f"(до {rep['bottom_layer']['max_bottom_shift_px']:.1f} px).")
    metrics = result.metrics
    print(f"Корреляция границ слоёв: {metrics['edge_corr_layers']:+.3f} (ближе к 0 = лучше)")
    for name in ("layer_1", "layer_2", "comparison", "shifts_plot", "shifts_csv", "report"):
        if name in result.files:
            print(f"  {name}: {result.files[name]}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
