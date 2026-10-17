#!/usr/bin/env python3
from __future__ import annotations

import argparse
import configparser
import math
import re
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTS = {".png", ".tif", ".tiff", ".bmp", ".jpg", ".jpeg"}

def evaluate_layer_quality(image, valid_mask=None):

    if image is None:
        raise ValueError("image не должен быть None")

    image = np.asarray(image)

    if image.ndim == 3:
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if image.ndim != 2 or min(image.shape) < 16:
        raise ValueError("Ожидается 2D-изображение размером минимум 16x16")

    image = image.astype(np.float32)

    finite = np.isfinite(image)
    if not finite.all():
        image = np.where(finite, image, 0)

    # Нормализация по процентилям: уменьшает зависимость от битности
    # и абсолютного уровня яркости.
    lo, hi = np.percentile(image, [1, 99])
    if hi <= lo:
        return {
            "quality_score": 0.0,
            "laplacian_variance": 0.0,
            "tenengrad": 0.0,
            "brenner": 0.0,
            "noise_estimate": 0.0,
            "edge_density": 0.0,
            "clipping_fraction": 1.0,
            "status": "uniform_or_invalid"
        }

    raw = image.copy()
    image = np.clip((image - lo) / (hi - lo), 0, 1).astype(np.float32)

    # Исключаем края и невалидные области из анализа.
    mask = np.ones(image.shape, dtype=bool)
    if valid_mask is not None:
        if valid_mask.shape != image.shape:
            raise ValueError("valid_mask должна совпадать с размером image")
        mask &= valid_mask.astype(bool)

    # Игнорируем узкую полосу у границы: производные там ненадёжны.
    mask[:2, :] = False
    mask[-2:, :] = False
    mask[:, :2] = False
    mask[:, -2:] = False

    if mask.sum() < 100:
        raise ValueError("Недостаточно валидных пикселей для оценки")

    # 1. Laplacian variance: чувствительность к мелким деталям.
    lap = cv2.Laplacian(image, cv2.CV_32F, ksize=3)
    lap_var = float(np.var(lap[mask]))

    # 2. Tenengrad: энергия градиента Sobel.
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    grad_sq = gx * gx + gy * gy
    tenengrad = float(np.mean(grad_sq[mask]))

    # 3. Brenner gradient: перепады яркости через два пикселя.
    diff2 = image[:, 2:] - image[:, :-2]
    mask2 = mask[:, 2:] & mask[:, :-2]
    brenner = float(np.mean(diff2[mask2] ** 2))

    # 4. Оценка шума по высокочастотному остатку в плоских областях.
    # Резкие границы исключаем, чтобы не считать дорожки шумом.
    smooth = cv2.GaussianBlur(image, (0, 0), 1.2)
    residual = image - smooth
    grad_mag = np.sqrt(grad_sq)
    flat = mask & (grad_mag < np.percentile(grad_mag[mask], 50))

    if flat.sum() >= 100:
        r = residual[flat]
        med = np.median(r)
        noise = float(1.4826 * np.median(np.abs(r - med)))
    else:
        noise = float("nan")

    # 5. Плотность выраженных границ.
    edge_map = cv2.Canny(
        np.uint8(image * 255), threshold1=40, threshold2=100
    )
    edge_density = float(np.mean(edge_map[mask] > 0))

    # 6. Доля пикселей, попавших в крайние уровни исходного диапазона.
    clipping = float(np.mean(
        (raw[mask] <= lo) | (raw[mask] >= hi)
    ))

    # Эвристическая оценка: резкость плюс умеренный штраф за шум.
    # Это НЕ обученная модель и не физическая точность реконструкции.
    sharpness = (
        0.40 * np.log1p(lap_var * 1000.0)
        + 0.40 * np.log1p(tenengrad * 100.0)
        + 0.20 * np.log1p(brenner * 100.0)
    )

    # Нормировка score — удобная шкала, но не калиброванная вероятность.
    sharpness_score = 100.0 * (1.0 - np.exp(-sharpness / 3.0))

    if np.isfinite(noise):
        noise_penalty = min(25.0, 250.0 * noise)
    else:
        noise_penalty = 0.0

    clipping_penalty = min(15.0, 100.0 * clipping)
    score = float(np.clip(
        sharpness_score - noise_penalty - clipping_penalty,
        0, 100
    ))

    return {
        "quality_score": round(score, 2),
        "laplacian_variance": round(lap_var, 7),
        "tenengrad": round(tenengrad, 7),
        "brenner": round(brenner, 7),
        "noise_estimate": (
            round(noise, 7) if np.isfinite(noise) else None
        ),
        "edge_density": round(edge_density, 6),
        "clipping_fraction": round(clipping, 6),
        "status": "ok"
    }



def read_log(path: Path) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser(interpolation=None)
    cfg.optionxform = str  # preserve key spelling
    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        cfg.read_file(f)
    return cfg


def get_float(cfg, section, key, default):
    try:
        return float(cfg[section][key].strip())
    except (KeyError, ValueError):
        return float(default)


def get_int(cfg, section, key, default):
    try:
        return int(float(cfg[section][key].strip()))
    except (KeyError, ValueError):
        return int(default)


def natural_key(path: Path):
    return [int(s) if s.isdigit() else s.lower()
            for s in re.split(r"(\d+)", path.name)]


def load_grayscale(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Не удалось прочитать изображение: {path}")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.dtype not in (np.uint8, np.uint16):
        image = np.nan_to_num(image)
        lo, hi = np.percentile(image, [0.1, 99.9])
        image = np.clip((image - lo) / max(hi - lo, 1e-9) * 65535, 0, 65535).astype(np.uint16)
    return image


def robust_normalize(image: np.ndarray) -> np.ndarray:
    """Convert to float32 [0,1] using robust percentiles."""
    a = image.astype(np.float32)
    lo, hi = np.percentile(a, [0.2, 99.8])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0)


def enhance_for_viewing(image01: np.ndarray) -> np.ndarray:
    """Local contrast enhancement for viewing; not used for reconstruction."""
    u8 = np.clip(image01 * 255.0, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    out = clahe.apply(u8)
    out = cv2.fastNlMeansDenoising(out, None, h=3, templateWindowSize=7, searchWindowSize=21)
    return out


def shift_image(image: np.ndarray, dx: float, dy: float) -> np.ndarray:
    h, w = image.shape
    matrix = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    return cv2.warpAffine(
        image, matrix, (w, h), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )


def reconstruct_plane(frames, frame_angles_deg, z_mm, pixel_mm, axis_angle_deg):
    """
    Approximate shift-and-add plane reconstruction.

    This model assumes depth parallax of approximately z*sin(view_angle)/pixel_size.
    The displacement direction rotates with the projection angle. Real systems may
    require a calibrated projection model, magnification correction, or axis mapping.
    """
    acc = np.zeros_like(frames[0], dtype=np.float32)
    weight = np.zeros_like(acc, dtype=np.float32)

    axis = math.radians(axis_angle_deg)
    for frame, angle_deg in zip(frames, frame_angles_deg):
        theta = math.radians(angle_deg) + axis

        # Parallax in output pixels. Scaled Image Pixel Size is used because the
        # frames are assumed to be represented at that effective pixel scale.
        distance_px = z_mm * math.sin(theta) / max(pixel_mm, 1e-9)
        dx = distance_px * math.cos(theta)
        dy = distance_px * math.sin(theta)

        shifted = shift_image(frame, dx, dy)
        valid = shift_image(np.ones_like(frame, dtype=np.float32), dx, dy)
        acc += shifted * valid
        weight += valid

    return acc / np.maximum(weight, 1e-6)


def main():
    ap = argparse.ArgumentParser(description="Реконструкция двух слоёв платы из рентгеновской серии")
    ap.add_argument("--input", type=Path, default="photo_for_test1", help="Папка с проекциями")
    ap.add_argument("--log", type=Path, default="photo_for_test1/s.log", help="INI-лог сканирования")
    ap.add_argument("--output",  default="reconstructed_layers", type=Path, help="Папка результатов")
    ap.add_argument("--z1", type=float, default=None, help="Глубина первого слоя в мм относительно опорной плоскости")
    ap.add_argument("--z2", type=float, default=None, help="Глубина второго слоя в мм относительно опорной плоскости")
    ap.add_argument("--max-frames", type=int, default=0, help="Для теста ограничить число кадров; 0 = все")
    args = ap.parse_args()

    if not args.input.is_dir():
        raise SystemExit(f"Папка изображений не найдена: {args.input}")
    if not args.log.is_file():
        raise SystemExit(f"Лог не найден: {args.log}")

    cfg = read_log(args.log)
    files = sorted((p for p in args.input.iterdir() if p.suffix.lower() in IMAGE_EXTS),
                   key=natural_key)
    if args.max_frames > 0:
        files = files[:args.max_frames]
    if len(files) < 2:
        raise SystemExit("Нужно минимум два изображения в папке.")

    expected = get_int(cfg, "Acquisition", "Number Of Files", len(files))
    step = get_float(cfg, "Acquisition", "Rotation Step (deg)", 10.0)
    sdd = get_float(cfg, "Acquisition", "Camera to Source (mm)", 237.6)
    sod = get_float(cfg, "Acquisition", "Object to Source (mm)", 106.041)
    scaled_um = get_float(cfg, "Acquisition", "Scaled Image Pixel Size (um)", 137.584)
    correction_angle = get_float(cfg, "GeometryCorrection", "Angle", 0.0)
    delta_z = get_float(cfg, "GeometryCorrection", "Delta Z", 1.0)
    direction = cfg.get("Acquisition", "Rotation Direction", fallback="CW").strip().upper()

    # Delta Z is used only as a convenient initial guess for two planes.
    # It is not assumed to be the actual distance between PCB copper layers.
    z1 = args.z1 if args.z1 is not None else -delta_z / 2.0
    z2 = args.z2 if args.z2 is not None else +delta_z / 2.0

    print(f"Найдено кадров: {len(files)} (в логе указано {expected})")
    print(f"Геометрия: SOD={sod:.4f} mm, SDD={sdd:.4f} mm, "
          f"масштаб={scaled_um:.4f} um/px")
    print(f"Угол коррекции={correction_angle:.3f}°, шаг={step:.3f}°, направление={direction}")
    print(f"Реконструируемые глубины: z1={z1:.4f} mm, z2={z2:.4f} mm")
    if len(files) != expected:
        print("ПРЕДУПРЕЖДЕНИЕ: количество кадров не совпадает с Number Of Files в логе.")

    raw = [load_grayscale(p) for p in files]
    shape = raw[0].shape
    if any(im.shape != shape for im in raw):
        raise SystemExit("Размеры изображений различаются. Сначала выровняйте/приведите кадры к одному размеру.")

    # Normalize each projection consistently to reduce exposure variation.
    frames = [robust_normalize(im) for im in raw]

    sign = -1.0 if direction.startswith("CCW") else 1.0
    angles = [sign * i * step for i in range(len(frames))]
    pixel_mm = scaled_um / 1000.0

    # If the scan reports SDD/SOD but the effective pixel scale is unavailable,
    # a more detailed calibrated projection model should be supplied.
    if sdd <= sod:
        print("ПРЕДУПРЕЖДЕНИЕ: SDD <= SOD; проверьте параметры геометрии.")

    args.output.mkdir(parents=True, exist_ok=True)
    for idx, (z, label) in enumerate(((z1, "layer_1"), (z2, "layer_2")), start=1):
        print(f"Реконструкция {label} ({z:+.4f} mm)...")
        recon = reconstruct_plane(frames, angles, z, pixel_mm, correction_angle)
        recon = np.clip(recon, 0.0, 1.0)

        # Save a 16-bit grayscale result and an 8-bit enhanced preview.
        out16 = np.round(recon * 65535.0).astype(np.uint16)
        out8 = enhance_for_viewing(recon)
        cv2.imwrite(str(args.output / f"{label}_raw16.png"), out16)
        cv2.imwrite(str(args.output / f"{label}_enhanced.png"), out8)

    layer1 = cv2.imread(str(args.output / "layer_1_raw16.png"), cv2.IMREAD_UNCHANGED)
    layer2 = cv2.imread(str(args.output / "layer_2_raw16.png"), cv2.IMREAD_UNCHANGED)
    if layer1 is None or layer2 is None:
        raise RuntimeError(f"Файлы не найдены в {args.output.resolve()}")

    result1 = evaluate_layer_quality(layer1)
    result2 = evaluate_layer_quality(layer2)

    print("Слой 1:", result1)
    print("Слой 2:", result2)

    print(f"Готово. Результаты сохранены в: {args.output.resolve()}")
    print("Важно: это стартовая геометрическая модель. Проверьте направление сдвигов и z1/z2 "
          "на тестовой серии; для гарантированного разделения нужен калибровочный фантом "
          "или известная геометрия слоёв.")


if __name__ == "__main__":
    main()
