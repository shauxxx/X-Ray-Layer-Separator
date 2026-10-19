#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Разделение двух слоёв двусторонней печатной платы по серии рентген-кадров
томосинтеза (DTS, источник движется по окружности).

Идея (геометрия + линейная алгебра, без нейросетей):

1. Закон Бера-Ламберта: после -log(I) вклад слоёв в кадр СКЛАДЫВАЕТСЯ:
       y_i(x) = A(x - d_i(rA)) + B(x - d_i(rB))
   где A, B - «чистые» изображения слоёв, а смещение слоя в i-м кадре
       d_i(r) = r * (cos θ_i, sin θ_i),   θ_i = ±i·шаг + φ0.
   r (пиксели) ~ высота слоя * tg(угол источника) / размер пикселя.
2. Параметры (r_A, r_B, φ0, направление вращения) берём из лога как
   стартовую оценку, а точно подбираем по самим кадрам: когда параметры
   верны, сложение сдвинутых кадров (shift-and-add) даёт максимально
   резкий слой.
3. Зная смещения, система уравнений на A и B решается ТОЧНО в частотной
   области (для каждой частоты - система 2x2 с регуляризацией). Это
   убирает «призрак» второго слоя, который остаётся при простом
   shift-and-add.

Запуск:  python tomo_layers.py   (аргументы не нужны, пути - в НАСТРОЙКАХ)
Зависимости: numpy, opencv-python (scipy - по желанию, ускоряет FFT).
"""
from __future__ import annotations

import configparser
import math
import re
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import scipy.fft as _fft
    _FFT_KW = {"workers": -1}
except Exception:                      # scipy не обязателен
    import numpy.fft as _fft
    _FFT_KW = {}

# ============================== НАСТРОЙКИ ==================================
INPUT_DIR = Path("photo_for_test1")            # папка с проекциями
LOG_PATH = INPUT_DIR / "s.log"                 # лог рентгена (INI)
OUTPUT_DIR = Path("reconstructed_layers")      # папка результатов (создаётся сама)

# Если автоопределение слоёв ошиблось - задайте смещения вручную
# (в пикселях полного кадра) и/или фазу. None = определять автоматически.
FORCE_R_LAYERS = None        # например (0.0, 20.7)
FORCE_PHI0_DEG = None        # например 35.0
FORCE_SENSE = None           # +1 или -1 (направление вращения); None = авто

# Какой из двух найденных слоёв назвать «верхним» (ближе к источнику).
# Из одних кадров это определить нельзя (зависит от ориентации детектора),
# поэтому если «верх» и «низ» перепутаны - поменяйте на False.
TOP_IS_LARGER_R = True

DENSE_IS_DARK = True         # True: металл тёмный (как на исходных снимках)
SAVE_SLICES = True           # дополнительно сохранить набор срезов shift-and-add
N_SLICES = 11
REGULARIZATION = 0.02        # сила регуляризации разделения (0.01..0.15)
SIGMA_BG = 60.0              # сигма удаления фона для разделения, px полного кадра
IMAGE_EXTS = {".png", ".tif", ".tiff", ".bmp", ".jpg", ".jpeg"}
# ===========================================================================


# ----------------------------- лог и файлы ---------------------------------
def read_log(path: Path) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser(interpolation=None, strict=False,
                                    allow_no_value=True)
    cfg.optionxform = str
    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        cfg.read_file(f)
    return cfg


def _get(cfg, section, key, default, cast=float):
    try:
        raw = cfg[section][key]
        return cast(float(str(raw).strip().replace(",", ".")))
    except Exception:
        return default


def parse_geometry(path: Path, n_files: int) -> dict:
    g = dict(step=10.0, sod=106.041, sdd=237.6, pix_mm=0.137584,
             angle=32.43, delta_z=4.4834, cw=True, n_expected=n_files)
    if not path.is_file():
        print(f"[!] Лог не найден ({path}) - беру значения по умолчанию.")
        return g
    cfg = read_log(path)
    g["step"] = _get(cfg, "Acquisition", "Rotation Step (deg)", g["step"])
    g["sod"] = _get(cfg, "Acquisition", "Object to Source (mm)", g["sod"])
    g["sdd"] = _get(cfg, "Acquisition", "Camera to Source (mm)", g["sdd"])
    g["pix_mm"] = _get(cfg, "Acquisition", "Scaled Image Pixel Size (um)",
                       g["pix_mm"] * 1000) / 1000.0
    g["angle"] = _get(cfg, "GeometryCorrection", "Angle", g["angle"])
    g["delta_z"] = _get(cfg, "GeometryCorrection", "Delta Z", g["delta_z"])
    g["n_expected"] = int(_get(cfg, "Acquisition", "Number Of Files", n_files))
    try:
        d = cfg["Acquisition"]["Rotation Direction"].strip().upper()
        g["cw"] = not d.startswith("CCW")
    except Exception:
        pass
    return g


def natural_key(p: Path):
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r"(\d+)", p.name)]


def load_frames(folder: Path):
    files = sorted((p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS),
                   key=natural_key)
    if len(files) < 4:
        raise SystemExit(f"В папке {folder} нужно минимум 4 изображения.")
    frames = []
    for p in files:
        im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if im is None:
            raise SystemExit(f"Не удалось прочитать {p}")
        if im.ndim == 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        if im.dtype == np.uint8:
            im = im.astype(np.float32) / 255.0
        elif im.dtype == np.uint16:
            im = im.astype(np.float32) / 65535.0
        else:
            im = im.astype(np.float32)
            im /= max(float(im.max()), 1e-6)
        frames.append(im)
    shape = frames[0].shape
    if any(f.shape != shape for f in frames):
        raise SystemExit("Размеры кадров различаются.")
    return files, frames


# --------------------------- предобработка ---------------------------------
def attenuation(img01: np.ndarray) -> np.ndarray:
    """-ln(I): после этого вклады слоёв аддитивны."""
    return -np.log(np.maximum(img01, 1e-4)).astype(np.float32)


def highpass(a: np.ndarray, sigma_bg: float, sigma_noise: float = 0.0) -> np.ndarray:
    """Убирает плавный фон (градиент засветки, виньетирование)."""
    if sigma_bg >= 16:                      # фон гладкий - считаем на уменьшенной копии
        k = 4 if sigma_bg < 40 else 8
        small = cv2.resize(a, None, fx=1.0 / k, fy=1.0 / k, interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), sigma_bg / k)
        bg = cv2.resize(small, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR)
    else:
        bg = cv2.GaussianBlur(a, (0, 0), sigma_bg)
    out = a - bg
    if sigma_noise > 0:
        out = cv2.GaussianBlur(out, (0, 0), sigma_noise)
    return out


def make_stack(att_frames, scale: float, sigma_bg: float, sigma_noise: float):
    out = []
    for a in att_frames:
        s = a if scale == 1.0 else cv2.resize(a, None, fx=scale, fy=scale,
                                              interpolation=cv2.INTER_AREA)
        out.append(highpass(s, sigma_bg * scale, sigma_noise))
    return out


# --------------------------- shift-and-add ---------------------------------
def shift_img(img, dx, dy):
    h, w = img.shape
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)


def thetas_for(n, step, sense, phi0_deg):
    return [math.radians(sense * i * step + phi0_deg) for i in range(n)]


def shift_and_add(stack, thetas, r_px):
    """Совмещает слой с радиусом r_px во всех кадрах и усредняет."""
    acc = np.zeros_like(stack[0])
    for f, th in zip(stack, thetas):
        acc += shift_img(f, -r_px * math.cos(th), -r_px * math.sin(th))
    return acc / len(stack)


def sharpness(img: np.ndarray) -> float:
    """Энергия градиента (Tenengrad): растёт, когда слой в фокусе."""
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.mean(gx * gx + gy * gy))


# ------------------------ автокалибровка геометрии -------------------------
def local_maxima(y: np.ndarray, min_prom: float):
    idx = []
    for i in range(1, len(y) - 1):
        if y[i] >= y[i - 1] and y[i] > y[i + 1]:
            lo = max(y[:i + 1].min(), y[i:].min())      # грубая «заметность»
            if y[i] - lo >= min_prom:
                idx.append(i)
    return idx


def parabolic(x, y, i):
    if i <= 0 or i >= len(y) - 1:
        return float(x[i])
    a, b, c = y[i - 1], y[i], y[i + 1]
    den = a - 2 * b + c
    if abs(den) < 1e-12:
        return float(x[i])
    off = 0.5 * (a - c) / den
    return float(x[i] + off * (x[1] - x[0]))


def focus_curve(stack, thetas, r_values, scale):
    return np.array([sharpness(shift_and_add(stack, thetas, r * scale))
                     for r in r_values])


def pick_two_layers(r_vals, curve, rg):
    """Выбирает два самых выраженных пика кривой резкости.
    rg - ожидаемое расстояние между слоями из лога (мягкая подсказка)."""
    c = (curve - curve.min()) / max(np.ptp(curve), 1e-12)
    k = max(3, int(round(len(c) / 60)))
    cs = np.convolve(c, np.ones(k) / k, mode="same")
    peaks = local_maxima(cs, 0.03)
    if len(peaks) < 2:
        return None
    min_sep = max(0.3 * rg, 3.0)
    best, best_score = None, -1
    for a in range(len(peaks)):
        for b in range(a + 1, len(peaks)):
            ia, ib = peaks[a], peaks[b]
            sep = abs(r_vals[ia] - r_vals[ib])
            if sep < min_sep:
                continue
            w = 0.6 + 0.4 / (1.0 + ((sep - rg) / (0.6 * rg)) ** 2)
            score = (cs[ia] + cs[ib]) * w
            if score > best_score:
                best, best_score = (ia, ib), score
    if best is None:
        return None
    ra = parabolic(r_vals, cs, best[0])
    rb = parabolic(r_vals, cs, best[1])
    return tuple(sorted((ra, rb)))


def calibrate(att, n, g, log_sense):
    """Возвращает (sense, phi0_deg, rA, rB, rg) - параметры движения слоёв."""
    step = g["step"]
    rg = abs(g["delta_z"] * math.tan(math.radians(g["angle"])) / g["pix_mm"])
    print(f"Ожидание по логу: разнос слоёв ≈ ΔZ·tg(угол)/пиксель = {rg:.1f} px")
    r_max = max(2.5 * rg, 30.0)

    # ---- этап 1: грубый перебор (1/4 масштаба, часть кадров) ---------------
    sc = 0.2
    sel = np.linspace(0, n - 1, min(n, 14)).round().astype(int)
    stack_c = make_stack([att[i] for i in sel], sc, 10.0, 0.6)
    r_grid = np.arange(-r_max, r_max + 1e-6, 2.5)
    excl = 0.25 * rg
    senses = (log_sense, -log_sense) if FORCE_SENSE is None else (FORCE_SENSE,)
    results = {}
    t0 = time.time()
    for sense in senses:
        for phi0 in range(0, 180, 20):
            if FORCE_PHI0_DEG is not None:
                phi0 = FORCE_PHI0_DEG % 180
            th = [math.radians(sense * i * step + phi0) for i in sel]
            cur = focus_curve(stack_c, th, r_grid, sc)
            m = np.abs(r_grid) >= excl
            results[(sense, phi0)] = (float(cur[m].max()),
                                      float(r_grid[m][cur[m].argmax()]))
            if FORCE_PHI0_DEG is not None:
                break
    print(f"  грубый перебор: {time.time() - t0:.1f} с")

    def best_for(sense):
        items = [(v[0], k[1], v[1]) for k, v in results.items() if k[0] == sense]
        return max(items)

    cand = {s: best_for(s) for s in senses}
    sense = max(cand, key=lambda s: cand[s][0])
    if FORCE_SENSE is None and log_sense in cand and \
            cand[log_sense][0] >= 0.97 * cand[sense][0]:
        sense = log_sense                         # доверяем логу при ничьей
    _, phi0, r_hint = cand[sense]

    # ---- этап 2: уточнение фазы (1/2 масштаба, все кадры) ------------------
    sc = 0.25
    stack_f = make_stack(att, sc, 12.0, 0.7)
    if FORCE_PHI0_DEG is None:
        best = (-1, phi0, r_hint)
        for dphi in np.arange(-16, 16.1, 2.0):
            p = phi0 + dphi
            th = thetas_for(n, step, sense, p)
            rr = np.arange(r_hint - 4, r_hint + 4.01, 1.0)
            cur = focus_curve(stack_f, th, rr, sc)
            if cur.max() > best[0]:
                best = (float(cur.max()), p, float(rr[cur.argmax()]))
        phi0 = best[1]
    else:
        phi0 = FORCE_PHI0_DEG

    # ---- этап 3: полная кривая резкости и выбор двух слоёв -----------------
    th = thetas_for(n, step, sense, phi0)
    r_vals = np.arange(-r_max, r_max + 1e-6, 0.75)
    curve = focus_curve(stack_f, th, r_vals, sc)
    pair = pick_two_layers(r_vals, curve, rg)
    if pair is None:
        print("[!] Не нашёл два чётких пика резкости, беру подсказку по логу.")
        i0 = int(curve.argmax())
        ra = float(r_vals[i0])
        pair = tuple(sorted((ra, ra + rg if ra <= 0 else ra - rg)))
    ra, rb = pair

    # ---- этап 4: тонкая подстройка фазы по обоим слоям ---------------------
    if FORCE_PHI0_DEG is None:
        best = (-1, phi0)
        for dphi in np.arange(-3, 3.01, 1.0):
            p = phi0 + dphi
            th = thetas_for(n, step, sense, p)
            s = sum(sharpness(shift_and_add(stack_f, th, r * sc)) for r in (ra, rb))
            if s > best[0]:
                best = (s, p)
        phi0 = best[1]

    # ---- этап 5: точное уточнение радиусов на 1/2 масштаба ------------------
    sc2 = 0.5
    stack_h = make_stack(att, sc2, 15.0, 0.8)
    th = thetas_for(n, step, sense, phi0)
    refined = []
    for r0 in (ra, rb):
        rr = np.arange(r0 - 1.5, r0 + 1.51, 0.5)
        cur = focus_curve(stack_h, th, rr, sc2)
        i = int(cur.argmax())
        refined.append(parabolic(rr, cur, i) if 0 < i < len(rr) - 1 else float(rr[i]))
    ra, rb = refined

    save_curve(r_vals, curve)
    return sense, phi0 % 360, ra, rb, rg


def save_curve(r_vals, curve):
    """Небольшой график резкости от r (для отладки): OUTPUT_DIR/focus_curve.png."""
    h, w = 300, 800
    img = np.full((h, w, 3), 255, np.uint8)
    c = (curve - curve.min()) / max(np.ptp(curve), 1e-12)
    pts = [(int(10 + (w - 20) * i / (len(c) - 1)), int(h - 20 - (h - 40) * v))
           for i, v in enumerate(c)]
    cv2.polylines(img, [np.array(pts, np.int32)], False, (200, 60, 0), 2)
    x0 = int(10 + (w - 20) * (0 - r_vals[0]) / (r_vals[-1] - r_vals[0]))
    cv2.line(img, (x0, 10), (x0, h - 10), (150, 150, 150), 1)
    cv2.putText(img, f"r = {r_vals[0]:.0f} ... {r_vals[-1]:.0f} px", (10, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUT_DIR / "focus_curve.png"), img)


# --------------- точное разделение двух слоёв (частотная область) ----------
def separate_two_layers(stack, thetas, rA, rB, reg):
    """Решает y_i = A(x-d_i^A) + B(x-d_i^B) методом наименьших квадратов
    отдельно для каждой пространственной частоты (матрица 2x2)."""
    n = len(stack)
    h, w = stack[0].shape
    pad = int(math.ceil(max(abs(rA), abs(rB)))) + 24
    hp = _fft.next_fast_len(h + 2 * pad)
    wp = _fft.next_fast_len(w + 2 * pad)
    top, left = pad, pad
    bottom, right = hp - h - top, wp - w - left

    fy = np.fft.fftfreq(hp).astype(np.float32)[:, None]
    fx = np.fft.rfftfreq(wp).astype(np.float32)[None, :]

    def ramp(r, th):
        dx, dy = r * math.cos(th), r * math.sin(th)
        ex = np.exp(-2j * np.pi * fx * dx).astype(np.complex64)
        ey = np.exp(-2j * np.pi * fy * dy).astype(np.complex64)
        return ey * ex

    b1 = np.zeros((hp, wp // 2 + 1), np.complex64)
    b2 = np.zeros_like(b1)
    cc = np.zeros_like(b1)
    for f, th in zip(stack, thetas):
        p = np.pad(f, ((top, bottom), (left, right)), mode="reflect")
        y = _fft.rfft2(p.astype(np.float32), **_FFT_KW)
        ea, eb = ramp(rA, th), ramp(rB, th)
        b1 += np.conj(ea) * y
        b2 += np.conj(eb) * y
        cc += np.conj(ea) * eb

    lam = reg * n
    d = (n + lam)
    det = d * d - (cc.real ** 2 + cc.imag ** 2)
    A = (d * b1 - cc * b2) / det
    B = (d * b2 - np.conj(cc) * b1) / det
    a = _fft.irfft2(A, s=(hp, wp), **_FFT_KW)[top:top + h, left:left + w]
    b = _fft.irfft2(B, s=(hp, wp), **_FFT_KW)[top:top + h, left:left + w]
    return a.astype(np.float32), b.astype(np.float32)


# ------------------------------ вывод --------------------------------------
def to_view(layer: np.ndarray, dense_dark: bool) -> np.ndarray:
    """Нормировка + локальный контраст для просмотра (8 бит)."""
    lo, hi = np.percentile(layer, [0.5, 99.7])
    x = np.clip((layer - lo) / max(hi - lo, 1e-9), 0, 1)
    if dense_dark:
        x = 1.0 - x
    u8 = (x * 255).astype(np.uint8)
    u8 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(u8)
    return u8


def to_16(layer: np.ndarray, dense_dark: bool) -> np.ndarray:
    lo, hi = np.percentile(layer, [0.5, 99.7])
    x = np.clip((layer - lo) / max(hi - lo, 1e-9), 0, 1)
    if dense_dark:
        x = 1.0 - x
    return np.round(x * 65535).astype(np.uint16)


def edge_correlation(a, b):
    ga = cv2.Laplacian(cv2.GaussianBlur(a, (0, 0), 1.5), cv2.CV_32F)
    gb = cv2.Laplacian(cv2.GaussianBlur(b, (0, 0), 1.5), cv2.CV_32F)
    ga, gb = ga - ga.mean(), gb - gb.mean()
    return float((ga * gb).sum() / (np.sqrt((ga ** 2).sum() * (gb ** 2).sum()) + 1e-12))


# ------------------------------- main --------------------------------------
def main():
    t_start = time.time()
    if not INPUT_DIR.is_dir():
        raise SystemExit(f"Папка с изображениями не найдена: {INPUT_DIR.resolve()}")
    files, frames = load_frames(INPUT_DIR)
    n = len(frames)
    g = parse_geometry(LOG_PATH, n)
    print(f"Кадров: {n} (в логе: {g['n_expected']}), шаг {g['step']}°, "
          f"вращение {'CW' if g['cw'] else 'CCW'}, "
          f"угол {g['angle']}°, пиксель {g['pix_mm'] * 1000:.3f} мкм")
    if n != g["n_expected"]:
        print("[!] Число кадров не совпадает с логом - углы считаю по шагу вращения.")

    if n * g["step"] < 300:
        print(f"[!] Кадры покрывают только {n * g['step']:.0f}° из 360° - слои разделятся "
              "хуже, а положение слоёв определится неоднозначно. Для лучшего результата "
              "нужна полная серия.")
    att = [attenuation(f) for f in frames]
    log_sense = 1 if g["cw"] else -1

    if FORCE_R_LAYERS is not None and FORCE_PHI0_DEG is not None:
        sense = FORCE_SENSE or log_sense
        phi0, (ra, rb) = FORCE_PHI0_DEG, sorted(FORCE_R_LAYERS)
        rg = abs(rb - ra)
    else:
        sense, phi0, ra, rb, rg = calibrate(att, n, g, log_sense)
        if FORCE_R_LAYERS is not None:
            ra, rb = sorted(FORCE_R_LAYERS)
    print(f"Найдено: направление={'+' if sense > 0 else '-'}, φ0={phi0:.1f}°, "
          f"r1={ra:.2f} px, r2={rb:.2f} px (разнос {rb - ra:.2f} px; по логу ≈ {rg:.1f})")

    # ---- точное разделение на полном разрешении ---------------------------
    stack = make_stack(att, 1.0, SIGMA_BG, 0.0)
    th = thetas_for(n, g["step"], sense, phi0)
    print("Разделяю слои...")
    L1, L2 = separate_two_layers(stack, th, ra, rb, REGULARIZATION)
    layers = {"layer_r1": L1, "layer_r2": L2}
    if TOP_IS_LARGER_R:
        names = {"layer_r1": "bottom_layer", "layer_r2": "top_layer"}
    else:
        names = {"layer_r1": "top_layer", "layer_r2": "bottom_layer"}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for key, img in layers.items():
        nm = names[key]
        cv2.imwrite(str(OUTPUT_DIR / f"{nm}.png"), to_view(img, DENSE_IS_DARK))
        cv2.imwrite(str(OUTPUT_DIR / f"{nm}_16bit.png"), to_16(img, DENSE_IS_DARK))
    print(f"Корреляция границ между слоями (меньше = лучше разделены): "
          f"{edge_correlation(L1, L2):.3f}")

    # ---- запасной вариант: набор срезов shift-and-add ---------------------
    if SAVE_SLICES:
        sd = OUTPUT_DIR / "slices"
        sd.mkdir(exist_ok=True)
        for old in sd.glob("slice_*.png"):          # убираем срезы прошлого запуска
            old.unlink()
        span = max(abs(rb - ra), 6.0)
        for k, r in enumerate(np.linspace(min(ra, rb) - 0.4 * span,
                                          max(ra, rb) + 0.4 * span, N_SLICES)):
            sl = shift_and_add(stack, th, float(r))
            cv2.imwrite(str(sd / f"slice_{k:02d}_r{r:+.1f}px.png"),
                        to_view(sl, DENSE_IS_DARK))
    print(f"Готово за {time.time() - t_start:.0f} с. Результаты: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()