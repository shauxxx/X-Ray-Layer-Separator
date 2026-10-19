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

NUM_LAYERS = 2               # 2 = верх + низ; 3 = ещё отдельный слой высоких компонентов
                             # (кварц, разъёмы) - они тогда не «пачкают» слои меди

DENSE_IS_DARK = True         # True: металл тёмный (как на исходных снимках)
SAVE_SLICES = True           # дополнительно сохранить набор срезов shift-and-add
SLICE_MARGIN_PX = 30.0       # на сколько px (по r) выходить за найденные слои
SLICE_STEP_PX = 1.5          # шаг между срезами по r
MAX_SLICES = 90
SLICE_R_RANGE = None         # или явно, например (5.0, 60.0) - тогда margin не нужен
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


def calibrate(att, n, g, log_sense):
    """Возвращает (sense, phi0_deg, [радиусы слоёв], rg)."""
    step = g["step"]
    rg = abs(g["delta_z"] * math.tan(math.radians(g["angle"])) / g["pix_mm"])
    print(f"Ожидание по логу: разнос слоёв ≈ ΔZ·tg(угол)/пиксель = {rg:.1f} px")
    r_max = max(2.5 * rg, 60.0)

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

    # ---- этап 3: кривая резкости -> область, где вообще есть слои -----------
    th = thetas_for(n, step, sense, phi0)
    r_vals = np.arange(-r_max, r_max + 1e-6, 1.0)
    curve = focus_curve(stack_f, th, r_vals, sc)
    c = (curve - curve.min()) / max(np.ptp(curve), 1e-12)
    idx = np.where(c > 0.12)[0]
    lo, hi = float(r_vals[idx.min()]) - 4.0, float(r_vals[idx.max()]) + 4.0
    print(f"  область со слоями по резкости: r от {lo:.0f} до {hi:.0f} px")

    # ---- этап 4: радиусы слоёв по невязке модели ----------------------------
    if FORCE_R_LAYERS is not None:
        rs = sorted(float(r) for r in FORCE_R_LAYERS)
    else:
        rs, _ = find_layers(att, n, step, sense, phi0, NUM_LAYERS, lo, hi)

    # ---- этап 5: тонкая подстройка фазы по невязке --------------------------
    if FORCE_PHI0_DEG is None:
        sp4 = Spectra(stack_f, 0.25, max(abs(lo), abs(hi)) + 40)
        best = (1e9, phi0)
        for dphi in np.arange(-3, 3.01, 1.0):
            v, _ = sp4.solve(thetas_for(n, step, sense, phi0 + dphi), rs, 0.003)
            if v < best[0]:
                best = (v, phi0 + dphi)
        phi0 = best[1]

    save_curve(r_vals, curve, rs)
    return sense, phi0 % 360, rs, rg


def save_curve(r_vals, curve, marks=()):
    """Небольшой график резкости от r (для отладки): OUTPUT_DIR/focus_curve.png."""
    h, w = 300, 800
    img = np.full((h, w, 3), 255, np.uint8)
    c = (curve - curve.min()) / max(np.ptp(curve), 1e-12)
    pts = [(int(10 + (w - 20) * i / (len(c) - 1)), int(h - 20 - (h - 40) * v))
           for i, v in enumerate(c)]
    cv2.polylines(img, [np.array(pts, np.int32)], False, (200, 60, 0), 2)
    x0 = int(10 + (w - 20) * (0 - r_vals[0]) / (r_vals[-1] - r_vals[0]))
    cv2.line(img, (x0, 10), (x0, h - 10), (150, 150, 150), 1)
    for r in marks:
        xm = int(10 + (w - 20) * (r - r_vals[0]) / (r_vals[-1] - r_vals[0]))
        cv2.line(img, (xm, 10), (xm, h - 10), (0, 0, 220), 1)
        cv2.putText(img, f"{r:.1f}", (xm + 3, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 220), 1)
    cv2.putText(img, f"r = {r_vals[0]:.0f} ... {r_vals[-1]:.0f} px", (10, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUT_DIR / "focus_curve.png"), img)


# --------------- разделение слоёв в частотной области -----------------------
class Spectra:
    """Спектры кадров на заданном масштабе. Для данного набора радиусов слоёв
    решает y_i = sum_k L_k(x - d_i(r_k)) методом наименьших квадратов отдельно
    для каждой частоты (матрица KxK) и считает невязку модели."""

    def __init__(self, stack, scale, r_abs_max, store=True):
        self.stack, self.scale, self.store = stack, scale, store
        self.n = len(stack)
        h, w = stack[0].shape
        self.h, self.w = h, w
        pad = int(math.ceil(r_abs_max * scale)) + 8
        self.pad = pad
        self.hp = _fft.next_fast_len(h + 2 * pad)
        self.wp = _fft.next_fast_len(w + 2 * pad)
        self.fy = np.fft.fftfreq(self.hp).astype(np.float32)[:, None]
        self.fx = np.fft.rfftfreq(self.wp).astype(np.float32)[None, :]
        wt = np.full((1, self.fx.shape[1]), 2.0, np.float32)
        wt[0, 0] = 1.0
        if self.wp % 2 == 0:
            wt[0, -1] = 1.0
        self.wt = wt
        self._Y = [self._fft_frame(f) for f in stack] if store else None
        self.eyy = (sum(float((wt * (y.real ** 2 + y.imag ** 2)).sum()) for y in self._Y)
                    if store else None)

    def _fft_frame(self, f):
        p = np.pad(f, ((self.pad, self.hp - self.h - self.pad),
                       (self.pad, self.wp - self.w - self.pad)), mode="reflect")
        return _fft.rfft2(p.astype(np.float32), **_FFT_KW)

    def _y(self, i):
        return self._Y[i] if self.store else self._fft_frame(self.stack[i])

    def _ramp(self, r, th):
        dx = r * self.scale * math.cos(th)
        dy = r * self.scale * math.sin(th)
        ex = np.exp(-2j * np.pi * self.fx * dx).astype(np.complex64)
        ey = np.exp(-2j * np.pi * self.fy * dy).astype(np.complex64)
        return ey * ex

    def solve(self, thetas, rs, reg, images=False):
        """Возвращает (относительная невязка, [изображения слоёв] или None)."""
        K, n = len(rs), self.n
        lam = reg * n
        hp, wf = self.hp, self.fx.shape[1]
        b = [np.zeros((hp, wf), np.complex64) for _ in range(K)]
        G = {}
        for k in range(K):
            for l in range(k + 1, K):
                G[(k, l)] = np.zeros((hp, wf), np.complex64)
        for i, th in enumerate(thetas):
            y = self._y(i)
            E = [self._ramp(r, th) for r in rs]
            for k in range(K):
                b[k] += np.conj(E[k]) * y
                for l in range(k + 1, K):
                    G[(k, l)] += np.conj(E[k]) * E[l]

        if K == 2:
            cc = G[(0, 1)]
            d = n + lam
            det = d * d - (cc.real ** 2 + cc.imag ** 2)
            X = [(d * b[0] - cc * b[1]) / det, (d * b[1] - np.conj(cc) * b[0]) / det]
        else:
            X = [np.zeros((hp, wf), np.complex64) for _ in range(K)]
            step = 64
            for r0 in range(0, hp, step):
                r1 = min(hp, r0 + step)
                M = np.zeros((r1 - r0, wf, K, K), np.complex64)
                for k in range(K):
                    M[:, :, k, k] = n + lam
                    for l in range(k + 1, K):
                        M[:, :, k, l] = G[(k, l)][r0:r1]
                        M[:, :, l, k] = np.conj(G[(k, l)][r0:r1])
                rhs = np.stack([b[k][r0:r1] for k in range(K)], axis=-1)[..., None]
                sol = np.linalg.solve(M, rhs)[..., 0]
                for k in range(K):
                    X[k][r0:r1] = sol[..., k]

        res = None
        if self.store:
            q = sum(2.0 * (np.conj(X[k]) * b[k]).real for k in range(K))
            xgx = sum(n * (X[k].real ** 2 + X[k].imag ** 2) for k in range(K))
            for (k, l), g in G.items():
                xgx = xgx + 2.0 * (np.conj(X[k]) * g * X[l]).real
            res = float((self.wt * (xgx - q)).sum() + self.eyy) / self.eyy
        imgs = None
        if images:
            imgs = [_fft.irfft2(x, s=(hp, self.wp), **_FFT_KW)
                    [self.pad:self.pad + self.h, self.pad:self.pad + self.w].astype(np.float32)
                    for x in X]
        return res, imgs


def refine_radii(sp, th, rs, reg, widths=(3.0, 1.5, 0.75), steps=(1.0, 0.5, 0.25),
                 min_sep=2.5):
    """Покоординатный спуск по радиусам слоёв: минимизируем невязку модели."""
    rs = list(rs)
    best, _ = sp.solve(th, rs, reg)
    for hw, st in zip(widths, steps):
        for k in range(len(rs)):
            for cand in np.arange(rs[k] - hw, rs[k] + hw + 1e-9, st):
                trial = rs.copy()
                trial[k] = float(cand)
                srt = sorted(trial)
                if any(b - a < min_sep for a, b in zip(srt, srt[1:])):
                    continue
                v, _ = sp.solve(th, trial, reg)
                if v < best:
                    best, rs = v, trial
    return sorted(rs), best


def find_layers(att, n, step, sense, phi0, K, lo, hi):
    """Ищет радиусы K слоёв, при которых модель «K слоёв» лучше всего объясняет
    все кадры (минимум невязки). Работает и когда слои стоят близко и пики
    резкости сливаются в один горб."""
    th = thetas_for(n, step, sense, phi0)
    reg = 0.003
    sc = 0.125
    sp8 = Spectra(make_stack(att, sc, 12.0, 0.5), sc, max(abs(lo), abs(hi)) + 40)
    grid = np.arange(lo, hi + 1e-6, 2.0)
    best, rs = 1e9, None
    for i, ra in enumerate(grid):
        for rb in grid[i + 1:]:
            if rb - ra < 3.0:
                continue
            v, _ = sp8.solve(th, (ra, rb), reg)
            if v < best:
                best, rs = v, [float(ra), float(rb)]
    print(f"  слои (грубо): r = {rs[0]:.1f}, {rs[1]:.1f} px, невязка {best:.3f}")

    if K >= 3:
        for _ in range(K - 2):
            cand_grid = np.arange(lo - 10, hi + 40 + 1e-6, 2.0)
            b3, r3 = 1e9, None
            for rc in cand_grid:
                if any(abs(rc - r) < 3.0 for r in rs):
                    continue
                v, _ = sp8.solve(th, rs + [float(rc)], reg)
                if v < b3:
                    b3, r3 = v, float(rc)
            print(f"  добавлен слой r≈{r3:.1f} px: невязка {best:.3f} -> {b3:.3f}")
            rs, best = sorted(rs + [r3]), b3

    sp4 = Spectra(make_stack(att, 0.25, 12.0, 0.7), 0.25, max(abs(lo), abs(hi)) + 40)
    rs, best = refine_radii(sp4, th, rs, reg)
    return rs, best


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
def local_flat(layer: np.ndarray, sigma: float = 25.0) -> np.ndarray:
    """Локальная нормировка (среднее 0, разброс 1): убирает ореолы у краёв платы
    и выравнивает контраст по всему полю - удобно для поиска дорожек."""
    mu = cv2.GaussianBlur(layer, (0, 0), sigma)
    d = layer - mu
    sd = np.sqrt(cv2.GaussianBlur(d * d, (0, 0), sigma)) + 1e-6 * float(layer.std() + 1e-9)
    sd = np.maximum(sd, 0.15 * float(np.median(sd)))
    return d / sd


def layer_names(rs):
    k = len(rs)
    if k == 2:
        return ["bottom_layer", "top_layer"] if TOP_IS_LARGER_R else ["top_layer", "bottom_layer"]
    order = list(range(k)) if TOP_IS_LARGER_R else list(range(k - 1, -1, -1))
    return [f"layer_{order[i] + 1}_r{rs[i]:+.1f}px" for i in range(k)]


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
        phi0, rs = FORCE_PHI0_DEG, sorted(float(r) for r in FORCE_R_LAYERS)
        rg = abs(rs[-1] - rs[0])
    else:
        sense, phi0, rs, rg = calibrate(att, n, g, log_sense)
    tan_a = math.tan(math.radians(g["angle"]))
    z_mm = [r * g["pix_mm"] / tan_a for r in rs]
    print(f"Найдено: направление={'+' if sense > 0 else '-'}, φ0={phi0:.1f}°")
    for r, z in zip(rs, z_mm):
        print(f"   слой: r = {r:+.2f} px  (≈ {z:.2f} мм над опорной плоскостью)")
    if len(rs) >= 2:
        print(f"   разнос крайних слоёв: {rs[-1] - rs[0]:.2f} px ≈ {z_mm[-1] - z_mm[0]:.2f} мм")

    # ---- точное разделение на полном разрешении ---------------------------
    stack = make_stack(att, 1.0, SIGMA_BG, 0.0)
    th = thetas_for(n, g["step"], sense, phi0)
    print("Разделяю слои на полном разрешении...")
    sp = Spectra(stack, 1.0, max(abs(r) for r in rs), store=False)
    _, layers = sp.solve(th, rs, REGULARIZATION, images=True)
    names = layer_names(rs)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for nm, img in zip(names, layers):
        cv2.imwrite(str(OUTPUT_DIR / f"{nm}.png"), to_view(img, DENSE_IS_DARK))
        cv2.imwrite(str(OUTPUT_DIR / f"{nm}_16bit.png"), to_16(img, DENSE_IS_DARK))
        flat = local_flat(img)
        lo, hi = np.percentile(flat, [0.5, 99.5])
        x = np.clip((flat - lo) / max(hi - lo, 1e-9), 0, 1)
        x = 1.0 - x if DENSE_IS_DARK else x
        cv2.imwrite(str(OUTPUT_DIR / f"{nm}_flat.png"), (x * 255).astype(np.uint8))
        np.save(OUTPUT_DIR / f"{nm}_float32.npy", img)        # «сырой» слой для анализа
    for i in range(len(layers)):
        for j in range(i + 1, len(layers)):
            print(f"Корреляция границ {names[i]} / {names[j]}: "
                  f"{edge_correlation(layers[i], layers[j]):+.3f} (ближе к 0 = лучше)")
    with open(OUTPUT_DIR / "geometry.txt", "w", encoding="utf-8") as fh:
        fh.write(f"sense={sense}\nphi0_deg={phi0:.2f}\nrotation_step_deg={g['step']}\n")
        for nm, r, z in zip(names, rs, z_mm):
            fh.write(f"{nm}: r_px={r:.3f} z_mm={z:.3f}\n")

    # ---- запасной вариант: набор срезов shift-and-add ---------------------
    if SAVE_SLICES:
        sd = OUTPUT_DIR / "slices"
        sd.mkdir(exist_ok=True)
        for old in sd.glob("slice_*.png"):          # убираем срезы прошлого запуска
            old.unlink()
        if SLICE_R_RANGE is not None:
            r_lo, r_hi = SLICE_R_RANGE
        else:
            r_lo, r_hi = min(rs) - SLICE_MARGIN_PX, max(rs) + SLICE_MARGIN_PX
        count = int(min(MAX_SLICES, max(2, round((r_hi - r_lo) / SLICE_STEP_PX) + 1)))
        r_list = np.linspace(r_lo, r_hi, count)
        print(f"Срезы: {count} шт., r от {r_lo:.1f} до {r_hi:.1f} px")
        for k, r in enumerate(r_list):
            sl = shift_and_add(stack, th, float(r))
            cv2.imwrite(str(sd / f"slice_{k:02d}_r{r:+.1f}px.png"),
                        to_view(sl, DENSE_IS_DARK))
    print(f"Готово за {time.time() - t_start:.0f} с. Результаты: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()