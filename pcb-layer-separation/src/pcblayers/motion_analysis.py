import numpy as np
from skimage.filters import sobel, window

def _prep(img):
    """Убираем фон, оставляем границы (дорожки), гасим края окном Ханна."""
    img = img - img.mean()
    g = sobel(img)
    return g * window("hann", g.shape)

def _corr_surface(a, b, eps=1e-8):
    """Фазовая корреляция: пик стоит там, где a = b, сдвинутое на (dy, dx)."""
    R = np.fft.fft2(a) * np.conj(np.fft.fft2(b))
    R /= np.abs(R) + eps
    return np.fft.ifft2(R).real

def _top_peaks(c, k=2, radius=5):
    """k сильнейших пиков; вокруг найденного глушим область, чтобы не взять соседа."""
    c = c.copy()
    H, W = c.shape
    peaks = []
    for _ in range(k):
        iy, ix = np.unravel_index(np.argmax(c), c.shape)
        peaks.append((iy, ix, c[iy, ix]))
        ys = np.arange(iy - radius, iy + radius + 1) % H
        xs = np.arange(ix - radius, ix + radius + 1) % W
        c[np.ix_(ys, xs)] = -np.inf
    return peaks

def _to_shift(iy, ix, shape):
    """Индекс в массиве -> знаковое смещение (с учётом заворачивания)."""
    H, W = shape
    dy = iy - H if iy > H // 2 else iy
    dx = ix - W if ix > W // 2 else ix
    return dx, dy

def estimate_shifts(frames):
    """Возвращает:
    shifts   (N, 2, 2): [кадр][слой 0=нижний,1=верхний][dx, dy]
    strength (N, 2):    высота пика для каждого слоя (для оценки надёжности)
    """
    ref = _prep(frames[0])
    N = len(frames)
    shifts = np.zeros((N, 2, 2))
    strength = np.ones((N, 2))
    for k in range(1, N):
        c = _corr_surface(_prep(frames[k]), ref)
        peaks = [(_to_shift(iy, ix, c.shape), v) for iy, ix, v in _top_peaks(c)]
        # нижний слой = пик с меньшим смещением, верхний = с большим
        peaks.sort(key=lambda p: np.hypot(*p[0]))
        for layer in range(2):
            shifts[k, layer] = peaks[layer][0]
            strength[k, layer] = peaks[layer][1]
    return shifts, strength