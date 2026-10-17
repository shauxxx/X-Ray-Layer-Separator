import cv2
import numpy as np
import os
from pathlib import Path



def load_images_from_folder(folder_path):
    """Загружает все изображения из папки, сортирует по имени и переводит в grayscale."""
    images = []
    # Поддерживаемые расширения
    valid_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
    files = sorted([f for f in os.listdir(folder_path) 
                    if Path(f).suffix.lower() in valid_extensions])
    
    for filename in files:
        filepath = os.path.join(folder_path, filename)
        # Читаем в grayscale
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
        else:
            print(f"Предупреждение: не удалось прочитать {filename}")
    
    if not images:
        raise ValueError("В папке не найдено подходящих изображений.")
    
    # Проверяем, что все изображения одного размера
    h, w = images[0].shape
    for img in images:
        if img.shape != (h, w):
            raise ValueError("Все изображения должны быть одного размера.")
    
    return images

def estimate_shifts(images):
    """
    Оценивает кумулятивный сдвиг каждого кадра относительно первого
    с помощью фазовой корреляции.
    Возвращает массив сдвигов (dx, dy) для каждого кадра.
    """
    shifts = [(0.0, 0.0)]  # Первый кадр — опорный
    cumulative_shift = np.array([0.0, 0.0])
    
    # Для фазовой корреляции рекомендуется использовать окно Ханна
    window = cv2.createHanningWindow(images[0].shape[::-1], cv2.CV_32F)
    
    for i in range(1, len(images)):
        # Оцениваем сдвиг между текущим и предыдущим кадром
        (dx, dy), response = cv2.phaseCorrelate(
            np.float32(images[i-1]), 
            np.float32(images[i]), 
            window
        )
        # Накапливаем сдвиг
        cumulative_shift += np.array([dx, dy])
        shifts.append(tuple(cumulative_shift))
    
    return np.array(shifts)  # shape (N, 2)

def shift_image(img, shift):
    """Сдвигает изображение на заданный вектор (dx, dy) с билинейной интерполяцией."""
    dx, dy = shift
    # Матрица аффинного преобразования для сдвига
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), 
                             flags=cv2.INTER_LINEAR, 
                             borderMode=cv2.BORDER_REPLICATE)
    return shifted

def reconstruct_layer(images, shifts, compensate=True):
    """
    Восстанавливает слой путём усреднения кадров.
    Если compensate=True, кадры предварительно сдвигаются на -shifts (компенсация движения).
    """
    if not images:
        raise ValueError("Нет изображений для восстановления.")
    
    h, w = images[0].shape
    accumulator = np.zeros((h, w), dtype=np.float64)
    
    for i, (img, shift) in enumerate(zip(images, shifts)):
        if compensate:
            # Сдвигаем на противоположный вектор, чтобы выровнять движущийся слой
            img_shifted = shift_image(img, (-shift[0], -shift[1]))
        else:
            img_shifted = img
        accumulator += img_shifted.astype(np.float64)
    
    result = accumulator / len(images)
    return np.clip(result, 0, 255).astype(np.uint8)
# -------------------- CONFIGURATION --------------------

INPUT_DIR = Path("photo_for_test1")
OUTPUT_DIR = Path("output")

# Если SHIFTS = None, программа попытается оценить их автоматически
# (нижний слой считается неподвижным, движение приписывается верхнему).
SHIFTS = None

# Пример ручного задания (4 колонки: dx_top, dy_top, dx_bottom, dy_bottom):
# SHIFTS = np.array([
#     [ 0.0, 0.0, 0.0, 0.0],
#     [ 0.4, 0.0, 0.0, 0.0],
#     [ 0.8, 0.0, 0.0, 0.0],
# ], dtype=np.float32)

TV_WEIGHT = 0.002
MAX_ITER = 150
DOWNSCALE = 1

# Параметры автооценки сдвига
HIGH_PASS_SIGMA = 5.0      # сглаживание для high-pass; увеличьте, если много шума
MIN_RESPONSE = 0.02        # порог "уверенности" фазовой корреляции


# -------------------- AUTO SHIFT ESTIMATION --------------------

def estimate_shifts(frames, high_pass_sigma=HIGH_PASS_SIGMA,
                    min_response=MIN_RESPONSE):
    """
    Оценивает накопленный сдвиг верхнего слоя в каждом кадре
    относительно первого. Нижний слой предполагается неподвижным.

    Используется фазовая корреляция между соседними high-pass
    отфильтрованными кадрами (чтобы подчеркнуть края слоёв).
    Возвращает массив (n, 4): [dx_top, dy_top, 0, 0].
    """
    n, h, w = frames.shape
    shifts = np.zeros((n, 4), dtype=np.float32)

    window = cv2.createHanningWindow((w, h), cv2.CV_32F)

    def high_pass(img):
        # img - float32, [0..255]-подобный диапазон
        return img - cv2.GaussianBlur(img, (0, 0), high_pass_sigma)

    cumulative = np.zeros(2, dtype=np.float64)

    for i in range(1, n):
        f_prev = high_pass(frames[i - 1]).astype(np.float32)
        f_curr = high_pass(frames[i]).astype(np.float32)

        (dx, dy), response = cv2.phaseCorrelate(f_prev, f_curr, window)

        if response < min_response:
            print(f"  [warn] кадры {i-1}->{i}: низкая уверенность "
                  f"({response:.3f}). Попробуйте увеличить HIGH_PASS_SIGMA "
                  f"или задать SHIFTS вручную.")

        cumulative += np.array([dx, dy])
        shifts[i, 0] = cumulative[0]   # dx_top
        shifts[i, 1] = cumulative[1]   # dy_top
        # shifts[i, 2:4] = 0          # нижний слой неподвижен

    total = float(np.hypot(*cumulative))
    print(f"  Суммарный сдвиг верхнего слоя: "
          f"dx={cumulative[0]:+.2f}, dy={cumulative[1]:+.2f} "
          f"(|s|={total:.2f} px)")
    if total < 1.0:
        print("  [warn] Сдвиг очень маленький — фазовая корреляция, "
              "вероятно, зацепилась за неподвижный нижний слой. "
              "Проверьте результат / задайте SHIFTS вручную.")

    return shifts


# -------------------- MAIN --------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frames, paths = load_frames(INPUT_DIR)

    print(f"Загружено {len(paths)} кадров, форма={frames.shape[1:]}")

    if SHIFTS is None:
        print("SHIFTS не задан → автооценка (нижний слой неподвижен)...")
        shifts = estimate_shifts(frames)
        # frames уже в DOWNSCALE-масштабе, поэтому сдвиги тоже в нём
    else:
        shifts = np.asarray(SHIFTS, dtype=np.float32).copy()
        if shifts.shape != (len(frames), 4):
            raise ValueError(
                f"SHIFTS должен иметь форму ({len(frames)}, 4), "
                f"получено {shifts.shape}"
            )
        # SHIFTS заданы в исходных пикселях → пересчёт в масштаб DOWNSCALE
        shifts /= DOWNSCALE

    top, bottom, result = reconstruct(frames, shifts)

    save_image(OUTPUT_DIR / "top_layer.tif", top)
    save_image(OUTPUT_DIR / "bottom_layer.tif", bottom)

    print("Оптимизация завершена:", result.success)
    print("Финальное значение функции потерь:", result.fun)
    print("Результаты сохранены в:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()