import os
import cv2
import numpy as np
from scipy.ndimage import shift

def reconstruct_shift_scan(images_dir, num_frames, step_deg, shift_amplitude):
    """
    images_dir: папка с 36 кадрами
    shift_amplitude: амплитуда (сила) сдвига в пикселях для текущего слоя
    """
    files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith('.png')])
    
    first_frame = cv2.imread(os.path.join(images_dir, files[0]), cv2.IMREAD_UNCHANGED)
    accumulator = np.zeros(first_frame.shape, dtype=np.float64)
    
    for i in range(num_frames):
        # Строго круговая траектория движения трубки по шагам 10 градусов
        angle_rad = np.radians(i * step_deg)
        
        # Вычисляем сдвиг по осям X и Y для этого конкретного кадра
        # Знак минус можно будет инвертировать, если что, но начнем с базового
        shift_x = shift_amplitude * np.cos(angle_rad)
        shift_y = shift_amplitude * np.sin(angle_rad)
        
        # Читаем кадр
        frame = cv2.imread(os.path.join(images_dir, files[i]), cv2.IMREAD_UNCHANGED).astype(np.float64)
        
        # Сдвигаем кадр целиком (в scipy: [Y, X])
        # borderMode: заполняем края средним значением, чтобы не было черных полос
        shifted_frame = shift(frame, [shift_y, shift_x], mode='nearest')
        
        accumulator += shifted_frame
        
    return np.clip(accumulator / num_frames, 0, 65535).astype(np.uint16)

if __name__ == "__main__":
    LOG_FILE = "photo_for_test1/s.log"
    IMAGES_FOLDER = "./photo_for_test1"
    OUTPUT_FOLDER = "./shift_scan_results"
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Параметры из вашего лога
    NUM_FILES = 36
    ROTATION_STEP = 10
    
    # МЫ УВЕЛИЧИВАЕМ ДИАПАЗОН СДВИГА ДЛЯ ПОИСКА СЛОЕВ
    # Перебираем силу сдвига от -60 до +60 пикселей с шагом в 2 пикселя.
    # Один слой платы сфокусируется при одной амплитуде, второй — при другой.
    amplitudes = np.arange(-60, 61, 2)
    
    print("Запуск послойного сканирования методом Shift-and-Add...")
    for amp in amplitudes:
        print(f"Сканируем слой с амплитудой сдвига: {amp} пикс.")
        layer_img = reconstruct_shift_scan(IMAGES_FOLDER, NUM_FILES, ROTATION_STEP, shift_amplitude=amp)
        
        # Сохраняем результат
        output_name = f"layer_amplitude_{amp:+03d}.png"
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, output_name), layer_img)
        
    print(f"\nГотово! Результаты сохранены в папку: {OUTPUT_FOLDER}")
