import os
import cv2
import numpy as np
from scipy.ndimage import shift
# Импортируем наш парсер из первого файла
from log_parser import TomosyntheticLogParser

def reconstruct_layer(images_dir, log_data, h_offset):
    """
    images_dir: путь к папке с рентгеновскими снимками
    log_data: объект распарсенного лога TomosyntheticLogParser
    h_offset: высота слоя относительно фокальной плоскости (в мм)
    """
    # Вытаскиваем параметры, которые нашел парсер
    H = log_data.camera_to_source
    FOD = log_data.object_to_source
    pixel_size = log_data.pixel_size_mm
    step_deg = log_data.rotation_step
    num_frames = log_data.num_files
    img_ext = log_data.image_format

    # Находим все файлы снимков с нужным расширением и строго сортируем их
    files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(f'.{img_ext}')])
    
    if len(files) < num_frames:
        raise ValueError(f"В папке найдено только {len(files)} изображений, а лог требует {num_frames}!")

    # Читаем первый кадр, чтобы узнать разрешение (работаем с 16-битными снимками)
    first_frame_path = os.path.join(images_dir, files[0])
    first_frame = cv2.imread(first_frame_path, cv2.IMREAD_UNCHANGED)
    
    if first_frame is None:
        raise ValueError(f"Не удалось прочитать файл изображения: {first_frame_path}")
        
    # Инициализируем пустую матрицу с повышенной точностью float64 для накопления слоев
    accumulator = np.zeros(first_frame.shape, dtype=np.float64)
    
    print(f"Сборка слоя для смещения h = {h_offset:+.3f} мм...")

    for i in range(num_frames):
        # 1. Считаем текущий угол источника для этого кадра (в радианах)
        # Учитываем стартовый угол из лога, если он был не нулевой
        current_angle_deg = log_data.start_angle + (i * step_deg)
        angle_rad = np.radians(current_angle_deg)
        
        # 2. Вычисляем величину радиального сдвига (в мм) по оптической геометрии
        # Масштаб смещения проекции зависит от того, насколько слой выше/ниже фокальной плоскости
        shift_mm = h_offset * (FOD / (H - FOD - h_offset))
        
        # 3. Переводим миллиметры сдвига в пиксели матрицы детектора
        shift_pixels = shift_mm / pixel_size
        
        # 4. Проекция кругового движения: раскладываем сдвиг на смещения по X и Y
        shift_x = shift_pixels * np.cos(angle_rad)
        shift_y = shift_pixels * np.sin(angle_rad)
        
        # 5. Загружаем сам снимок
        frame_path = os.path.join(images_dir, files[i])
        frame = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED).astype(np.float64)
        
        # 6. Субпиксельный сдвиг кадра. 
        # В scipy.ndimage.shift координаты передаются как [Y, X]
        # Заполняем края средним значением кадра (cval), чтобы не было черных полос по краям
        shifted_frame = shift(frame, [shift_y, shift_x], mode='constant', cval=np.mean(frame))
        
        # Аккумулируем кадры
        accumulator += shifted_frame

    # Усредняем накопленную сумму интенсивностей рентгена
    final_layer = accumulator / num_frames
    
    # Возвращаем результат, приведенный обратно к исходному 16-битному диапазону (0-65535)
    return np.clip(final_layer, 0, 65535).astype(np.uint16)


# === ГЛАВНЫЙ ЗАПУСК ВСЕЙ ПРОГРАММЫ ===
if __name__ == "__main__":
    # Настройки путей (поменяйте на свои папки)
    LOG_FILE = "photo_for_test1/s.log"
    IMAGES_FOLDER = "./photo_for_test1"
    OUTPUT_FOLDER = "./reconstructed_layers"
    
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Шаг 1: Инициализируем парсер
    log_info = TomosyntheticLogParser(LOG_FILE)
    log_info.print_summary() # Выведет параметры в консоль для контроля
    
    # Шаг 2: Поиск слоев методом "сканирования по высоте"
    # Давайте пройдемся от -1.5 мм до +1.5 мм с шагом 0.2 мм, чтобы поймать оба слоя платы
    height_steps = np.arange(-3.4, 3, 0.3)
    
    print("\nНачало послойного сканирования платы...")
    for h in height_steps:
        layer_img = reconstruct_layer(IMAGES_FOLDER, log_info, h_offset=h)
        
        # Формируем имя файла (например: layer_+0.200.png)
        output_name = f"layer_{h:+.3f}.png"
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, output_name), layer_img)
        
    print(f"\nГотово! Все слои сохранены в папку: {OUTPUT_FOLDER}")
