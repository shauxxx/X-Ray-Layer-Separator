import configparser
import os

class TomosyntheticLogParser:
    def __init__(self, log_file_path):
        if not os.path.exists(log_file_path):
            raise FileNotFoundError(f"Файл лога не найден по пути: {log_file_path}")
            
        self.config = configparser.ConfigParser()
        self.config.read(log_file_path, encoding='utf-8')
        
        # Парсим параметры
        self._parse_parameters()

    def _parse_parameters(self):
        # Секция [Acquisition]
        acq = self.config['Acquisition']
        self.num_files = acq.getint('Number Of Files')
        self.camera_to_source = acq.getfloat('Camera to Source (mm)')
        self.object_to_source = acq.getfloat('Object to Source (mm)')
        
        # Берем именно масштабированный размер пикселя и переводим в мм (из микрон)
        self.pixel_size_mm = acq.getfloat('Scaled Image Pixel Size (um)') / 1000.0
        
        self.rotation_step = acq.getfloat('Rotation Step (deg)')
        self.image_format = acq.get('Image Format').lower()
        
        # Секция [GeometryCorrection] (если понадобятся начальные углы)
        if 'GeometryCorrection' in self.config:
            geom = self.config['GeometryCorrection']
            self.start_angle = geom.getfloat('Angle', fallback=0.0)
        else:
            self.start_angle = 0.0

    def print_summary(self):
        """Выводит сводку параметров для проверки"""
        print("="*40)
        print("ПАРАМЕТРЫ СЪЕМКИ УСПЕШНО ЗАГРУЖЕНЫ:")
        print(f"Всего кадров: {self.num_files}")
        print(f"Расстояние Источник-Детектор (H): {self.camera_to_source} мм")
        print(f"Расстояние Источник-Объект (FOD): {self.object_to_source} мм")
        print(f"Размер пикселя детектора: {self.pixel_size_mm:.6f} мм")
        print(f"Шаг вращения источника: {self.rotation_step} град.")
        print("="*40)

# Пример изолированной проверки парсера:
if __name__ == "__main__":
    # Замените на имя вашего файла лога
    log_path = "xray_log.txt" 
    
    try:
        parser = TomosyntheticLogParser(log_path)
        parser.print_summary()
    except Exception as e:
        print(f"Ошибка при чтении лога: {e}")
