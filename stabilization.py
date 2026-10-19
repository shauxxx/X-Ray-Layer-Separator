
import cv2
import numpy as np


def register_projection(
    reference,
    moving,
    motion=cv2.MOTION_EUCLIDEAN,
    max_iterations=150,
    epsilon=1e-6,
):
    """
    Совмещает moving с reference посредством ECC.

    По умолчанию оценивает только сдвиг и поворот,
    не разрешая произвольное масштабирование.

    Returns
    -------
    aligned : ndarray
        Совмещённое изображение moving.
    valid_mask : ndarray[bool]
        Область, в которой после преобразования есть реальные данные.
    warp : ndarray
        Матрица преобразования 2x3.
    correlation : float
        Значение корреляции ECC (не вероятность правильного совмещения).
    """
    if reference is None or moving is None:
        raise ValueError("Оба изображения должны быть заданы")

    if reference.ndim != 2 or moving.ndim != 2:
        raise ValueError("Ожидаются одноканальные изображения")

    if reference.shape != moving.shape:
        raise ValueError("Изображения должны иметь одинаковый размер")

    if reference.shape[0] < 32 or reference.shape[1] < 32:
        raise ValueError("Изображения слишком малы")

    # Для регистрации используем нормализованные копии.
    # Исходные данные при этом не изменяются.
    def prepare(image):
        image = image.astype(np.float32)
        lo, hi = np.percentile(image, [1, 99])

        if hi <= lo:
            raise ValueError("Изображение не содержит достаточного контраста")

        image = np.clip((image - lo) / (hi - lo), 0, 1)
        image = cv2.GaussianBlur(image, (0, 0), 1.0)
        return image

    ref = prepare(reference)
    mov = prepare(moving)

    warp = np.eye(2, 3, dtype=np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        max_iterations,
        epsilon,
    )

    try:
        correlation, warp = cv2.findTransformECC(
            ref,
            mov,
            warp,
            motion,
            criteria,
            None,
            5,
        )
    except cv2.error as exc:
        raise RuntimeError(
            "ECC не смог зарегистрировать кадр. "
            "Проверь контраст, начальное совмещение и выбранную область."
        ) from exc

    height, width = reference.shape

    # ECC задаёт преобразование для обратного отображения:
    # используем WARP_INVERSE_MAP при применении матрицы.
    aligned = cv2.warpAffine(
        moving,
        warp,
        (width, height),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    # Та же геометрия для маски: нули обозначают пустые края.
    source_mask = np.full(moving.shape, 255, dtype=np.uint8)

    valid_mask = cv2.warpAffine(
        source_mask,
        warp,
        (width, height),
        flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ) > 0

    return aligned, valid_mask, warp, float(correlation)
