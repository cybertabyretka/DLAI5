import random
from typing import Tuple, List

import cv2
import numpy as np
from PIL import Image, ImageEnhance


class RandomBlur:
    """
    Применяет гауссово размытие к изображению с заданной вероятностью.
    """
    def __init__(self, p: float = 0.5, ksize: Tuple[int, int] = (5, 5)) -> None:
        """
        Функция инициализации класса для применения гауссового размытия к изображению с заданной вероятностью.
        :param p: Вероятность применения размытия.
        :param ksize: Размер ядра размытия (должен быть нечетным).
        """
        self.p = p
        self.ksize = ksize

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Применяет размытие к изображению, если выполнено условие вероятности.
        :param img: Входное изображение PIL.
        :return: Изображение с размытием или оригинал.
        """
        if random.random() > self.p:
            return img
        img_np = np.array(img)
        blurred = cv2.GaussianBlur(img_np, self.ksize, 0)
        return Image.fromarray(blurred)


def _find_perspective_coeffs(
        pa: List[Tuple[float, float]],
        pb: List[Tuple[float, float]]
) -> List[float]:
    """
    Вычисляет коэффициенты преобразования перспективы из точек pa в pb.
    :param pa: Список исходных точек (x, y).
    :param pb: Список целевых точек (x, y).
    :return: Список коэффициентов преобразования.
    """
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0,
                       -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1,
                       -p2[1]*p1[0], -p2[1]*p1[1]])

    A = matrix
    B = []
    for p in pb:
        B.append(p[0])
        B.append(p[1])

    res = [round(x, 6) for x in list(np.linalg.solve(A, B))]
    return res


class RandomPerspective:
    """
    Применяет искажение перспективы к изображению с заданной вероятностью.
    """
    def __init__(self, p: float = 0.5, distortion_scale: float = 0.5) -> None:
        """
        Функция инициализации класса для применения икажения перспективы к изображению с заданной вероятностью.
        :param p: Вероятность применения искажения.
        :param distortion_scale: Масштаб искажения (0 — нет искажения, 1 — максимум).
        """
        self.p = p
        self.distortion_scale = distortion_scale

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Применяет случайное преобразование перспективы.
        :param img: Входное изображение PIL.
        :return: Преобразованное изображение или оригинал.
        """
        if random.random() > self.p:
            return img

        width, height = img.size

        startpoints = [
            (0, 0),
            (width, 0),
            (width, height),
            (0, height),
        ]

        def distort(x: float, max_shift: float) -> float:
            return x + random.uniform(-max_shift, max_shift)

        max_dx = self.distortion_scale * width
        max_dy = self.distortion_scale * height

        endpoints = [
            (distort(0, max_dx), distort(0, max_dy)),
            (distort(width, max_dx), distort(0, max_dy)),
            (distort(width, max_dx), distort(height, max_dy)),
            (distort(0, max_dx), distort(height, max_dy)),
        ]

        coeffs = _find_perspective_coeffs(startpoints, endpoints)
        return img.transform((width, height), Image.PERSPECTIVE, coeffs, Image.BICUBIC)


class RandomBrightnessContrast:
    """
    Случайным образом изменяет яркость и контраст изображения.
    """
    def __init__(
            self,
            p: float = 0.5,
            brightness: Tuple[float, float] = (0.5, 1.5),
            contrast: Tuple[float, float] = (0.5, 1.5)
    ) -> None:
        """
        Функция инициализации класса для изменения яркость и контраст изображения.
        :param p: Вероятность применения изменений.
        :param brightness: Диапазон изменения яркости (min, max).
        :param contrast: Диапазон изменения контраста (min, max).
        """
        self.p = p
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Применяет изменения яркости и контраста к изображению.
        :param img: Входное изображение PIL.
        :return: Преобразованное изображение или оригинал.
        """
        if random.random() > self.p:
            return img
        b_factor = random.uniform(*self.brightness)
        img = ImageEnhance.Brightness(img).enhance(b_factor)
        c_factor = random.uniform(*self.contrast)
        img = ImageEnhance.Contrast(img).enhance(c_factor)
        return img
