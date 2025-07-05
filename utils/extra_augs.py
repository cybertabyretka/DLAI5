import random
from typing import Union, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps


def to_numpy(img: Union[Image.Image, torch.Tensor, np.ndarray]) -> np.ndarray:
    """
    Преобразует PIL Image, torch.Tensor или numpy.ndarray в numpy-массив формата CHW float32 в диапазоне [0, 1].
    :param img: Изображение (PIL.Image, torch.Tensor или numpy.ndarray)
    :return: numpy.ndarray (CHW, float32, [0,1])
    """
    if isinstance(img, torch.Tensor):
        arr = img.detach().cpu().numpy()
    elif isinstance(img, Image.Image):
        arr = np.array(img).astype(np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)
    elif isinstance(img, np.ndarray):
        arr = img.astype(np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0
        if arr.ndim == 3 and arr.shape[2] in (1,3):
            arr = arr.transpose(2, 0, 1)
    else:
        raise TypeError(f"Unsupported type {type(img)}")
    return arr


def to_tensor(img_np: np.ndarray) -> torch.Tensor:
    """
    Преобразует numpy-массив (CHW float32) в torch.Tensor.
    :param img_np: numpy.ndarray (CHW, float32)
    :return: torch.Tensor
    """
    return torch.from_numpy(img_np.copy())


def to_pil(img_np: np.ndarray) -> Image.Image:
    """
    Преобразует numpy-массив (CHW float32) в PIL Image.
    :param img_np: numpy.ndarray (CHW, float32, [0,1])
    :return: PIL.Image
    """
    arr = (img_np.transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


class AddGaussianNoise:
    """
    Добавляет гауссов шум к изображению.
    """
    def __init__(self, mean: float = 0.0, std: float = 0.1) -> None:
        """
        Функция инициализации класса для добавления гауссового шума к изображению.
        :param mean: Среднее значение шума.
        :param std: Стандартное отклонение шума.
        """
        self.mean = mean
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Возвращает тензор с добавленным к нему гауссовым шумом.
        :param tensor: Тензор, к которому нужно добавить гауссов шум.
        :return: Тензор с добавленным к нему гауссовым шумом.
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
        noise = torch.randn_like(tensor) * self.std + self.mean
        return tensor + noise


class RandomErasing:
    """
    Случайно затирает прямоугольную область изображения.
    """
    def __init__(
            self,
            p: float = 0.5,
            scale: Tuple[float, float] = (0.02, 0.2),
            ratio: Tuple[float, float] = (0.3, 3.3),
            value: int = 0
    ) -> None:
        """
        Функция инициализации класса для случайного затирания прямоугольной области изображения.
        :param p:  Вероятность применения операции.
        :param scale: Диапазон относительных размеров затираемой области от общей площади изображения.
        :param ratio: Диапазон сторон прямоугольника (aspect ratio).
        :param value: Значение, которым заполняется область.
        """
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value

    def __call__(
            self,
            img: Union[Image.Image, torch.Tensor, np.ndarray]
    ) -> Union[Image.Image, torch.Tensor, np.ndarray]:
        """
        Выполняет затирание прямоугольной области изображения.
        :param img: Изображение на вход.
        :return: Изображение с затёртой областью.
        """
        if random.random() > self.p:
            return img
        arr = to_numpy(img)
        c, h, w = arr.shape
        area = h * w

        for _ in range(10):
            target_area = random.uniform(*self.scale) * area
            aspect = random.uniform(*self.ratio)
            erase_h = int(round(np.sqrt(target_area * aspect)))
            erase_w = int(round(np.sqrt(target_area / aspect)))

            if erase_h < h and erase_w < w:
                x = random.randint(0, w - erase_w)
                y = random.randint(0, h - erase_h)
                arr[:, y : y + erase_h, x : x + erase_w] = self.value
                break

        if isinstance(img, Image.Image):
            return to_pil(arr)
        elif isinstance(img, torch.Tensor):
            return to_tensor(arr)
        else:
            return arr


class CutOut:
    """
    Вырезает одну или несколько квадратных областей (дырок) из изображения.
    """
    def __init__(self, p: float = 0.5, num_holes: int = 1, max_size: float = 0.2, value: int = 0):
        """
        Функция инициализации класса для вырезания областей из изображения.
        :param p: Вероятность применения операции.
        :param num_holes: Количество вырезаемых дырок.
        :param max_size: Максимальный размер дырки относительно размера изображения (0–1).
        :param value: Значение, которым заполняется дырка.
        """
        self.p = p
        self.num_holes = num_holes
        self.max_size = max_size
        self.value = value

    def __call__(
            self,
            img: Union[Image.Image, torch.Tensor, np.ndarray]
    ) -> Union[Image.Image, torch.Tensor, np.ndarray]:
        """
        Вырезает квадратную область из изображения.
        :param img: Изображение на вход.
        :return: Изображение с вырезанным квадратом.
        """
        if random.random() > self.p:
            return img
        arr = to_numpy(img)
        c, h, w = arr.shape

        for _ in range(self.num_holes):
            size = random.uniform(0, self.max_size) * min(h, w)
            hole_h = hole_w = int(size)
            if hole_h < 1:
                continue
            x = random.randint(0, w - hole_w)
            y = random.randint(0, h - hole_h)
            arr[:, y : y + hole_h, x : x + hole_w] = self.value

        if isinstance(img, Image.Image):
            return to_pil(arr)
        elif isinstance(img, torch.Tensor):
            return to_tensor(arr)
        else:
            return arr


class Solarize:
    """
    Инвертирует все пиксели, значения которых превышают порог.
    """
    def __init__(self, threshold: int = 128) -> None:
        """
        Функция инициализации класса для инвертирования пискселей, значения которых превышают порог.
        :param threshold: Порог, превышение которого вызывает инвертирование пикселя.
        """
        self.threshold = threshold

    def __call__(
            self,
            img: Union[Image.Image, np.ndarray]
    ) -> Union[Image.Image, np.ndarray]:
        """
        Выполняет инвертирование пикселей значения, которых превышают порог.
        :param img: Изображение на вход.
        :return: Обработанное изображение.
        """
        if isinstance(img, Image.Image):
            return ImageOps.solarize(img, threshold=self.threshold)
        arr = np.array(img)
        mask = arr > self.threshold
        arr[mask] = 255 - arr[mask]
        return arr


class Posterize:
    """
    Уменьшает количество бит на канал в изображении.
    """
    def __init__(self, bits: int = 4) -> None:
        """
        Функция инициализации класса для уменьшения количества бит на канал в изображении.
        :param bits: Количество бит на канал (например, 4 означает максимум 16 оттенков).
        """
        self.bits = bits

    def __call__(
            self,
            img: Union[Image.Image, np.ndarray]
    ) -> Union[Image.Image, np.ndarray]:
        """
        Выполняет уменьшение количества бит на канал в изображении.
        :param img: Изображение на вход.
        :return: Обработанное изображение.
        """
        if isinstance(img, Image.Image):
            return ImageOps.posterize(img, self.bits)
        arr = np.array(img).astype(np.uint8)
        shift = 8 - self.bits
        arr = np.right_shift(arr, shift)
        arr = np.left_shift(arr, shift)
        return arr


class AutoContrast:
    """
    Автоматически растягивает гистограмму изображения, максимизируя контраст.
    """
    def __init__(self, p: float = 0.5) -> None:
        """
        Функция инициализации класса для растягивания гистограммы изображения.
        :param p: вероятность выполнения операции.
        """
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Растягивает гистограмму изображения.
        :param img: Изображение на вход.
        :return: Обработанное изображение.
        """
        if random.random() > self.p or not isinstance(img, Image.Image):
            return img
        return ImageOps.autocontrast(img)


class ElasticTransform:
    """
    Применяет эластичное преобразование к изображению, имитируя случайные искажения (например, деформацию бумаги).
    """
    def __init__(self, p: float = 0.5, alpha: float = 1.0, sigma: int = 50) -> None:
        """
        Функция инициализации класса для применения эластичного преобразования.
        :param p: Вероятность применения операции.
        :param alpha: Масштаб смещения.
        :param sigma: Степень сглаживания смещений.
        """
        self.p = p
        self.alpha = alpha
        self.sigma = sigma

    def __call__(
            self,
            img: Union[Image.Image, torch.Tensor, np.ndarray]
    ) -> Union[Image.Image, torch.Tensor]:
        """
        Выполняет эластичное преобразование изображения.
        :param img: Изображение на вход.
        :return: Обработанное изображение.
        """
        if random.random() > self.p:
            return img
        arr = to_numpy(img).transpose(1, 2, 0)
        h, w = arr.shape[:2]

        dx = (np.random.rand(h, w) * 2 - 1) * self.alpha
        dy = (np.random.rand(h, w) * 2 - 1) * self.alpha

        dx = cv2.GaussianBlur(dx, (0, 0), self.sigma)
        dy = cv2.GaussianBlur(dy, (0, 0), self.sigma)

        x, y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)

        deformed = cv2.remap(arr, map_x, map_y, interpolation=cv2.INTER_LINEAR)
        deformed = deformed.transpose(2, 0, 1)
        tensor = to_tensor(deformed)

        if isinstance(img, Image.Image):
            return to_pil(deformed)
        return tensor


class MixUp:
    """
    Выполняет смешивание двух изображений по методу MixUp: комбинирует два изображения с весами, взятыми из бета-распределения.
    """
    def __init__(self, p: float = 0.5, alpha: float = 0.2) -> None:
        """
        Функция инициализации класса для смешивания двух изображений.
        :param p: Вероятность применения MixUp.
        :param alpha: Параметр бета-распределения.
        """
        self.p = p
        self.alpha = alpha

    def __call__(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Выполняет смешивание двух изображений.
        :param img1: Первое изображение.
        :param img2: Второе изображение.
        :return: Результат смешивания изображений.
        """
        if random.random() > self.p:
            return img1
        if not isinstance(img1, torch.Tensor) or not isinstance(img2, torch.Tensor):
            raise TypeError("MixUp inputs must be torch.Tensor")
        lam = np.random.beta(self.alpha, self.alpha)
        return lam * img1 + (1 - lam) * img2
