from typing import List, Callable

from PIL import Image


class AugmentationPipeline:
    """
    Простейший пайплайн аугментаций изображений.
    """
    def __init__(self) -> None:
        """
        Функция инициализации пайплайна аугментаций изображений.
        """
        self.augmentations = {}

    def add_augmentation(self, name: str, aug: Callable[[Image.Image], Image.Image]) -> None:
        """
        Добавление аугментации в пайплайн.
        :param name: Имя аугментации.
        :param aug: Аугментация.
        :return: None
        """
        self.augmentations[name] = aug

    def remove_augmentation(self, name: str) -> None:
        """
        Удаление аугментации из пайплайна.
        :param name: Имя аугментации.
        :return: None
        """
        if name in self.augmentations:
            del self.augmentations[name]

    def apply(self, image: Image.Image) -> Image.Image:
        """
        Применение всех аугментаций к изображению.
        :param image: Изображение, на которого необходмио применить аугментации.
        :return: Аугментированное изображение.
        """
        for aug in self.augmentations.values():
            image = aug(image)
        return image

    def get_augmentations(self) -> List:
        """
        Получение списка всех аугментаций в пайплайне.
        :return: Список всех аугментаций в пайплайне.
        """
        return list(self.augmentations.keys())
