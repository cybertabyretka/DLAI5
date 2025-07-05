import random
from typing import List


def select_indices(num_images: int) -> List[int]:
    """
    Выбирает num_images разных персонажей, возвращает случайный индекс внутри каждого блока персонажа.
    :param num_images: Количество изображений, которые необходимо выполнить.
    :return: Индексы выбранных изображений.
    """
    indices = []
    used_blocks = set()

    while len(indices) < num_images:
        block_id = random.randint(0, 5)
        if block_id in used_blocks:
            continue
        used_blocks.add(block_id)

        start_idx = block_id * 30
        end_idx = start_idx + 30 - 1
        idx = random.randint(start_idx, end_idx)
        indices.append(idx)

    return indices
