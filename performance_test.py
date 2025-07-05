import os
import time
import tracemalloc
from typing import List, Callable

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from utils.datasets import CustomImageDataset
from utils.extra_augs import ElasticTransform, Posterize, CutOut, Solarize, AutoContrast

sizes = [(64, 64), (128, 128), (224, 224), (512, 512)]
results = []

all_paths = []
for root, _, files in os.walk('data/train'):
    for f in files:
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            all_paths.append(os.path.join(root, f))
all_paths = sorted(all_paths)[:100]

augs = [
    ElasticTransform(p=1.0, alpha=3.0, sigma=50),
    Posterize(bits=4),
    CutOut(p=1.0, num_holes=1, max_size=0.2),
    Solarize(threshold=128),
    AutoContrast(p=1.0),
]


def _apply_augs(
        img: Image.Image,
        augs_list: List[Callable[[Image.Image], Image.Image]]
) -> Image.Image:
    """
    Применяет аугментации.
    :param img: Изображение на вход.
    :param augs_list: Список аугментаций для применения.
    :return: Обработанное изображение.
    """
    for aug in augs_list:
        img = aug(img)
    return img


for size in sizes:
    dataset = CustomImageDataset(
        root_dir='data/train',
        target_size=size,
        transform=lambda img: _apply_augs(img, augs)
    )

    dataset.images = all_paths
    dataset.labels = [0] * len(all_paths)

    tracemalloc.start()
    t0 = time.perf_counter()

    for idx in range(len(dataset)):
        img, _ = dataset[idx]

    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    results.append({
        'Size': f'{size[0]}×{size[1]}',
        'Time (s) for 100 imgs': elapsed,
        'Peak Memory (MB)': peak / (1024 ** 2)
    })


df = pd.DataFrame(results)
df.to_csv('results/performance_test/performance.csv')

plt.figure()
plt.plot(df['Size'], df['Time (s) for 100 imgs'], marker='o')
plt.xlabel('Image Size')
plt.ylabel('Time (s) for 100 images')
plt.title('Processing Time vs Image Size')
plt.tight_layout()
plt.savefig('results/performance_test/time.png')

plt.figure()
plt.plot(df['Size'], df['Peak Memory (MB)'], marker='o')
plt.xlabel('Image Size')
plt.ylabel('Peak Memory (MB)')
plt.title('Peak Memory Usage vs Image Size')
plt.tight_layout()
plt.savefig('results/performance_test/memory.png')
