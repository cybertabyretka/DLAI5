import os

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

root_dir = 'data/train'

records = []
for class_name in sorted(os.listdir(root_dir)):
    class_dir = os.path.join(root_dir, class_name)
    if not os.path.isdir(class_dir):
        continue
    for fname in os.listdir(class_dir):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            continue
        path = os.path.join(class_dir, fname)
        with Image.open(path) as img:
            w, h = img.size
        records.append({
            'class': class_name,
            'width': w,
            'height': h,
            'area': w * h
        })

df = pd.DataFrame(records)

counts = df['class'].value_counts().sort_index()

stats = df[['width', 'height', 'area']].agg(['min', 'max', 'mean']).round(2)

print("=== Количество изображений в классах ===")
print(counts.to_string())
print("\n=== Статистика по width/height/area ===")
print(stats)

plt.figure(figsize=(6, 4))
plt.hist(df['area'], bins=30)
plt.title('Распределение площадей изображений')
plt.xlabel('Площадь (px²)')
plt.ylabel('Частота')
plt.tight_layout()
plt.savefig('results/analysis/areas_distribution.png')

plt.figure(figsize=(6, 4))
counts.plot(kind='bar')
plt.title('Число изображений в каждом классе')
plt.xlabel('Класс')
plt.ylabel('Количество')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('results/analysis/images_numbers.png')
