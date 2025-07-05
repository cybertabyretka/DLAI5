import matplotlib.pyplot as plt
from torchvision import transforms

from utils.custom_augs import RandomBlur, RandomPerspective, RandomBrightnessContrast
from utils.datasets import CustomImageDataset
from utils.extra_augs import AddGaussianNoise, CutOut, ElasticTransform

train = CustomImageDataset('data/train')
class_names = train.get_class_names()

selected_indices = [3, 30, 63, 92, 120]

custom_augs = {
    'RandomBlur': RandomBlur(p=1.0),
    'RandomPerspective': RandomPerspective(p=1.0, distortion_scale=0.25),
    'RandomBrContrast': RandomBrightnessContrast(p=1.0),
}
extra_augs = {
    'AddGaussianNoise': AddGaussianNoise(mean=0., std=0.2),
    'CutOut': CutOut(p=1.0),
    'ElasticTransform': ElasticTransform(p=1.0, alpha=5, sigma=50),
}

n = len(selected_indices)
total_cols = 1 + len(custom_augs) + len(extra_augs)
fig, axes = plt.subplots(n, total_cols, figsize=(4 * total_cols, 4 * n))

for row, idx in enumerate(selected_indices):
    img, lbl = train[idx]
    axes[row, 0].imshow(img)
    axes[row, 0].set_title(f'Original\n{class_names[lbl]}')
    axes[row, 0].axis('off')

    for col, (name, aug) in enumerate(custom_augs.items(), start=1):
        axes[row, col].imshow(aug(img))
        axes[row, col].set_title(name)
        axes[row, col].axis('off')

    offset = 1 + len(custom_augs)
    for i, (name, aug) in enumerate(extra_augs.items()):
        tensor = transforms.ToTensor()(img)
        out_tensor = aug(tensor)
        out_pil = transforms.ToPILImage()(out_tensor.clamp(0, 1))
        axes[row, offset + i].imshow(out_pil)
        axes[row, offset + i].set_title(name)
        axes[row, offset + i].axis('off')

plt.tight_layout()
plt.savefig('results/custom_augs/custom_augs.png')
