import matplotlib.pyplot as plt
from torchvision import transforms

from utils.datasets import CustomImageDataset

augmentations = {
    'RandomHorizontalFlip': transforms.RandomHorizontalFlip(p=1.0),
    'RandomCrop': transforms.RandomCrop(200),
    'ColorJitter': transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
    'RandomRotation': transforms.RandomRotation(degrees=45),
    'RandomGrayscale': transforms.RandomGrayscale(p=1.0),
}

all_augment = transforms.Compose(list(augmentations.values()))

train = CustomImageDataset('data/train')
class_names = train.get_class_names()

selected_indices = [3, 30, 63, 92, 121]

n_rows = len(selected_indices)
n_cols = len(augmentations) + 2
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))

for row, idx in enumerate(selected_indices):
    img, label = train[idx]

    axes[row, 0].imshow(img)
    axes[row, 0].set_title(f'Original\n{class_names[label]}')
    axes[row, 0].axis('off')

    for col, (name, aug) in enumerate(augmentations.items(), start=1):
        aug_img = aug(img)
        axes[row, col].imshow(aug_img)
        axes[row, col].set_title(name)
        axes[row, col].axis('off')

    combined_img = all_augment(img)
    axes[row, -1].imshow(combined_img)
    axes[row, -1].set_title('All Combined')
    axes[row, -1].axis('off')

plt.tight_layout()
plt.savefig('results/stand_augs/stand_augs.png')
