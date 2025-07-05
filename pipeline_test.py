import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import ToPILImage

from utils.custom_augs import RandomPerspective
from utils.custom_pipeline import AugmentationPipeline
from utils.datasets import CustomImageDataset
from utils.extra_augs import RandomErasing, Posterize, CutOut, Solarize, AutoContrast
from utils.pipeline_test import select_indices

to_pil = ToPILImage()
train = CustomImageDataset(root_dir='data/train')

light_pipeline = AugmentationPipeline()
light_pipeline.add_augmentation("random_erasing", RandomErasing(p=1))

medium_pipeline = AugmentationPipeline()
medium_pipeline.add_augmentation("random_erasing", RandomErasing(p=1))
medium_pipeline.add_augmentation("color_jitter", Posterize())

heavy_pipeline = AugmentationPipeline()
heavy_pipeline.add_augmentation("cut_out", CutOut())
heavy_pipeline.add_augmentation("random_perspective", RandomPerspective())
heavy_pipeline.add_augmentation("solarize", Solarize())
heavy_pipeline.add_augmentation("auto_contrast", AutoContrast())

pipelines = {
    "light": (light_pipeline, 2),
    "medium": (medium_pipeline, 3),
    "heavy": (heavy_pipeline, 5)
}

for name, (pipeline, num_images) in pipelines.items():
    fig, axes = plt.subplots(
        nrows=num_images,
        ncols=2,
        figsize=(8, 4 * num_images)
    )
    fig.suptitle(f"{name.capitalize()} pipeline", fontsize=16)

    indices = select_indices(num_images)
    if num_images == 1:
        axes = axes[np.newaxis, :]

    for row_idx, idx in enumerate(indices):
        img, _ = train[idx]
        img_pil = img if isinstance(img, Image.Image) else to_pil(img)

        aug_img = pipeline.apply(img_pil)

        ax_orig = axes[row_idx, 0]
        ax_orig.imshow(img_pil)
        ax_orig.set_title(f"Original (idx {idx})")
        ax_orig.axis('off')

        ax_aug = axes[row_idx, 1]
        ax_aug.imshow(aug_img)
        ax_aug.set_title("Augmented")
        ax_aug.axis('off')

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(f"results/pipeline_tests/pipeline_{name}.png")
