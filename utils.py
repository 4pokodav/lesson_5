import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms

def show_images(images, labels=None, nrow=8, title=None, size=128):
    """Визуализирует батч изображений."""
    images = images[:nrow]
    
    # Увеличиваем изображения до 128x128 для лучшей видимости
    resize_transform = transforms.Resize((size, size), antialias=True)
    images_resized = [resize_transform(img) for img in images]
    
    # Создаем сетку изображений
    fig, axes = plt.subplots(1, nrow, figsize=(nrow*2, 2))
    if nrow == 1:
        axes = [axes]
    
    for i, img in enumerate(images_resized):
        img_np = img.numpy().transpose(1, 2, 0)
        # Нормализуем для отображения
        img_np = np.clip(img_np, 0, 1)
        axes[i].imshow(img_np)
        axes[i].axis('off')
        if labels is not None:
            axes[i].set_title(f'Label: {labels[i]}')
    
    if title:
        fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()

def show_single_augmentation(original_img, augmented_img, title="Аугментация", save_path=None, file_name=''):
    """Визуализирует оригинальное и аугментированное изображение рядом."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    resize_transform = transforms.Resize((128, 128), antialias=True)
    to_tensor = transforms.ToTensor()

    def prepare_image(img):
        # Resize
        img = resize_transform(img)
        # Convert to tensor if not already
        if not isinstance(img, torch.Tensor):
            img = to_tensor(img)
        # Convert to numpy for plotting
        img_np = img.numpy().transpose(1, 2, 0)
        return np.clip(img_np, 0, 1)

    orig_np = prepare_image(original_img)
    aug_np = prepare_image(augmented_img)

    ax1.imshow(orig_np)
    ax1.set_title("Оригинал")
    ax1.axis('off')

    ax2.imshow(aug_np)
    ax2.set_title(title)
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig(f'{save_path}{file_name}')
    plt.show()

def show_multiple_augmentations(original_img, augmented_imgs, titles, save_path=None, file_name=''):
    """Визуализирует оригинальное изображение и несколько аугментаций."""
    n_augs = len(augmented_imgs)
    fig, axes = plt.subplots(1, n_augs + 1, figsize=((n_augs + 1) * 2, 2))

    resize_transform = transforms.Resize((128, 128), antialias=True)
    to_tensor = transforms.ToTensor()

    def prepare_image(img):
        img = resize_transform(img)
        if not isinstance(img, torch.Tensor):
            img = to_tensor(img)
        img_np = img.numpy().transpose(1, 2, 0)
        return np.clip(img_np, 0, 1)

    # Оригинал
    orig_np = prepare_image(original_img)
    axes[0].imshow(orig_np)
    axes[0].set_title("Оригинал")
    axes[0].axis('off')

    # Аугментированные изображения
    for i, (aug_img, title) in enumerate(zip(augmented_imgs, titles)):
        aug_np = prepare_image(aug_img)
        axes[i + 1].imshow(aug_np)
        axes[i + 1].set_title(title)
        axes[i + 1].axis('off')

    plt.tight_layout()
    plt.savefig(f'{save_path}{file_name}')
    plt.show()