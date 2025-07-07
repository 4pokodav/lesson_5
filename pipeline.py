import os
from torchvision import transforms
from datasets import CustomImageDataset
from utils import show_multiple_augmentations
from extra_augs import (
    AddGaussianNoise,
    RandomErasingCustom,
    CutOut,
    Solarize,
    Posterize,
    AutoContrast,
    ElasticTransform
)
import torch
from PIL import Image

def load_sample_images(dataset, num_samples, image_size):
    """Выбирает по одному изображению из каждого класса"""
    class_to_idx = {}
    images = []
    labels = []

    for idx in range(len(dataset)):
        _, label = dataset[idx]
        if label not in class_to_idx:
            class_to_idx[label] = idx
            images.append(dataset.images[idx])
            labels.append(label)
        if len(images) == num_samples:
            break

    return images, labels

def get_custom_transforms():
    """Возвращает список аугментаций из extra_augs"""
    return [
        AddGaussianNoise(std=0.1),
        RandomErasingCustom(p=1.0),
        CutOut(p=1.0),
        Solarize(threshold=128),
        Posterize(bits=4),
        AutoContrast(p=1.0),
        ElasticTransform(p=1.0)
    ], [
        "GaussianNoise",
        "RandomErasing",
        "CutOut",
        "Solarize",
        "Posterize",
        "AutoContrast",
        "ElasticTransform"
    ]

def apply_augmentations_and_save(images, labels, image_size, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    transforms_list, titles = get_custom_transforms()

    for idx, (img_path, label) in enumerate(zip(images, labels)):
        img = Image.open(img_path).convert('RGB').resize(image_size)
        img_tensor = transforms.ToTensor()(img)

        aug_tensors = [t(img_tensor.clone()) for t in transforms_list]
        aug_pils = [transforms.ToPILImage()(tensor) for tensor in aug_tensors]

        print(f"Label {label}, image: {os.path.basename(img_path)}")

        filename = f"pipeline_label{label}_img{idx}.png"
        show_multiple_augmentations(
            img,
            aug_pils,
            titles,
            save_path=save_dir,
            file_name=filename
        )

def main():
    root_dir = 'homework_5/data/train'
    num_classes_to_sample = 5
    image_size = (224, 224)
    save_dir = 'homework_5/results/'

    dataset = CustomImageDataset(root_dir=root_dir, transform=None, target_size=image_size)
    images, labels = load_sample_images(dataset, num_classes_to_sample, image_size)
    apply_augmentations_and_save(images, labels, image_size, save_dir)


if __name__ == "__main__":
    main()