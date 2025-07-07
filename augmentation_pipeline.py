import os
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from collections import OrderedDict
from tqdm import tqdm

from custom_augs import RandomBlur, RandomPerspective
from extra_augs import AddGaussianNoise
from datasets import CustomImageDataset

class AugmentationPipeline:
    '''
    Класс для создания и управления пайплайном аугментаций изображений.
    '''
    def __init__(self):
        self.augmentations = OrderedDict()

    def add_augmentation(self, name, aug):
        self.augmentations[name] = aug

    def remove_augmentation(self, name):
        if name in self.augmentations:
            del self.augmentations[name]

    def get_augmentations(self):
        return list(self.augmentations.keys())

    def apply(self, image):
        transform = transforms.Compose(list(self.augmentations.values()))
        return transform(image)

def save_augmented_images(pipeline, dataset, save_dir, num_images=50):
    '''
    Применяет пайплайн к изображениям из датасета и сохраняет аугментированные изображения в указанную директорию.
    '''
    os.makedirs(save_dir, exist_ok=True)
    for i in tqdm(range(min(num_images, len(dataset))), desc=f"Сохраняем в {save_dir}"):
        img_path = dataset.images[i]
        label = dataset.labels[i]
        image = Image.open(img_path).convert('RGB')
        aug_image = pipeline.apply(image)

        class_name = dataset.get_class_names()[label]
        class_dir = os.path.join(save_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        save_path = os.path.join(class_dir, f"aug_{i}.png")
        save_image(aug_image, save_path)

# Конфигурации аугментаций
def build_light_pipeline():
    pipeline = AugmentationPipeline()
    pipeline.add_augmentation("resize", transforms.Resize((224, 224)))
    pipeline.add_augmentation("horizontal_flip", transforms.RandomHorizontalFlip(p=0.3))
    pipeline.add_augmentation("to_tensor", transforms.ToTensor())
    
    return pipeline

def build_medium_pipeline():
    pipeline = AugmentationPipeline()
    pipeline.add_augmentation("resize", transforms.Resize((224, 224)))
    pipeline.add_augmentation("horizontal_flip", transforms.RandomHorizontalFlip(p=0.5))
    pipeline.add_augmentation("color_jitter", transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))
    pipeline.add_augmentation("random_blur", RandomBlur(p=0.5))
    pipeline.add_augmentation("to_tensor", transforms.ToTensor())

    return pipeline

def build_heavy_pipeline():
    pipeline = AugmentationPipeline()
    pipeline.add_augmentation("resize", transforms.Resize((224, 224)))
    pipeline.add_augmentation("horizontal_flip", transforms.RandomHorizontalFlip(p=0.7))
    pipeline.add_augmentation("random_perspective", RandomPerspective(p=0.7))
    pipeline.add_augmentation("random_blur", RandomBlur(p=0.8))
    pipeline.add_augmentation("color_jitter", transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1))
    pipeline.add_augmentation("to_tensor", transforms.ToTensor()) 
    pipeline.add_augmentation("gaussian_noise", AddGaussianNoise(std=0.05)) 

    return pipeline


if __name__ == "__main__":
    train_dir = "homework_5/data/train"
    dataset = CustomImageDataset(root_dir=train_dir)

    # Light pipeline
    light_pipeline = build_light_pipeline()
    save_augmented_images(light_pipeline, dataset, "homework_5/output_augmented/light", num_images=50)

    # Medium pipeline
    medium_pipeline = build_medium_pipeline()
    save_augmented_images(medium_pipeline, dataset, "homework_5/output_augmented/medium", num_images=50)

    # Heavy pipeline
    heavy_pipeline = build_heavy_pipeline()
    save_augmented_images(heavy_pipeline, dataset, "homework_5/output_augmented/heavy", num_images=50)

    print("Все аугментированные изображения успешно сохранены.")