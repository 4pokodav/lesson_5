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

def build_custom_pipeline():
    '''
    Создает и возвращает кастомный пайплайн с конкретным набором аугментаций
    '''
    pipeline = AugmentationPipeline()
    pipeline.add_augmentation("resize", transforms.Resize((224, 224)))
    pipeline.add_augmentation("random_blur", RandomBlur(p=0.8))
    pipeline.add_augmentation("random_perspective", RandomPerspective(p=0.7))
    pipeline.add_augmentation("to_tensor", transforms.ToTensor())
    pipeline.add_augmentation("gaussian_noise", AddGaussianNoise(std=0.05))
    return pipeline

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


if __name__ == "__main__":
    train_dir = "homework_5/data/train"
    dataset = CustomImageDataset(root_dir=train_dir)

    pipeline = build_custom_pipeline()

    save_dir = "homework_5/output_augmented/custom_pipeline"
    save_augmented_images(pipeline, dataset, save_dir, num_images=50)

    print("Аугментированные изображения успешно сохранены.")