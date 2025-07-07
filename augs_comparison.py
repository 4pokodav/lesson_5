import os
from datasets import CustomImageDataset
from custom_augs import RandomBlur, RandomPerspective, RandomBrightnessContrast
from extra_augs import AddGaussianNoise, RandomErasingCustom, Solarize
from utils import show_multiple_augmentations
from torchvision import transforms
from PIL import Image

root_dir = 'homework_5/data/train'
image_size = (224, 224)

dataset = CustomImageDataset(root_dir=root_dir, transform=None, target_size=image_size)

sample_path = dataset.images[0]
sample_img = Image.open(sample_path).convert('RGB').resize(image_size)

# Аугментации
blur = RandomBlur(p=1.0)
perspective = RandomPerspective(p=1.0)
brightness_contrast = RandomBrightnessContrast(p=1.0)

custom_aug_imgs = [
    blur(sample_img),
    perspective(sample_img),
    brightness_contrast(sample_img)
]
custom_titles = ["RandomBlur", "RandomPerspective", "BrightnessContrast"]

# Готовые аугментации
to_tensor = transforms.ToTensor()
sample_tensor = to_tensor(sample_img)

noise = AddGaussianNoise(std=0.1)
erasing = RandomErasingCustom(p=1.0)
solarize = Solarize(threshold=128)

extra_aug_imgs = [
    transforms.ToPILImage()(noise(sample_tensor.clone())),
    transforms.ToPILImage()(erasing(sample_tensor.clone())),
    transforms.ToPILImage()(solarize(sample_tensor.clone()))
]
extra_titles = ["GaussianNoise", "RandomErasing", "Solarize"]

# Визуализация
print("Кастомные аугментации:")
show_multiple_augmentations(sample_img, custom_aug_imgs, custom_titles, save_path='homework_5/results/', file_name='custom_augs.png')

print("Готовые аугментации из extra_augs.py:")
show_multiple_augmentations(sample_img, extra_aug_imgs, extra_titles, save_path='homework_5/results/', file_name='original_augs.png')