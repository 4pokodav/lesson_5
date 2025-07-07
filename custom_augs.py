import random
import torch
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import torchvision.transforms.functional as F
from torchvision.transforms import RandomPerspective as TorchRandomPerspective

class RandomBlur:
    '''
    Применяет случайное размытие изображения с заданной вероятностью и радиусом.
    '''
    def __init__(self, p=0.5, radius_range=(1, 3)):
        self.p = p
        self.radius_range = radius_range

    def __call__(self, img):
        if random.random() > self.p:
            return img
        radius = random.uniform(*self.radius_range)
        return img.filter(ImageFilter.GaussianBlur(radius))

class RandomPerspective:
    '''
    Применяет случайную перспективную трансформацию изображения с заданной вероятностью.
    '''
    def __init__(self, p=0.5, distortion_scale=0.5):
        self.p = p
        self.distortion_scale = distortion_scale
        self.transform = TorchRandomPerspective(distortion_scale=distortion_scale, p=1.0)

    def __call__(self, img):
        if random.random() > self.p:
            return img
        return self.transform(img)

class RandomBrightnessContrast:
    '''
    Случайным образом изменяет яркость и контраст изображения с заданной вероятностью.
    '''
    def __init__(self, p=0.5, brightness=0.5, contrast=0.5):
        self.p = p
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, img):
        if random.random() > self.p:
            return img
        brightness_factor = random.uniform(1 - self.brightness, 1 + self.brightness)
        contrast_factor = random.uniform(1 - self.contrast, 1 + self.contrast)
        img = ImageEnhance.Brightness(img).enhance(brightness_factor)
        img = ImageEnhance.Contrast(img).enhance(contrast_factor)
        return img