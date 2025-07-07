import time
import psutil
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from datasets import CustomImageDataset
from custom_augs import RandomBlur, RandomPerspective
from extra_augs import AddGaussianNoise
from tqdm import tqdm

SIZES = [64, 128, 224, 512]
NUM_IMAGES = 100
TRAIN_DIR = "homework_5/data/train"

def build_pipeline(target_size):
    return transforms.Compose([
        transforms.Resize((target_size, target_size)),
        RandomBlur(p=0.7),
        RandomPerspective(p=0.7),
        transforms.ToTensor(),
        AddGaussianNoise(std=0.05)
    ])

def measure_pipeline(dataset, transform, num_images):
    times = []
    mem_usages = []
    process = psutil.Process()

    for i in tqdm(range(num_images), desc="Processing"):
        img_path = dataset.images[i]
        img = Image.open(img_path).convert('RGB')

        start_time = time.time()
        mem_before = process.memory_info().rss / (1024 ** 2)  # in MB

        _ = transform(img)

        mem_after = process.memory_info().rss / (1024 ** 2)
        end_time = time.time()

        times.append(end_time - start_time)
        mem_usages.append(mem_after - mem_before)

    return sum(times) / len(times), sum(mem_usages) / len(mem_usages)


if __name__ == "__main__":
    dataset = CustomImageDataset(root_dir=TRAIN_DIR, transform=None)

    avg_times = []
    avg_mems = []

    for size in SIZES:
        print(f"\nОбработка размера: {size}x{size}")
        transform = build_pipeline(size)
        avg_time, avg_mem = measure_pipeline(dataset, transform, NUM_IMAGES)
        avg_times.append(avg_time)
        avg_mems.append(avg_mem)

    # График: Время
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(SIZES, avg_times, marker='o', color='blue')
    plt.title("Время обработки vs Размер")
    plt.xlabel("Размер изображения (px)")
    plt.ylabel("Время (сек)")

    # График: Память
    plt.subplot(1, 2, 2)
    plt.plot(SIZES, avg_mems, marker='o', color='red')
    plt.title("Потребление памяти vs Размер")
    plt.xlabel("Размер изображения (px)")
    plt.ylabel("Память (MB)")

    plt.tight_layout()
    plt.savefig("homework_5/results/size_vs_time_memory.png")
    plt.show()

    print("Эксперимент завершён.")