from datasets import CustomImageDataset
from collections import Counter
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os

def plot_class_diagram(class_counts: dict, save_path=None, file_name=''):
    plt.figure(figsize=(8, 4))
    plt.bar(class_counts.keys(), class_counts.values(), color='skyblue')
    plt.xticks(rotation=45)
    plt.title("Количество изображений по классам (все папки)")
    plt.ylabel("Количество")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, file_name))
    plt.close()

def plot_size_diagram(widths, heights, save_path=None, file_name=''):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(widths, bins=20, color='lightgreen', edgecolor='black')
    plt.title("Распределение ширины изображений (все папки)")
    plt.xlabel("Ширина (px)")
    plt.ylabel("Количество")

    plt.subplot(1, 2, 2)
    plt.hist(heights, bins=20, color='salmon', edgecolor='black')
    plt.title("Распределение высоты изображений (все папки)")
    plt.xlabel("Высота (px)")
    plt.ylabel("Количество")

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, file_name))
    plt.close()

def analyze_folders(folders, save_path='homework_5/results/'):
    all_labels = []
    all_widths = []
    all_heights = []

    os.makedirs(save_path, exist_ok=True)

    for folder in folders:
        print(f"Загружаем изображения из: {folder}")
        dataset = CustomImageDataset(root_dir=folder)
        all_labels.extend(dataset.labels)

        for img_path in dataset.images:
            img = Image.open(img_path)
            w, h = img.size
            all_widths.append(w)
            all_heights.append(h)

    widths = np.array(all_widths)
    heights = np.array(all_heights)

    # Статистика
    min_size = (int(widths.min()), int(heights.min()))
    max_size = (int(widths.max()), int(heights.max()))
    mean_size = (int(widths.mean()), int(heights.mean()))

    print(f"\nОбщая статистика по изображениям:")
    print(f"Минимальный размер изображения: {min_size}")
    print(f"Максимальный размер изображения: {max_size}")
    print(f"Средний размер изображения: {mean_size}")

    # Подсчёт по классам
    dataset_for_labels = CustomImageDataset(root_dir=folders[0])  # Используем первую папку для получения имён классов
    class_names = dataset_for_labels.get_class_names()
    label_counts = Counter(all_labels)
    class_counts = {class_names[k]: v for k, v in label_counts.items()}

    plot_class_diagram(class_counts, save_path, file_name='class_diagram_all.png')
    plot_size_diagram(widths, heights, save_path, file_name='size_diagram_all.png')


if __name__ == '__main__':
    folders = [
        "homework_5/data/train",
        "homework_5/data/val",
        "homework_5/data/test"
    ]
    analyze_folders(folders)