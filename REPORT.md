# Домашнее задание к уроку 5: Аугментации и работа с изображениями

## Задание 1: Стандартные аугментации torchvision (15 баллов)
Создал пайплайн со стандартными аугментациями. 
Результаты пайплайна:

![Original augmentation](https://github.com/4pokodav/lesson_5/raw/main/results/pipeline_label0_img0.png)
![Original augmentation](https://github.com/4pokodav/lesson_5/raw/main/results/pipeline_label1_img1.png)
![Original augmentation](https://github.com/4pokodav/lesson_5/raw/main/results/pipeline_label2_img2.png)
![Original augmentation](https://github.com/4pokodav/lesson_5/raw/main/results/pipeline_label3_img3.png)
![Original augmentation](https://github.com/4pokodav/lesson_5/raw/main/results/pipeline_label4_img4.png)

Все готовые аугментации успешно применяются.

## Задание 2: Кастомные аугментации (20 баллов)
Реализовал 3 кастомные аугментации: RandomBlur (случайное размытие изображения), RandomPerspective (случайное перспективная трансформация), RandomBrightnessContrast (Случайное изменение яркости и контраста).

Готовые аугментации:

![Original augmentations results](https://github.com/4pokodav/lesson_5/raw/main/results/original_augs.png)

Кастомные аугментации:

![Custom augmentations results](https://github.com/4pokodav/lesson_5/raw/main/results/custom_augs.png)

Кастомные аугментации успешно применяются, способы аугментации отличаются, но есть схожесть между RandomBrightnessContrast и Solarize, но RandomBrightnessContrast случайно изменяет яркость и контраст, а Solarize просто инвертирует пиксели выше определенного порога яркости.

## Задание 3: Анализ датасета (10 баллов)
Провел анализ датасета: подсчитал количество изображений в каждом классе, нашел минимальный, средний и максимальные размеры изображений.

**Общая статистика по изображениям:**
Минимальный размер изображения: (210, 220)  
Максимальный размер изображения: (736, 1308)
Средний размер изображения: (545, 629)

Распределение изображений по классам:
![Class diagram](https://github.com/4pokodav/lesson_5/raw/main/results/class_diagram_all.png)

Распределение размеров изображений:
![Size diagram](https://github.com/4pokodav/lesson_5/raw/main/results/size_diagram_all.png)

В каждом классе содержится одинаковое количество изображений. Изображения представлены разных размеров, но преобладают изображения размером 545x629.

## Задание 4: Pipeline аугментаций (20 баллов)

Создал пайплайн аугментаций, создал несколько конфигураций.
Pipeline |  Аугментации
light    |	Resize, HorizontalFlip(0.3)
medium   |	Resize, HorizontalFlip(0.5), ColorJitter, RandomBlur(0.5)
heavy    |	Resize, HorizontalFlip(0.7), RandomPerspective, RandomBlur(0.8), ColorJitter+, GaussianNoise

Результаты применения конфигураций:

**1) light**

![Light configure](https://github.com/4pokodav/lesson_5/raw/main/output_augmented/light/Гароу/aug_0.png) 
![Light configure](https://github.com/4pokodav/lesson_5/raw/main/output_augmented/light/Гароу/aug_1.png) 
![Light configure](https://github.com/4pokodav/lesson_5/raw/main/output_augmented/light/Гароу/aug_2.png)
![Light configure](https://github.com/4pokodav/lesson_5/raw/main/output_augmented/light/Гароу/aug_5.png)
![Light configure](https://github.com/4pokodav/lesson_5/raw/main/output_augmented/light/Генос/aug_30.png) 
![Light configure](https://github.com/4pokodav/lesson_5/raw/main/output_augmented/light/Генос/aug_31.png) 
![Light configure](https://github.com/4pokodav/lesson_5/raw/main/output_augmented/light/Генос/aug_32.png)
![Light configure](https://github.com/4pokodav/lesson_5/raw/main/output_augmented/light/Генос/aug_35.png)


**2) medium**

![Medium configure](https://github.com/4pokodav/lesson_5/raw/main/output_augmented/medium/Гароу/aug_0.png) 
![Medium configure](https://github.com/4pokodav/lesson_5/raw/main/output_augmented/medium/Гароу/aug_1.png) 
![Medium configure](https://github.com/4pokodav/lesson_5/raw/main/output_augmented/medium/Гароу/aug_2.png)
![Medium configure](https://github.com/4pokodav/lesson_5/raw/main/output_augmented/medium/Гароу/aug_5.png)
![Medium configure](https://github.com/4pokodav/lesson_5/raw/main/output_augmented/medium/Генос/aug_30.png) 
![Medium configure](https://github.com/4pokodav/lesson_5/raw/main/output_augmented/medium/Генос/aug_31.png) 
![Medium configure](https://github.com/4pokodav/lesson_5/raw/main/output_augmented/medium/Генос/aug_32.png)
![Medium configure](https://github.com/4pokodav/lesson_5/raw/main/output_augmented/medium/Генос/aug_35.png)


**3) heavy**

![Heavy configure](https://github.com/4pokodav/lesson_5/raw/main/output_augmented/heavy/Гароу/aug_0.png) 
![Heavy configure](https://github.com/4pokodav/lesson_5/raw/main/output_augmented/heavy/Гароу/aug_1.png) 
![Heavy configure](https://github.com/4pokodav/lesson_5/raw/main/output_augmented/heavy/Гароу/aug_2.png)
![Heavy configure](https://github.com/4pokodav/lesson_5/raw/main/output_augmented/heavy/Гароу/aug_5.png)
![Heavy configure](https://github.com/4pokodav/lesson_5/raw/main/output_augmented/heavy/Генос/aug_30.png) 
![Heavy configure](https://github.com/4pokodav/lesson_5/raw/main/output_augmented/heavy/Генос/aug_31.png) 
![Heavy configure](https://github.com/4pokodav/lesson_5/raw/main/output_augmented/heavy/Генос/aug_32.png)
![Heavy configure](https://github.com/4pokodav/lesson_5/raw/main/output_augmented/heavy/Генос/aug_35.png)


Все конфигурации пайплайнов успешно исполняются. Чем "сильнее" конфигурация, тем сильнее изменяется изображение.

## Задание 5: Эксперимент с размерами (10 баллов)

Провел эксперименты с разными размерами изображений (64x64, 128x128, 224x224, 512x512). Для каждого размера измерил время загрузки и применения аугментаций к 100 изображениям, а также потребление памяти.

![Augmentation](https://github.com/4pokodav/lesson_5/raw/main/results/size_vs_time_memory.png)

Прослеживается очевидная зависимость времени обработки и потребления памяти от размера изображения. Чем больше размер изображения, тем больше потребление памяти и время обработки.

## Задание 6: Дообучение предобученных моделей (25 баллов)

Дообучил модель efficientnet, проверил точность.

![Augmentation](https://github.com/4pokodav/lesson_5/raw/main/results/training_plot.png)

Удалось достичь хорошей точности модели (0.994), но после 4 эпохи началось переобучение.
