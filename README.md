# README: Трекинг муравьев с обнаружением встреч

Этот проект выполнен на учебном курсе "Обработка цифровых изображений" и представляет собой систему для трекинга муравьев на видео с возможностью обнаружения их встреч. Код написан на Python с использованием библиотек OpenCV и NumPy. Основная цель проекта — отслеживать перемещение муравьев, определять их встречи и сохранять результаты в CSV-файл.

---

## Основные функции

1. **Трекинг муравьев**:
   - Используется алгоритм трекинга `CSRT` из OpenCV.
   - Каждому муравью присваивается уникальный ID.
   - Трекеры обновляются на каждом кадре видео.

2. **Обнаружение встреч**:
   - Если два муравья пересекаются (их bounding box'ы пересекаются), информация о встрече сохраняется в файл `ant_meetings.csv`.
   - Встречи определяются на основе расстояния между центрами bounding box'ов и их пересечения.

3. **Удаление статичных трекеров**:
   - Если муравей долго стоит на месте (внутри заданного круга), его трекер удаляется.

4. **Визуализация**:
   - На видео отображаются bounding box'ы муравьев, их ID, а также круг, внутри которого удаляются статичные трекеры.

5. **Сохранение результатов**:
   - Траектории муравьев сохраняются в файл `ant_tracking.csv`.
   - Встречи муравьев сохраняются в файл `ant_meetings.csv`.

---

## Требования

Для запуска кода необходимо установить следующие библиотеки:

```bash
pip install opencv-python numpy
```

Видео, использованное при работе, под которые подстроены все настройки, можно найти по ссылке

`https://drive.google.com/file/d/13pu38VXQlUCoZK4I-rwebVcPFw7Juidc/view?usp=sharing`
---

## Конфигурация

Параметры программы настраиваются через файл `config.txt`. Пример содержимого:

```ini
[VideoProcessing]
; Начальный кадр для обработки (0 - начать с первого кадра)
FRAME_START = 0

; Конечный кадр для обработки (inf - обрабатывать до конца видео)
FRAME_END = inf


[AntColorsCount]
; Порог для определения "темных" пикселей (значение от 0 до 255)
COLOR_THRESHOLD = 55


[AntDetection]
; Минимальный периметр контура для обнаружения муравья
MIN_PERIMETER = 10

; Максимальный периметр контура для обнаружения муравья
MAX_PERIMETER = 200

; Точность аппроксимации контура (меньше значение - точнее аппроксимация)
APPROX_EPSILON = 0.04

; Минимальное количество вершин у аппроксимированного контура
MIN_VERTICES = 5

; Максимальное количество вершин у аппроксимированного контура
MAX_VERTICES = 9

; Минимальное количество "темных" пикселей для обнаружения муравья
MIN_DARK_COUNT = 10

; Максимальное количество "темных" пикселей для обнаружения муравья
MAX_DARK_COUNT = 100

; Максимальный процент "темных" пикселей в bounding box'е
MAX_DARK_PERCENT = 50

; Минимальная ширина bounding box'а для обнаружения муравья
MIN_WIDTH = 10

; Максимальная ширина bounding box'а для обнаружения муравья
MAX_WIDTH = 50

; Минимальная высота bounding box'а для обнаружения муравья
MIN_HEIGHT = 10

; Максимальная высота bounding box'а для обнаружения муравья
MAX_HEIGHT = 50


[TrackerSettings]
; Максимальное количество пропущенных кадров для удаления трекера
MAX_MISSED_FRAMES = 25

; Максимальное количество статичных кадров для удаления трекера
MAX_STATIC_FRAMES = 25


; Максимальное расстояние между центрами bounding box'ов для обнаружения встречи
MEETING_DISTANCE = 50


[FramePreprocessing]
; Размер ядра для размытия (формат: ширина, высота)
BLUR_KERNEL_SIZE = 3,3

; Пороговое значение для бинаризации (значение от 0 до 255)
THRESHOLD_VALUE = 100

; Максимальное значение для бинаризации (обычно 255)
MAX_THRESHOLD = 255


[Circle]
; Координата X центра круга для удаления статичных трекеров
CIRCLE_CENTER_X = 785

; Координата Y центра круга для удаления статичных трекеров
CIRCLE_CENTER_Y = 500

; Радиус круга для удаления статичных трекеров
CIRCLE_RADIUS = 180
```

---

## Запуск программы

### Аргументы командной строки:

- `-v` или `--video`: Путь к входному видеофайлу.
- `-o` или `--output`: Имя выходного видеофайла (по умолчанию `out.mp4`).
- `-s` или `--show`: Показывать видео во время обработки (опционально).

### Показ видео в реальном времени

Для того, чтобы закрыть видео, нажмите Escape

---

## Выходные данные

1. **`ant_tracking.csv`**:
   - Содержит траектории муравьев в формате:
     ```
     Frame,ID,X,Y
     100,1,320,240
     101,1,322,242
     ```

2. **`ant_meetings.csv`**:
   - Содержит информацию о встречах муравьев в формате:
     ```
     Frame,Ant1_ID,Ant2_ID
     100,1,2
     150,3,4
     ```

3. **Выходное видео**:
   - Видео с отрисованными bounding box'ами, ID муравьев и кругом для удаления статичных трекеров.



## Пример использования

1. Запустите программу с видеофайлом:

   ```bash
   python main.py -v input_video.mkv -o output.mp4 --show
   ```

2. После завершения обработки:
   - Проверьте файлы `ant_tracking.csv` и `ant_meetings.csv`.
   - Откройте выходное видео `output.mp4` для визуальной проверки.

---

## Возможные улучшения

1. **Оптимизация производительности**:
   - Использование более эффективных алгоритмов трекинга.
   - Параллельная обработка кадров.

2. **Расширение функционала**:
   - Добавление анализа поведения муравьев (например, скорость, направление движения).

3. **Улучшение точности**:
   - Использование машинного обучения для более точного обнаружения муравьев.

---
