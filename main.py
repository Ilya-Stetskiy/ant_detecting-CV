import argparse
import csv
import cv2
import numpy as np
import configparser
import random
from collections import deque  


# Загрузка конфигурации из файла
config = configparser.ConfigParser()
config.read('config.txt')

# Парсинг констант
FRAME_START = int(config['VideoProcessing']['FRAME_START'])
FRAME_END = float(config['VideoProcessing']['FRAME_END']) if config['VideoProcessing']['FRAME_END'] != 'inf' else float('inf')
FRAME_PRINT_STEP = int(config['VideoProcessing']['FRAME_PRINT_STEP'])

COLOR_THRESHOLD = int(config['AntColorsCount']['COLOR_THRESHOLD'])

MIN_PERIMETER = int(config['AntDetection']['MIN_PERIMETER'])
MAX_PERIMETER = int(config['AntDetection']['MAX_PERIMETER'])
APPROX_EPSILON = float(config['AntDetection']['APPROX_EPSILON'])
MIN_VERTICES = int(config['AntDetection']['MIN_VERTICES'])
MAX_VERTICES = int(config['AntDetection']['MAX_VERTICES'])
MIN_DARK_COUNT = int(config['AntDetection']['MIN_DARK_COUNT'])
MAX_DARK_COUNT = int(config['AntDetection']['MAX_DARK_COUNT'])
MAX_DARK_PERCENT = int(config['AntDetection']['MAX_DARK_PERCENT'])
MIN_WIDTH = int(config['AntDetection']['MIN_WIDTH'])
MAX_WIDTH = int(config['AntDetection']['MAX_WIDTH'])
MIN_HEIGHT = int(config['AntDetection']['MIN_HEIGHT'])
MAX_HEIGHT = int(config['AntDetection']['MAX_HEIGHT'])
MAX_AREA = int(config['AntDetection']['MAX_AREA'])

MIN_TRACKER_DARK_COUNT = int(config['TrackerSettings']['MIN_TRACKER_DARK_COUNT'])
MAX_TRACKER_DARK_COUNT = int(config['TrackerSettings']['MAX_TRACKER_DARK_COUNT'])
MAX_MISSED_FRAMES = int(config['TrackerSettings']['MAX_MISSED_FRAMES'])
MAX_STATIC_FRAMES = int(config['TrackerSettings']['MAX_STATIC_FRAMES'])
MIN_POINTS_FOR_HISTORY = int(config['TrackerSettings']['MIN_POINTS_FOR_HISTORY'])
MEETING_DISTANCE = int(config['TrackerSettings']['MEETING_DISTANCE'])

BLUR_KERNEL_SIZE = tuple(map(int, config['FramePreprocessing']['BLUR_KERNEL_SIZE'].split(',')))
THRESHOLD_VALUE = int(config['FramePreprocessing']['THRESHOLD_VALUE'])
MAX_THRESHOLD = int(config['FramePreprocessing']['MAX_THRESHOLD'])
CIRCLE_CENTER = (int(config['Circle']['CIRCLE_CENTER_X']), int(config['Circle']['CIRCLE_CENTER_Y']))
CIRCLE_RADIUS = int(config['Circle']['CIRCLE_RADIUS'])

# Глобальный счетчик для ID муравьев
next_id = 1

class TrackerWrapper:
    def __init__(self, tracker, box):
        global next_id
        self.tracker = tracker
        self.box = box
        self.missed_frames = 0
        self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        self.centers = []
        self.id = next_id
        next_id += 1
        self.last_known_position = self._calculate_center(box)
        self.history = deque(maxlen=10)  # История последних позиций
        self.static_frames = 0  # Счетчик кадров без движения

    def update(self, frame, all_centers, frameN):
        success, box = self.tracker.update(frame)
        if success:
            center = self._calculate_center(box)
            self.history.append(center)

            # Проверка на движение
            if len(self.history) > 1:
                last_center = self.history[-2]
                distance = np.linalg.norm(np.array(center) - np.array(last_center))
                if distance < 2:  # Порог для определения движения (в пикселях)
                    self.static_frames += 1
                else:
                    self.static_frames = 0  # Сбрасываем счетчик, если есть движение

            self.box = box
            self.missed_frames = 0
            self.centers.append(center)
            self.last_known_position = center
            all_centers.append((frameN, self.id, center))
        else:
            self.missed_frames += 1
        return success, self.box

    def is_static(self):
        return self.static_frames >= MAX_STATIC_FRAMES

    def is_active(self):
        return self.missed_frames <= MAX_MISSED_FRAMES

    def _calculate_center(self, box):
        x, y, w, h = box
        return (int(x + w / 2), int(y + h / 2))

# сохранение точек перемещения
def save_to_csv(all_centers, filename='ant_tracking.csv'):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Frame', 'ID', 'X', 'Y'])
        for frameN, id, center in all_centers:
            writer.writerow([frameN, id, center[0], center[1]])
            
# сохранение встреч
def save_meeting_to_csv(frameN, ant1_id, ant2_id, filename='ant_meetings.csv'):
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([frameN, ant1_id, ant2_id])
        
# Функция для проверки что точка в кругу
def is_point_inside_circle(point, center, radius):
    return np.linalg.norm(np.array(point) - np.array(center)) <= radius

# Функция для проверки, находится ли точка внутри bounding box
def is_point_inside_box(point, box):
    x, y, w, h = box
    px, py = point
    return (x <= px <= x + w) and (y <= py <= y + h)

#Проверка функции пересечения боксов, для детекта встречи
def do_boxes_intersect(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    return not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1)

# Функция для вычисления площади пересечения двух прямоугольников
def intersectionSquare(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    width = x_right - x_left
    height = y_bottom - y_top
    return width * height if width > 0 and height > 0 else 0

# нахождение центра бокса
def calculate_centers(boxes):
    centers = []
    for box in boxes:
        x, y, w, h = box
        centers.append((x + w / 2, y + h / 2))
    return np.array(centers)

# Функция для подсчета "темных" пикселей
def getAntColorsCount(frame, rect):
    x0, y0, w, h = rect
    if w <= 0 or h <= 0:
        return 0, 0.0  # Возвращаем 0 пикселей и 0%
    roi = frame[y0:y0+h, x0:x0+w]
    if roi.size == 0:
        return 0, 0.0  # Возвращаем 0 пикселей и 0%
    mask = np.all(roi <= COLOR_THRESHOLD, axis=-1)
    dark_count = np.sum(mask)
    total_pixels = w * h
    dark_percent = (dark_count / total_pixels) * 100 if total_pixels > 0 else 0.0
    return dark_count, dark_percent

# Функция для проверки, является ли контур муравьем
def isAnt(contour, frame):
    perimeter = cv2.arcLength(contour, True)
    if not (MIN_PERIMETER < perimeter < MAX_PERIMETER):
        return False, None

    # Более строгая аппроксимация формы
    approx = cv2.approxPolyDP(contour, APPROX_EPSILON * perimeter, True)
    if len(approx) < MIN_VERTICES or len(approx) > MAX_VERTICES:
        return False, None

    # Проверка соотношения сторон
    (x, y, w, h) = cv2.boundingRect(approx)
    aspect_ratio = w / float(h)
    if not (0.3 < aspect_ratio < 3.5):  # Муравьи обычно вытянуты
        return False, None

    # Фильтр по текстуре (контрастность внутри ROI)
    roi = frame[y:y+h, x:x+w]
    if roi.size == 0:
        return False, None
        
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    std_dev = np.std(gray_roi)
    if std_dev < 15:  # Отсеивает однородные области
        return False, None

    # Ужесточенные проверки
    dark_count, dark_percent = getAntColorsCount(frame, (x, y, w, h))
    if not (MIN_DARK_COUNT <= dark_count <= MAX_DARK_COUNT) or dark_percent > MAX_DARK_PERCENT:
        return False, None

    # Дополнительная проверка площади
    area = cv2.contourArea(contour)
    solidity = area / (w * h) if (w * h) > 0 else 0
    if not (0.2 < solidity < 0.9):  # Исключает слишком "рыхлые" или плотные объекты
        return False, None

    return True, (x, y, w, h)

# поиск муравьёв 
def detect_ants(frame, thresh):
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detected_ants = []
    for contour in contours:
        success, detected_box = isAnt(contour, frame)
        if success:
            detected_ants.append(detected_box)
    return detected_ants

# проверка расстояния для встреч
def check_distances(centers):
    n = len(centers)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            distances[i, j] = np.linalg.norm(centers[i] - centers[j])
            distances[j, i] = distances[i, j]
    return distances < MEETING_DISTANCE

# нахождение встреч
def detect_meetings(trackers, frameN):
    boxes = [(tw.id, tw.box) for tw in trackers]
    if len(boxes) < 2:
        return

    ids = [box[0] for box in boxes]
    boxes_list = [box[1] for box in boxes]

    # Вычисляем центры bounding box'ов
    centers = calculate_centers(boxes_list)

    # Векторизованная проверка расстояний
    proximity_matrix = check_distances(centers)

    # Проверка пересечений bounding box'ов
    for i in range(len(boxes_list)):
        for j in range(i + 1, len(boxes_list)):
            if proximity_matrix[i, j] and do_boxes_intersect(boxes_list[i], boxes_list[j]):
                print(f"Встреча обнаружена: муравьи {ids[i]} и {ids[j]} на кадре {frameN}")
                save_meeting_to_csv(frameN, ids[i], ids[j])
                
                
# cоздание новых трекеров               
def add_new_trackers(frame, detected_ants, trackers, boxes):
    # Преобразуем boxes в список, если это numpy.ndarray
    if isinstance(boxes, np.ndarray):
        boxes = boxes.tolist()
    elif boxes is None:
        boxes = []

    existing_centers = [tw.last_known_position for tw in trackers]
    
    for detected_box in detected_ants:
        new_center = (detected_box[0] + detected_box[2]/2, detected_box[1] + detected_box[3]/2)
        
        # Проверка на близость к существующим трекерам
        is_new = True
        for idx, center in enumerate(existing_centers):
            distance = np.linalg.norm(np.array(center) - np.array(new_center))
            if distance < 50:  # Оптимизированный порог расстояния
                # Пытаемся обновить существующий трекер вместо создания нового
                tracker_wrapper = trackers[idx]
                if tracker_wrapper.missed_frames > 0:
                    tracker = cv2.TrackerCSRT_create()
                    tracker.init(frame, detected_box)
                    tracker_wrapper.tracker = tracker
                    tracker_wrapper.box = detected_box
                    tracker_wrapper.missed_frames = 0
                    is_new = False
                    break

        if is_new:
            # Проверка перекрытия с существующими боксами
            overlap = False
            for box in boxes:
                if intersectionSquare(detected_box, box) > 0.1 * (detected_box[2]*detected_box[3] + box[2]*box[3])/2:
                    overlap = True
                    break
            
            if not overlap:
                tracker = cv2.TrackerCSRT_create()
                tracker.init(frame, detected_box)
                trackers.append(TrackerWrapper(tracker, detected_box))
                boxes.append(detected_box)  # Теперь boxes — это список, и append работает

    # Преобразуем boxes обратно в numpy array, если это необходимо
    if len(boxes) > 0:
        boxes = np.array(boxes, dtype=np.int32)
    else:
        boxes = np.array([], dtype=np.int32)

    return trackers, boxes
      
#обновление трекеров   
def update_trackers(trackers, frame, all_centers, frameN, frame_width, frame_height):
    boxes = []
    inactive_trackers = []

    for tracker_wrapper in trackers:
        success, box = tracker_wrapper.update(frame, all_centers, frameN)


        # Если трекер не смог обновиться
        if not success:
            tracker_wrapper.missed_frames += 1
            if tracker_wrapper.missed_frames > MAX_MISSED_FRAMES:
                inactive_trackers.append(tracker_wrapper)
            continue

        # Проверка на статичность только внутри круга
        center = tracker_wrapper._calculate_center(box)
        if is_point_inside_circle(center, CIRCLE_CENTER, CIRCLE_RADIUS) and tracker_wrapper.is_static():
            print(f"Трекер {tracker_wrapper.id} удален: стоит на месте слишком долго внутри круга.")
            inactive_trackers.append(tracker_wrapper)
            continue

        # Если всё в порядке, обновляем bounding box
        tracker_wrapper.missed_frames = 0
        boxes.append(box)

    # Удаляем неактивные трекеры
    for tracker_wrapper in inactive_trackers:
        trackers.remove(tracker_wrapper)

    return boxes

# Удаление пересекающихся трекеров
def remove_overlapping_trackers(trackers, boxes):
    if len(boxes) > 0:
        boxes = np.array(boxes)  # Преобразуем в numpy array, если это еще не сделано
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = x1 + boxes[:, 2]
        y2 = y1 + boxes[:, 3]

        overlap_matrix = (
            (np.maximum(x1[:, None], x1[None, :]) < np.minimum(x2[:, None], x2[None, :])) &
            (np.maximum(y1[:, None], y1[None, :]) < np.minimum(y2[:, None], y2[None, :]))
        )

        to_keep = np.where(~np.triu(overlap_matrix, k=1).any(axis=1))[0]
        boxes = boxes[to_keep]
        trackers = [trackers[i] for i in to_keep]

    return trackers, boxes


# Функции для обработки видео
def preprocess_frame(frame):
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if BLUR_KERNEL_SIZE[0] % 2 == 1 and BLUR_KERNEL_SIZE[1] % 2 == 1:
        blurred = cv2.GaussianBlur(gray_image, BLUR_KERNEL_SIZE, 0)
    else:
        blurred = gray_image
    _, thresh = cv2.threshold(blurred, THRESHOLD_VALUE, MAX_THRESHOLD, cv2.THRESH_BINARY_INV)
    return thresh

# отрисовка всей информации
def draw_tracking_info(frame, trackers, all_centers, frameN, frame_width):
    # Отрисовка круга
    cv2.circle(frame, CIRCLE_CENTER, CIRCLE_RADIUS, (0, 255, 0), 2)

    for tmp, id, center in all_centers:
        cv2.circle(frame, center, 2, (0, 0, 255), -1)

    for tracker_wrapper in trackers:
        box = tracker_wrapper.box
        cv2.rectangle(frame, tuple(box), tracker_wrapper.color, 1)
        if tracker_wrapper.id is not None:
            x, y, w, h = box
            cv2.putText(frame, f"ID: {tracker_wrapper.id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, tracker_wrapper.color, 2)
        for center in tracker_wrapper.centers:
            cv2.circle(frame, center, 3, tracker_wrapper.color, -1)

    cv2.putText(frame, f"Frame: {frameN}", (frame_width - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# сохраниение результата   
def save_results(all_centers, output_video, video_capture):
    save_to_csv(all_centers)
    output_video.release()
    video_capture.release()
    cv2.destroyAllWindows()


# Основной цикл обработки видео
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video", type=str, required=True, help="path to input video file")
    parser.add_argument("-o", "--output", type=str, default="out.mp4", help="output file name")
    parser.add_argument("-s", "--show", action="store_true", help="show video during processing")
    args = vars(parser.parse_args())

    video = cv2.VideoCapture(args["video"])
    if not video.isOpened():
        print("Ошибка: Не удалось открыть видео.")
        return

    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    output = cv2.VideoWriter(args["output"], cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))

    trackers = []
    all_centers = []
    frameN = 0
    show_video = args["show"]
    window_closed = False

    # Очистка файла ant_meetings.csv перед началом записи
    with open('ant_meetings.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Frame', 'Ant1_ID', 'Ant2_ID'])

    while True:
        success, frame = video.read()
        if not success:
            break

        frameN += 1
        if frameN < FRAME_START:
            continue
        if FRAME_END != float('inf') and frameN > FRAME_END:
            break

        # Обновляем трекеры
        boxes = update_trackers(trackers, frame, all_centers, frameN, frame_width, frame_height)
        trackers, boxes = remove_overlapping_trackers(trackers, boxes)

        # Обработка нового кадра
        thresh = preprocess_frame(frame)
        detected_ants = detect_ants(frame, thresh)
        trackers, boxes = add_new_trackers(frame, detected_ants, trackers, boxes)

        # Обнаружение встреч муравьев
        detect_meetings(trackers, frameN)

        # Отрисовка и сохранение результатов
        draw_tracking_info(frame, trackers, all_centers, frameN, frame_width)

        if show_video and not window_closed:
            cv2.imshow("Tracking", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                cv2.destroyWindow("Tracking")
                window_closed = True

        output.write(frame)

    save_results(all_centers, output, video)
    
if __name__ == "__main__":
    main()