from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
#tumhari mai ke chodo
cap = cv2.VideoCapture("E:\\Car Counter\\Test Video2.mp4")

model = YOLO("../Yolo-Weights/yolov8l.pt")
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird",
              "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
              "backpack", "umbrella", "handbag", "tie", "suitcase", "risbee", "skis", "snowboard",
              "sportsball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
              "tennis racket", "bottle", "wineglass", "cup", "Fork", "knife", "spoon", "bowl",
              "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
              "donut", "cake", "chain", "sofa", "pottedplant", "bed", "diningtable", "toilet",
              "tomonitor", "Laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
              "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

mask = cv2.imread("mask.png")

tracker = Sort(max_age=25, min_hits=3, iou_threshold=0.3)

limits = [460, 250, 700, 250]
totalCount = []

while True:
    success, img = cap.read()

    if not success:
        print("Failed to read frame or video ended.")
        break

    img = cv2.resize(img, (1280, 720))
    mask_resized = cv2.resize(mask, (1280, 720))

    if mask_resized.shape[2] == 4:
        mask_resized = cv2.cvtColor(mask_resized, cv2.COLOR_BGRA2BGR)

    if len(mask_resized.shape) == 2:
        mask_resized = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)

    imgRegion = cv2.bitwise_and(img, mask_resized)

    imgGraphics = cv2.imread("car.png", cv2.IMREAD_UNCHANGED)
    cvzone.overlayPNG(img, imgGraphics, (0, 0))

    results = model(imgRegion, stream=True)
    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:

            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])

            if cls < len(classNames):
                currentclass = classNames[cls]
            else:
                print(
                    f"Class index {cls} is out of range. Skipping detection.")
                continue

            if (currentclass in ["car", "truck", "bus", "motorbike"]) and conf > 0.3:
                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)),
                                   scale=0.6, thickness=1, offset=3)

                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)
    cv2.line(img, (limits[0], limits[1]),
             (limits[2], limits[3]), (0, 0, 255), 5)

    for result in resultsTracker:
        x1, y1, x2, y2, trackerId = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f' {int(trackerId)}', (max(0, x1), max(35, y1)),
                           scale=0.8, thickness=1, offset=8)

        cx, cy = x1+w//2, y1+h//2
        cv2. circle(img, (cx, cy), 5, (255, 0, 255), cv2. FILLED)

        if limits[0] < cx < limits[2] and limits[1] - 20 < cy < limits[1] + 20:
            if totalCount.count(trackerId) == 0:
                totalCount .append(trackerId)
                cv2.line(img, (limits[0], limits[1]),
                         (limits[2], limits[3]), (0, 255, 0), 5)

        cv2.putText(img, str(len(totalCount)), (255, 100),
                    cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8)

    cv2.imshow("Image", img)

    cv2.waitKey(1)
