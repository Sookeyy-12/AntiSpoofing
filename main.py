from ultralytics import YOLO
import cv2
import cvzone
import math
import time

confidence = 0.9

cap = cv2.VideoCapture(0) # for webcam
cap.set(3, 640)
cap.set(4, 480)
# cap = cv2.VideoCapture("test.mp4") # for video file

model = YOLO("models/best.pt")

classNames = ["fake", "real"]

prev_frame_time = 0
new_frame_time = 0

while True:
    new_frame_time = time.time()
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            w, h = x2-x1, y2-y1
            # Confidence
            conf = math.ceil((box.conf[0] * 100))/100
            # class name
            if conf>confidence:
                cls = int(box.cls[0])
                if classNames[cls]=="real":
                    color=(0, 255, 0)   #green
                else:
                    color=(0, 0, 255)   #red

                cvzone.cornerRect(img, (x1, y1, w, h), colorC=color, colorR=color)
                cvzone.putTextRect(img, f'{classNames[cls].upper()} {int(conf*100)}%', 
                                (max(0, x1), max(35, y1)), scale = 1, thickness=1, colorR=color, colorB=color)

    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(fps)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1) & 0xFF

    # Exit if 'q' key is pressed
    if key == ord("q"):
        break

cv2.destroyAllWindows()