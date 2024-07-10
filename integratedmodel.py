from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import pickle
import face_recognition
import numpy as np
import os

confidence = 0.9
face_match_threshold = 0.4

model = YOLO("models/best.pt")

classNames = ["fake", "real"]

def load_encodings(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
    
encodeDict = {}
if os.path.exists('encodings.pickle'):
    encodeDict = load_encodings('encodings.pickle')
    
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

prev_frame_time = 0
new_frame_time = 0

while True:
    new_frame_time = time.time()
    success, img = cap.read()
    results = model(img, stream=True)
    is_real = False

    for r in results:
        boxes = r.boxes
        for box in boxes:
            #x1, y1, x2, y2 = box.xyxy[0]
            #x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            #w, h = x2-x1, y2-y1
            conf = math.ceil((box.conf[0] * 100))/100
            if conf > confidence:
                cls = int(box.cls[0])
                if classNames[cls] == "real":
                    is_real = True
            #cvzone.cornerRect(img, (x1, y1, w, h), colorC=color, colorR=color)
                #cvzone.putTextRect(img, f'{classNames[cls].upper()} {int(conf*100)}%',
                 #                  (max(0, x1), max(35, y1)), scale=1, thickness=1, colorR=color, colorB=color)
    if is_real:
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        faces_in_frame = face_recognition.face_locations(imgS)
        encoded_faces = face_recognition.face_encodings(imgS, faces_in_frame)
        
        for encode_face, faceloc in zip(encoded_faces, faces_in_frame):
            matchIndex = None
            min_distance = float('inf')
            for className, encoding_list in encodeDict.items():
                for encoding in encoding_list:
                    distance = face_recognition.face_distance([encoding], encode_face)
                    if distance < min_distance:
                        min_distance = distance
                        matchIndex = className

            if matchIndex is not None and min_distance <= face_match_threshold:
                name = matchIndex.upper().lower()
            else:
                name = "unknown"

            y1, x2, y2, x1 = faceloc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-5), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
    else:
        print("Spoofing attempt")
        break
    
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    
    cv2.imshow("Image", img)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
