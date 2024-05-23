import cvzone
from cvzone.FaceDetectionModule import FaceDetector
import cv2

####################################
offsetPercentageW = 10
offsetPercentageH = 20
confidence = 0.8
camWidth, camHeight = 640, 480
floatingPrecision = 6
####################################

cap = cv2.VideoCapture(1)
cap.set(3, camWidth)
cap.set(3, camHeight)
detector = FaceDetector()

while True:
    success, img = cap.read()
    img, bboxs = detector.findFaces(img, draw=False)

    if bboxs:
        # bboxInfo - "id", "bbox", "score", "center"
        center = bboxs[0]["center"]
        for bbox in bboxs:
            x, y, w, h = bbox["bbox"]
            score = bbox["score"][0]

            # ----- Check Score ----- #
            if score > confidence:

                # ----- adding an offset to the face detected ----- #
                offsetW = (offsetPercentageW / 100)*w
                offsetH = (offsetPercentageH / 100)*h
                x = int(x - offsetW)
                w = int(w + offsetW*2)
                y = int(y - offsetH*2.5)
                h = int(h + offsetH*3)

                # ----- To avoid values below 0 ----- #
                if x < 0: x = 0
                if y < 0: y = 0
                if w < 0: w = 0
                if h < 0: h = 0

                # ----- Find Blurriness ----- #
                imgFace = img[y:y+h, x:x+w]
                cv2.imshow("Face", imgFace)
                blurValue = int(cv2.Laplacian(imgFace, cv2.CV_64F).var())

                # ----- Normalize Values ----- #
                ih, iw, _ = img.shape
                xc, yc = (x+w/2), (y+h/2)
                xcn, ycn = round(xc/iw, floatingPrecision), round(yc/ih, floatingPrecision)
                wn, hn = round(w/iw, floatingPrecision), round(h/ih, floatingPrecision)
                print(xcn, ycn, wn, hn, blurValue)

                # ----- To avoid values below 0 ----- #
                if xcn > 1: xcn = 1
                if ycn > 1: ycn = 1
                if wn > 1: wn = 1
                if hn > 1: hn = 1

                # ----- Drawing ----- #            
                cv2.rectangle(img, (x, y, w, h), (255, 0, 0), 1)
                cvzone.putTextRect(img, f'Score: {int(score*100)}% Blur: {blurValue}', (x, y-30), scale=1.25, thickness = 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)