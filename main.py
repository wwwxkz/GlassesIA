import numpy as np
import cv2
import imutils

# Data Models
#https://github.com/Itseez/opencv/tree/master/data/haarcascades
face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('data/haarcascade_eye.xml')

# Webcam
cap = cv2.VideoCapture(0)

# Glasses
glasses = cv2.imread('images/glasses.png', -1)

# Overlay webcam video
def transparentOverlay(src, overlay, pos=(0, 0), scale=1):
    overlay = cv2.resize(overlay, (0, 0), fx=scale, fy=scale)
    h, w, _ = overlay.shape
    rows, cols, _ = src.shape
    y, x = pos[0], pos[1]

    for i in range(h):
        for j in range(w):
            if x + i >= rows or y + j >= cols:
                continue
            alpha = float(overlay[i][j][3] / 255.0)  # Read the alpha channel
            src[x + i][y + j] = alpha * overlay[i][j][:3] + (1 - alpha) * src[x + i][y + j]
    return src

while 1:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Eyes weights
        eyes = eye_cascade.detectMultiScale(roi_gray)
        # Eye counter
        eye = 0
        for (ex,ey,ew,eh) in eyes:
            # Picking up each eyes position
            if(eye == 0):
                # If is first eye record the cordinates
                ey1 = ey
                eye += 1
                print(eye)
            else:
                # Rotating glasses
                if(ey1 > ey):
                    glasses_rotated = imutils.rotate(glasses, abs(int(ey1+ey1/2)-int(ey+ey/2)))
                else:
                    glasses_rotated = imutils.rotate(glasses, -abs(int(ey1+ey1/2)-int(ey+ey/2)))

                # Eye middle point
                cv2.circle(roi_color, (ex, int(ey+ey/2)), 4, (255, 0, 0), -1)
                # Glasses size
                glasses_symin = int(y + 1.2 * h / 5)
                glasses_symax = int(y + 2.8 * h / 5)
                sh_glasses = glasses_symax - glasses_symin

                face_glasses_roi_color = frame[glasses_symin:glasses_symax, x:x+w]

                # Puting glasses into video thoroght transparentOverlay
                specs = cv2.resize(glasses_rotated, (w, sh_glasses),interpolation=cv2.INTER_CUBIC)
                transparentOverlay(face_glasses_roi_color,specs)

            # Eye rectangle marker
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    # Name of frame
    cv2.imshow('Glasses',frame)
    # Quit program (esc)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# Destroy windows
cap.release()
cv2.destroyAllWindows()
