import cv2
import numpy as np
import math

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Camera not opened")
    exit()

canvas = None
prev_x, prev_y = None, None

# BLUE HSV RANGE
lower_blue = np.array([90, 60, 60])
upper_blue = np.array([130, 255, 255])

def detect_shapes(canvas_img, frame):
    gray = cv2.cvtColor(canvas_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < 2000:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)

        corners = len(approx)
        x, y, w, h = cv2.boundingRect(approx)

        circularity = (4 * math.pi * area) / (peri * peri)

        shape_name = "Unknown"

        if corners == 3:
            shape_name = "Triangle"

        elif corners == 4:
            shape_name = "Quadrilateral"

        elif circularity > 0.75:
            shape_name = "Circle"

        else:
            shape_name = "unknown"

        # Draw clean contour
        cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3)

        # Write name beside shape
        cv2.putText(frame, shape_name,(x + w + 10, y + 30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255, 0, 0),2)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)

    if canvas is None:
        canvas = np.zeros_like(frame)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)

        if area > 1000:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))

            if prev_x is None:
                prev_x, prev_y = center
            else:
                cv2.line(canvas, (prev_x, prev_y), center, (0, 0, 255), 6)
                prev_x, prev_y = center
        else:
            prev_x, prev_y = None, None
    else:
        prev_x, prev_y = None, None

    combined = cv2.add(frame, canvas)

    # Detect ALL shapes
    detect_shapes(canvas, combined)

    cv2.imshow("Smart Multi Shape Detector | C: Clear | ESC: Exit", combined)

    key = cv2.waitKey(1)

    if key == 27:
        break
    elif key == ord('c'):
        canvas = np.zeros_like(frame)

cap.release()
cv2.destroyAllWindows()