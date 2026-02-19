import cv2
import numpy as np

# Load image
img = cv2.imread("example-of-2d-shapes.png")


# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Blur
blur = cv2.GaussianBlur(gray, (3, 3), 0)

# Edge detection
edges = cv2.Canny(blur, 50, 150)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for c in contours:
    area = cv2.contourArea(c)

    if area < 500:
        continue

    arc_len = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * arc_len, True)

    corners = len(approx)

    x, y, w, h = cv2.boundingRect(approx)

    shape_name = ""

    if corners == 3:
        shape_name = "Triangle"

    elif corners == 4:
        aspect_ratio = float(w) / h
        if 0.95 <= aspect_ratio <= 1.05:
            shape_name = "Square"
        else:
            shape_name = "Rectangle"

    elif corners == 5:
        shape_name = "Pentagon"

    elif corners == 6:
        shape_name = "Hexagon"

    elif corners > 6:
        shape_name = "Circle"

    # Draw contour
    cv2.drawContours(img, [approx], -1, (0, 255, 0), 3)

    # Put text
    cv2.putText(img, shape_name, (x, y - 10),cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)

cv2.imshow("Detected Shapes", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
