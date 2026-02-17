import cv2
import numpy as np

img = cv2.imread("traffic.jpeg")
blur = cv2.GaussianBlur(img, (5,5), 0)

gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)


edges = cv2.Canny(gray, 100, 200)

# Morphological closing (connect broken edges)
kernel = np.ones((5,5), np.uint8)
edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

# Dilate to strengthen shapes
edges = cv2.dilate(edges, kernel, iterations=1)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Vehicle Counter
vehicle_count = 0


for cnt in contours:

    area = cv2.contourArea(cnt)

    if area < 500:
        continue

    x, y, w, h = cv2.boundingRect(cnt)

    aspect_ratio = w / float(h)
    
    """if 0.5 < aspect_ratio < 3.5:
        vehicle_count += 1
        cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)"""

    vehicle_count += 1

    # Draw rectangle around detected vehicle
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

print("Total Vehicles Detected:", vehicle_count)

# Display count on image
cv2.putText(img, f"Total Vehicles: {vehicle_count}", (20,40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

# ---------- Show ----------
#cv2.imshow("Edges", edges)
cv2.imshow("Vehicles Detected", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
