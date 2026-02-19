import cv2
import numpy as np

image = cv2.imread("lane detection.jpg")

lane_image = image.copy()

#Convert to Grayscale
gray_image = cv2.cvtColor(lane_image, cv2.COLOR_BGR2GRAY)

# Blur
blur = cv2.GaussianBlur(gray_image, (5, 5), 0)

#Edge Detection 
edges = cv2.Canny(blur, 50, 150)

#Get Image Size 
height, width = edges.shape

#Create Mask
mask = np.zeros_like(edges)

polygon = np.array([[
    (0, height),
    (width, height),
    (int(width/2), int(height*0.55))
]])

cv2.fillPoly(mask, polygon, 255)

# Apply Mask
cropped_edges = cv2.bitwise_and(edges, mask)

#Detect Lines 
lines = cv2.HoughLinesP(
    cropped_edges,
    2,
    np.pi/180,
    100,
    np.array([]),
    minLineLength=40,
    maxLineGap=5
)

# Separate Left and Right
left_fit = []
right_fit = []

if lines is not None:

    for line in lines:

        x1, y1, x2, y2 = line.reshape(4)

        if x1 == x2:
            continue

        slope, intercept = np.polyfit((x1, x2), (y1, y2), 1)

        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

# Define Lane Drawing Limits
y1 = height
y2 = int(height * 0.6)

# Draw Left Lane 
if len(left_fit) > 0:

    slope, intercept = np.average(left_fit, axis=0)

    if slope == 0:
        slope = 0.001

    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)

    cv2.line(lane_image, (x1, y1), (x2, y2), (255, 0, 0), 10)

# Draw Right Lane 
if len(right_fit) > 0:

    slope, intercept = np.average(right_fit, axis=0)

    if slope == 0:
        slope = 0.001

    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)

    cv2.line(lane_image, (x1, y1), (x2, y2), (255, 0, 0), 10)

cv2.imshow("Result", lane_image)

#cv2.imwrite("detected_refined_lanes.jpg", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
