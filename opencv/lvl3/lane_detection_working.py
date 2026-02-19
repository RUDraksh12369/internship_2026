import cv2
import numpy as np

# Load image
img = cv2.imread("lane detection.jpg")
lane_img = img.copy()

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Blur to reduce noise
blur = cv2.GaussianBlur(gray, (5,5), 0)

# Detect edges
edges = cv2.Canny(blur, 50, 150)

# Get image size
height, width = edges.shape

# Create wider triangular mask (covers full driver lane)
mask = np.zeros_like(edges)

triangle = np.array([[
    (0, height),                     # bottom left (wider)
    (width, height),                 # bottom right (wider)
    (int(width*0.5), int(height*0.55))  # top center
]])

cv2.fillPoly(mask, triangle, 255)

# Apply mask
cropped = cv2.bitwise_and(edges, mask)

# Detect lines (more sensitive settings)
lines = cv2.HoughLinesP(
    cropped,
    1,              # distance resolution (pixels)
    np.pi/180,      # angle resolution (1 degree)
    30,             # lower threshold = detect weaker lines
    30,             # minimum line length
    100             # allow larger gaps between segments
)

left = []
right = []

# Separate left and right lane lines
if lines is not None:
    for line in lines:
        x1,y1,x2,y2 = line[0]

        if x1 == x2:
            continue

        slope = (y2-y1)/(x2-x1)

        # Allow more slopes (detect distant lanes)
        if abs(slope) < 0.3:
            continue

        intercept = y1 - slope*x1

        if slope < 0:
            left.append((slope, intercept))
        else:
            right.append((slope, intercept))


# Draw left lane
if len(left) > 0:

    slope, intercept = np.average(left, axis=0)

    y1 = height
    y2 = int(height*0.6)

    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)

    cv2.line(lane_img,(x1,y1),(x2,y2),(0,255,0),5)


# Draw right lane
if len(right) > 0:

    slope, intercept = np.average(right, axis=0)

    y1 = height
    y2 = int(height*0.6)

    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)

    cv2.line(lane_img,(x1,y1),(x2,y2),(0,255,0),5)


# Show result
#cv2.imshow("Driver Lane Only", lane_img)

# Optional debug views
#cv2.imshow("Edges", edges)
cv2.imshow("Masked Edges", cropped)

cv2.waitKey(0)
cv2.destroyAllWindows()
