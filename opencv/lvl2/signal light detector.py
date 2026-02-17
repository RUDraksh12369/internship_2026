import cv2
import numpy as np

img = cv2.imread("traffic_signal.jpg")



img = cv2.resize(img, (400,800))

h, w, _ = img.shape

# Convert to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Divide into 3 regions (top, middle, bottom)
top = hsv[0:h//3, :]
middle = hsv[h//3:2*h//3, :]
bottom = hsv[2*h//3:h, :]

# RED
lower_red1 = np.array([0,120,70])
upper_red1 = np.array([10,255,255])
lower_red2 = np.array([170,120,70])
upper_red2 = np.array([180,255,255])

mask_red1 = cv2.inRange(top, lower_red1, upper_red1)
mask_red2 = cv2.inRange(top, lower_red2, upper_red2)
red_count = np.count_nonzero(mask_red1 + mask_red2)

#  YELLOW
lower_yellow = np.array([22,150,150])
upper_yellow = np.array([32,255,255])
mask_yellow = cv2.inRange(middle, lower_yellow, upper_yellow)
yellow_count = np.count_nonzero(mask_yellow)

#  GREEN
lower_green = np.array([40,100,100])
upper_green = np.array([85,255,255])
mask_green = cv2.inRange(bottom, lower_green, upper_green)
green_count = np.count_nonzero(mask_green)

print("Red pixels:", red_count)
print("Yellow pixels:", yellow_count)
print("Green pixels:", green_count)

# Decide active light
if red_count > yellow_count and red_count > green_count:
    print("ACTIVE LIGHT: RED")

elif yellow_count > red_count and yellow_count > green_count:
    print("ACTIVE LIGHT: YELLOW")

else:
    print("ACTIVE LIGHT: GREEN")

cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
