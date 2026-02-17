import cv2
import numpy as np

# Load image
img = cv2.imread("download.jpg")

h, w, c = img.shape

original = img.copy()
cv2.putText(original, "Original", (20,40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

blue = img[:,:,0]
blue = cv2.cvtColor(blue, cv2.COLOR_GRAY2BGR)
cv2.putText(blue, "Blue Channel", (20,40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

flip = cv2.flip(img, 1)
cv2.putText(flip, "Flip Horizontal", (20,40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

center = (w//2, h//2)
matrix = cv2.getRotationMatrix2D(center, 45, 1)
rotate = cv2.warpAffine(img, matrix, (w,h))
cv2.putText(rotate, "Rotate 45", (20,40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

original = cv2.resize(original, (400,300))
blue = cv2.resize(blue, (400,300))
flip = cv2.resize(flip, (400,300))
rotate = cv2.resize(rotate, (400,300))

top = np.hstack((original, blue))
bottom = np.hstack((flip, rotate))

collage = np.vstack((top, bottom))

cv2.imshow("Project 1 Collage", collage)
cv2.imwrite("project1_collage.jpg", collage)

cv2.waitKey(0)
cv2.destroyAllWindows()
