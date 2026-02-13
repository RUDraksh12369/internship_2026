import cv2
import numpy as np

image_path = "image.jpg"
img = cv2.imread(image_path)

h, w, c = img.shape
print("Height:", h)
print("Width :", w)
print("Channels:", c)

horizontal_flip = cv2.flip(img, 1)#flip horizontally
vertical_flip = cv2.flip(img, 0)#flip vertically

horizontal_symmetry = np.array_equal(img, horizontal_flip)
vertical_symmetry = np.array_equal(img, vertical_flip)

print("Sumetry Analysis")
print("Horizontal symmetry :", "yes" if horizontal_symmetry else "no")#check if the image is same when flipped horizontally
print("Vertical symmetry   :", "yes" if vertical_symmetry else "no")#check if the image is same when flipped vertically

blue = img[:, :, 0]#blue channel
green = img[:, :, 1]#green channel
red = img[:, :, 2]#red channel

total_mean = img.mean()

blue_ratio = blue.mean() / total_mean
green_ratio = green.mean() / total_mean
red_ratio = red.mean() / total_mean

print("Blue channel dominance ratio : ",blue_ratio)
print("Blue channel influence :", "HIGH" if blue_ratio > 1.1 else "LOW" if blue_ratio < 0.9 else "NORMAL")

print("Green channel dominance ratio : ",green_ratio)
print("Green channel influence :", "HIGH" if green_ratio > 1.1 else "LOW" if green_ratio < 0.9 else "NORMAL")

print("Red channel dominance ratio : ",red_ratio)
print("Red channel influence :", "HIGH" if red_ratio > 1.1 else "LOW" if red_ratio < 0.9 else "NORMAL")

cv2.imshow("Original Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("Blue Channel", blue)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("Green Channel", green)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("Red Channel", red)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("blue_channel.jpg", blue)
cv2.imwrite("green_channel.jpg", green)
cv2.imwrite("red_channel.jpg", red)

print("\nForensic output files saved:")
print("- blue_channel.jpg")
print("- green_channel.jpg")
print("- red_channel.jpg")
