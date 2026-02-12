import cv2
img = cv2.imread("image.jpg")

h,w,c = img.shape
print(h,w,c)

"""#resize
resize = cv2.resize(img,(500,500))
cv2.imshow("resized",resize)
cv2.waitKey(0)
cv2.destroyAllWindows()"""

"""#crop
cropped = img[0:400,0:400]
cv2.imshow("cropped",cropped)
cv2.waitKey(0)
cv2.destroyAllWindows()"""

"""#rotate
center = (w//2,h//2)
matrix = cv2.getRotationMatrix2D(center,90,1)
rotated = cv2.warpAffine(img,matrix,(w,h))
cv2.imshow("rotated",rotated)
cv2.waitKey(0)
cv2.destroyAllWindows()"""

#flip
flipped = cv2.flip(img,0)#vertical
flipped = cv2.flip(img,1)#horizontal
flipped = cv2.flip(img,-1)#both
cv2.imshow("flipped",flipped)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("flipped.jpg",flipped)