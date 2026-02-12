import cv2
img = cv2.imread("image.jpg")
cv2.imshow("My Image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(type(img))
print(img.shape)#shows height,width,channals as oputput

red = img[:,:,2]
cv2.imshow("Red",red)
cv2.waitKey(0)
cv2.destroyAllWindows()

h,w = red.shape
print(h,w)