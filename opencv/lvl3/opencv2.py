import cv2

img = cv2.imread("coin.jpeg")

grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#convert image to grey scale

"""cv2.imshow("grey",grey)
cv2.waitKey(0)
cv2.destroyAllWindows()"""

# blur using gaussian blur
blur = cv2.GaussianBlur(grey,(5,5),0)#(source,(intensity for blurring), The standard deviation in the X direction)
"""cv2.imshow("blur",blur)
cv2.waitKey(0)
cv2.destroyAllWindows()"""

edges = cv2.Canny(blur,50,150)#edge detection using canny (source,minval,maxval)
"""cv2.imshow("edges",edges)
cv2.waitKey(0)
cv2.destroyAllWindows()"""

countours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)#(source,mode,method)

cnt = 0
for c in countours:
    cnt +=1

print(cnt)