import cv2
import numpy as np

# Load image
image = cv2.imread("download.jpg")

image = cv2.resize(image, (400, 400))
filtered_images = []

# values for text placement
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.9
thickness = 2
text_y = 35

# 1. Clarendon

clarendon = cv2.convertScaleAbs(image, alpha=1.3, beta=15)
clarendon[:, :, 0] = cv2.add(clarendon[:, :, 0], 30)
clarendon[:, :, 2] = cv2.subtract(clarendon[:, :, 2], 10)

cv2.rectangle(clarendon, (0, 0), (400, 50), (0, 0, 0), -1)
cv2.putText(clarendon, "Clarendon", (10, text_y),font, font_scale, (255,255,255), thickness)
filtered_images.append(clarendon)


# 2. Juno

juno = cv2.convertScaleAbs(image, alpha=1.2, beta=10)
juno[:, :, 2] = cv2.add(juno[:, :, 2], 40)
juno[:, :, 1] = cv2.add(juno[:, :, 1], 10)

cv2.rectangle(juno, (0, 0), (400, 50), (0, 0, 0), -1)
cv2.putText(juno, "Juno", (10, text_y),font, font_scale, (255,255,255), thickness)
filtered_images.append(juno)

# 3. Lark

lark = cv2.convertScaleAbs(image, alpha=1.1, beta=30)
lark[:, :, 0] = cv2.add(lark[:, :, 0], 20)
lark[:, :, 1] = cv2.add(lark[:, :, 1], 10)

cv2.rectangle(lark, (0, 0), (400, 50), (0, 0, 0), -1)
cv2.putText(lark, "Lark", (10, text_y),font, font_scale, (255,255,255), thickness)
filtered_images.append(lark)

# 4. Gingham

gingham = cv2.convertScaleAbs(image, alpha=0.9, beta=20)
gingham = cv2.cvtColor(gingham, cv2.COLOR_BGR2HSV)
gingham[:, :, 1] = gingham[:, :, 1] * 0.5
gingham = cv2.cvtColor(gingham, cv2.COLOR_HSV2BGR)

cv2.rectangle(gingham, (0, 0), (400, 50), (0, 0, 0), -1)
cv2.putText(gingham, "Gingham", (10, text_y),font, font_scale, (255,255,255), thickness)
filtered_images.append(gingham)

# 5. Valencia
valencia = image.copy()
valencia[:, :, 2] = cv2.add(valencia[:, :, 2], 25)
valencia[:, :, 1] = cv2.add(valencia[:, :, 1], 15)
blur = cv2.GaussianBlur(valencia, (0,0), 3)
valencia = cv2.addWeighted(image, 0.8, blur, 0.2, 0)

cv2.rectangle(valencia, (0, 0), (400, 50), (0, 0, 0), -1)
cv2.putText(valencia, "Valencia", (10, text_y),font, font_scale, (255,255,255), thickness)
filtered_images.append(valencia)

# 6. Nashville
nashville = image.copy()
nashville[:, :, 2] = cv2.add(nashville[:, :, 2], 35)
nashville[:, :, 0] = cv2.add(nashville[:, :, 0], 15)

cv2.rectangle(nashville, (0, 0), (400, 50), (0, 0, 0), -1)
cv2.putText(nashville, "Nashville", (10, text_y),font, font_scale, (255,255,255), thickness)
filtered_images.append(nashville)

# 7. Moon
moon = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
moon = cv2.convertScaleAbs(moon, alpha=1.6, beta=0)
moon = cv2.cvtColor(moon, cv2.COLOR_GRAY2BGR)

cv2.rectangle(moon, (0, 0), (400, 50), (0, 0, 0), -1)
cv2.putText(moon, "Moon", (10, text_y),font, font_scale, (255,255,255), thickness)
filtered_images.append(moon)

# 8. Rise
rise = cv2.convertScaleAbs(image, alpha=1.05, beta=20)
rise[:, :, 2] = cv2.add(rise[:, :, 2], 20)

cv2.rectangle(rise, (0, 0), (400, 50), (0, 0, 0), -1)
cv2.putText(rise, "Rise", (10, text_y),font, font_scale, (255,255,255), thickness)
filtered_images.append(rise)

# 9. Hudson
hudson = image.copy()
hudson[:, :, 0] = cv2.add(hudson[:, :, 0], 40)
hudson[:, :, 1] = cv2.subtract(hudson[:, :, 1], 10)

cv2.rectangle(hudson, (0, 0), (400, 50), (0, 0, 0), -1)
cv2.putText(hudson, "Hudson", (10, text_y),font, font_scale, (255,255,255), thickness)
filtered_images.append(hudson)

# 10. Slumber

slumber = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
slumber[:, :, 1] = slumber[:, :, 1] * 0.4
slumber = cv2.cvtColor(slumber, cv2.COLOR_HSV2BGR)
slumber = cv2.convertScaleAbs(slumber, alpha=0.95, beta=10)

cv2.rectangle(slumber, (0, 0), (400, 50), (0, 0, 0), -1)
cv2.putText(slumber, "Slumber", (10, text_y),font, font_scale, (255,255,255), thickness)
filtered_images.append(slumber)

# Create Collage 
row1 = np.hstack(filtered_images[:5])
row2 = np.hstack(filtered_images[5:])
collage = np.vstack((row1, row2))

cv2.imwrite("filtercollage.jpg", collage)
cv2.imshow(" Filters", collage)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("filtercollage.jpg")
