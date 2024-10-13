import cv2

img = cv2.imread("D:\Gitesh\BootCamp\Day 4\giraf.jpg")
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# resizing image
resized_img = cv2.resize(img, (500,500))
cv2.imshow("Resized Image", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# converting to gray scale
gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray image", gray_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# crop image
cropped_img = resized_img[100:200,100:400]
cv2.imshow("Cropped image", cropped_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

(h,w) = resized_img.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, 37, 1, 0)
rotated_img = cv2.warpAffine(resized_img, M, (w,h))
cv2.imshow("Rotated image", rotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.rotate(resized_img, cv2.ROTATE_90_CLOCKWISE)