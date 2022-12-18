import cv2
# LOAD AN IMAGE USING 'IMREAD'
img = cv2.imread("Voitko.jpg")
# DISPLAY
cv2.imshow("Voitko", img)
cv2.waitKey(0)