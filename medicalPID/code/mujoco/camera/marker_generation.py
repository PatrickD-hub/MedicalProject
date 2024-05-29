import cv2
import cv2.aruco


dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
id = 0
size_pixel = 100
img = cv2.aruco.drawMarker(dictionary,id,size_pixel)
cv2.imshow("Charuco",img)

cv2.waitKey(0)
cv2.imwrite("aruco_marker.png", img)