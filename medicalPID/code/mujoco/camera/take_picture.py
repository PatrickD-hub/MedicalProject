import cv2
import os

folder = './calib'
webcam = cv2.VideoCapture(0)
if not webcam.isOpened():
    print("Cannot open camera")
    exit

image_counter = 0

while True:
    check, frame = webcam.read()

    if not check:
        print("cannot receive frame")
        break

    cv2.imshow("Capturing", frame)
    key = cv2.waitKey(1)

    if key == ord('s'):

        cv2.imwrite(os.path.join(folder, f'{image_counter}.png'), frame)
        image_counter += 1
        print(f"Image {image_counter}")

    elif key == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
        

