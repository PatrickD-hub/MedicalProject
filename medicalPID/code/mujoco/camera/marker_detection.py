import cv2
from matplotlib import image
#import cv2.aruco
import numpy as np


def toTransformMatrix(rvec, tvec):
    rotmat, Jc = cv2.Rodrigues(rvec)
    tmat = np.identity(4)
    tmat[0:3, 0:3] = rotmat
    tmat[0:3, 3] = np.squeeze(tvec)
    return tmat


def trackMarker(img, cameraMatrix, markerLength, showImage=False, bw=True):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)


    if (bw):
        dst = cv2.equalizeHist(img) # only for bw images. If rgb image, remove this line.
    else:
        dst = img

    
    corners, ids, rejected_points = detector.detectMarkers(img)


    distCoeffs = np.zeros(4)

    rvecs, tvecs, objpoints = cv2.aruco.estimatePoseSingleMarkers(
        corners, markerLength, cameraMatrix, distCoeffs)

    if tvecs is not None:
        camTmarker = toTransformMatrix(rvecs[0], tvecs[0])
        if showImage:
            print(camTmarker)
            cv2.drawFrameAxes(img, cameraMatrix, distCoeffs,
                            rvecs[0], tvecs[0], 0.1)
            
    cv2.imshow("Aruco", img)

    #return camTmarker


if __name__ == "__main__":

    markerLength = 0.038

    cameraMatrix = np.load('mtx.npy')
 
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read(0)

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        markerPosIrFrame = trackMarker(gray, cameraMatrix, markerLength, True)
        # Display the resulting frame
        #cv2.imshow('frame', gray)
        
        if cv2.waitKey(1) == ord('q'):
            break
        
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
