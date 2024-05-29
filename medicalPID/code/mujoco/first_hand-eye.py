import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

# Set global print options
np.set_printoptions(precision=5, suppress=True)

def Rodrigues(rvec, R):
    r = R.from_rotvec(rvec)
    R[:] = r.as_matrix()

def calculate_marker_position(marker_id, corners, ids, marker_size_meters, img_width, img_height):
    marker_indices = np.where(ids == marker_id)[0]
    if len(marker_indices) == 0:
        return None
    
    marker_corners = corners[marker_indices[0]]
    marker_corners = marker_corners.reshape((4, 2))
    obj_points = np.array([[0, 0, 0],
                            [marker_size_meters, 0, 0],
                            [marker_size_meters, marker_size_meters, 0],
                            [0, marker_size_meters, 0]], dtype=np.float32)
    image_points = np.array(marker_corners, dtype=np.float32)
    _, rvec, tvec = cv2.solvePnP(obj_points, image_points, np.eye(3), None)
    
    return tvec.flatten()

def aruco_display(corners, ids, rejected, image, img_width, img_height):
    pose = None 
    if len(corners) > 0:
        ids = ids.flatten()
        
        for (markerCorner, markerID) in zip(corners, ids):
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners

            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

            cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)

            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)

            cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
            
            # Calculate distance based on the known physical size of the marker
            marker_size_meters = 0.04  # Adjust this value based on the actual size of the marker
            distance, rvec, tvec = calculate_distance_and_pose(marker_size_meters, img_width, img_height, corners)

            # Print the size in the image and the calculated distance
            size_in_image = np.linalg.norm(np.array(topLeft) - np.array(topRight))
            print(f"Marker ID: {markerID}")
            print(f"Size in image: {size_in_image:.2f} pixels")
            print(f"Calculated distance: {distance:.2f} meters")

            # Print the x and y coordinates of the marker center
            print(f"Center coordinates: ({cX}, {cY})")

            # Convert rotation vector to quaternions
            tvec = tvec.flatten()
            rvec = rvec.flatten()
            pose = np.hstack((tvec,rvec))
            # rotation = R.from_rotvec(rvec)
            # quat = rotation.as_quat()

            # Print rotation and translation vectors
            print(f"Rotation vector (rvec): {rvec}")
            print(f"Translation vector (tvec): {tvec}")
            # print("Pose : ", pose)

            # Print the distance
            cv2.putText(image, f"Distance: {distance:.2f} meters", (cX - 50, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
        error = np.linalg.norm(tvec[0] - tvec[1])
        print(f"Distance between markers: {error: .5f} meters")
    
    return image, pose



def calculate_distance_and_pose(marker_size_meters, img_width, img_height, marker_corners):
    # Assuming the camera focal length in pixels (you may replace this with actual camera intrinsics)
    focal_length_px = max(img_width, img_height)  # Assuming worst case scenario

    # Calculate the distance using the formula: distance = (size_of_marker * focal_length) / size_in_image
    size_in_image = np.linalg.norm(marker_corners[0] - marker_corners[2])  # Euclidean distance between opposite corners
    distance = (marker_size_meters * focal_length_px) / size_in_image

    # Calculate rotation and translation vectors
    obj_points = np.array([[0, 0, 0],
                            [marker_size_meters, 0, 0],
                            [marker_size_meters, marker_size_meters, 0],
                            [0, marker_size_meters, 0]], dtype=np.float32)
    image_points = np.array(marker_corners, dtype=np.float32)
    _, rvec, tvec = cv2.solvePnP(obj_points, image_points, np.eye(3), None)

    return distance, rvec, tvec

arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
arucoParams = cv2.aruco.DetectorParameters()

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while cap.isOpened():
    ret, img = cap.read()

    h, w, _  = img.shape

    width = 1000
    height = int(width*(h/w))

    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)

    detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)

    corners, ids, rejected = detector.detectMarkers(img)

    detected_markers, pose = aruco_display(corners, ids, rejected, img, w, h)
    
    print("Pose : ", pose)
    cv2.imshow("Image", detected_markers)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()
