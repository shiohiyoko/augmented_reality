
import cv2
import math
import threading
import numpy as np
import matplotlib.pyplot as plt
from objloader_simple import *
from collections import deque
from square_detection import findSquares
from projection_matrix import projection_matrix

def render(frame, obj, projection, referenceImage, model_scale, color=False):
    """
    Render a loaded obj model into the current video frame
    """
    vertices = obj.vertices
    scale_matrix = np.eye(3) * model_scale
    h, w = referenceImage.shape

    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)

        # render model in the middle of the reference surface. To do so,
        # model points must be displaced
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        framePts = np.int32(dst)

        cv2.fillConvexPoly(frame, framePts, (137, 27, 211))

    return frame

def main():
    model_scale = 30
    obj = OBJ("./models/Treee.obj", swapyz=True)

    # estimated from cameracallibration.py
    camera_parameters = np.array([[666.60249289,   0.,         281.64082013],
                                [  0.,         665.30488438, 250.22007747],
                                [  0.,           0.,           1.        ]])

    # refrence image and camera setting
    referenceImage = cv2.imread("./img/hiro.jpg", 0)
    cap = cv2.VideoCapture(0)

    while True:
        # read the current frame
        ret, frame = cap.read()
        
        if not ret:
            print("frame capturing failed(;;)")
        else:
            # creating a binary image for extracting squares
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            # Find the squares and get the coordinate for all rectangles
            approx_point, frame = findSquares(bw, frame)

            print(len(approx_point), "squares found!")
            if len(approx_point) > 0:

                # get the largest rectangle
                marker_point = np.float32(approx_point[0][0]).reshape(-1,1,2)

                # homography calculation
                ref_w ,ref_h= referenceImage.shape
                ref_point = [[0,0], [0, ref_h-1], [ref_w-1, ref_h-1], [ref_w-1,0]]
                ref_point = np.float32(ref_point).reshape(-1,1,2)
                homography, _ = cv2.findHomography( ref_point, marker_point, cv2.RANSAC, 5.0)
                
                transformedCorners = cv2.perspectiveTransform(ref_point, homography)

                # Draw a polygon on the second image joining the transformed corners
                frame = cv2.polylines(
                    frame, [np.int32(transformedCorners)], True, 255, 3, cv2.LINE_AA,
                    )
                
                # obtain 3D projection matrix from homography matrix and camera parameters
                projection = projection_matrix(camera_parameters, homography)

                # project cube or model
                frame = render(frame, obj, projection, referenceImage, model_scale, False)

                # show result
            cv2.imshow("norm", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    main()
