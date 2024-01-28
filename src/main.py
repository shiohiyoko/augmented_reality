
import cv2
import math
import threading
import numpy as np
import matplotlib.pyplot as plt
from objloader_simple import *
from collections import deque
from square_detection import findSquares
from projection_matrix import projection_matrix
from feature_extractor import FeatureExtractor

def render(frame, obj, projection, target_image, model_scale, color=False):
    """
    Render a loaded obj model into the current video frame
    """
    vertices = obj.vertices
    scale_matrix = np.eye(3) * model_scale
    h, w = target_image.shape

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


def compute_reprojection_error(projections):
    min_error = 1000
    projection = []

    for homography, points1, points2 in homography:
        # Transform points using homography
        projected_points = cv2.perspectiveTransform(points1, homography)
        print(projected_points.shape)
        print(points2.shape)
    
    # Calculate Euclidean distance between projected points and actual points
        errors = np.sqrt(np.sum((projected_points - points2)**2, axis=2))
        
        # Compute mean reprojection error
        mean_error = np.mean(errors)
        if mean_error < min_error:
            projection = [homography, points1, points2]
        
    return projection

def min_reprojection_error(projections):
    min_error = 1000
    projection = None

    for homography, points1, points2 in projections:
        # Transform points using homography
        projected_points = cv2.perspectiveTransform(points1, homography)
        print(projected_points.shape)
        print(points2.shape)
    
    # Calculate Euclidean distance between projected points and actual points
        errors = np.sqrt(np.sum((projected_points - points2)**2, axis=2))
        
        # Compute mean reprojection error
        mean_error = np.mean(errors)
        if mean_error < min_error:
            projection = [homography, points1, points2]
    
    return projection


def main():
    model_scale = 30
    obj = OBJ("./models/Treee.obj", swapyz=True)

    # estimated from cameracallibration.py
    camera_parameters = np.array([[666.60249289,   0.,         281.64082013],
                                [  0.,         665.30488438, 250.22007747],
                                [  0.,           0.,           1.        ]])

    # refrence image and camera setting
    target_image = cv2.imread("./img/hiro.jpg")
    target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
    cap = cv2.VideoCapture(0)

    while True:
        # read the current frame
        ret, frame = cap.read()
        
        if not ret:
            print("frame capturing failed(;;)")
        else:
            # creating a binary image for extracting squares
            reference_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, bw = cv2.threshold(reference_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            # Find the squares and get the coordinate for all rectangles
            approx_point, frame = findSquares(bw, frame)

            # feature extraction
            projection_list = []

            for square in approx_point:

                point = square[0]
                # print(approx_point)
                min_point = [np.min(point[:,0]), np.min(point[:,1])]
                max_point = [np.max(point[:,0]), np.max(point[:,1])]
                
                ref_img = reference_image[min_point[1]:max_point[1], min_point[0]:max_point[0]]

                # Initialize ORB detector
                orb = cv2.ORB_create()

                keypoints_target, keypoints_reference, matches = FeatureExtractor(target_image, ref_img, orb)
            

                if len(matches) > 4:
                    eq = lambda x, y : x + y

                    target_point = np.float32([keypoints_target[match.queryIdx].pt for match in matches]).reshape(-1, 1, 2)
                    ref_point = np.float32([list(map(eq, list(keypoints_reference[match.trainIdx].pt), min_point)) for match in matches]).reshape(-1, 1, 2)
                    # print(keypoints_reference[0].pt[0])


                    homography, _ = cv2.findHomography( target_point, ref_point, cv2.RANSAC, 5.0)

                    projection_list.append([homography, target_point, ref_point])

            projection = min_reprojection_error(projection_list)
            if projection is not None:
                homography, target_point, ref_point = projection
                # homography calculation
                ref_w ,ref_h= target_image.shape
                ref_point = [[0,0], [0, ref_h-1], [ref_w-1, ref_h-1], [ref_w-1,0]]
                ref_point = np.float32(ref_point).reshape(-1,1,2)

                transformedCorners = cv2.perspectiveTransform(ref_point, homography)

                # Draw a polygon on the second image joining the transformed corners
                frame = cv2.polylines(
                    frame, [np.int32(transformedCorners)], True, 255, 3, cv2.LINE_AA,
                    )
                
                # obtain 3D projection matrix from homography matrix and camera parameters
                projection = projection_matrix(camera_parameters, homography)

                # project cube or model
                frame = render(frame, obj, projection, target_image, model_scale, False)

                # show result
            cv2.imshow("norm", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    main()
