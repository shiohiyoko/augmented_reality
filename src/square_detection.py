import cv2
import math

def angle(pt1, pt2, pt0) -> float:
    dx1 = float(pt1[0,0] - pt0[0,0])
    dy1 = float(pt1[0,1] - pt0[0,1])
    dx2 = float(pt2[0,0] - pt0[0,0])
    dy2 = float(pt2[0,1] - pt0[0,1])
    v = math.sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) )
    return (dx1*dx2 + dy1*dy2)/ v


# Detect squares
def findSquares(bin_image, image, cond_area = 1000):
    
    approx_point = []

    # contour extraction
    contours, _ = cv2.findContours(bin_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    for i, cnt in enumerate(contours):
        # Approximates contours with accuracy proportional to contour perimeter
        arclen = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, arclen*0.02, True)

        # check if the contour has 4 corners, not too small, and convex
        area = abs(cv2.contourArea(approx))
        if approx.shape[0] == 4 and area > cond_area and cv2.isContourConvex(approx) :

            rcnt = approx.reshape(-1,2)

            # draw the rectangle
            cv2.polylines(image, [rcnt], True, (0,0,255), thickness=2, lineType=cv2.LINE_8)

            # append
            approx_point.append([rcnt, area])
    
    approx_point.sort(key=lambda x: x[1])
    return approx_point, image

