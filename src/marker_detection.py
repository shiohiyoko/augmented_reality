import cv2

def findMarker(points):
    centroid = []

    for point in points:
        coord = [0,0]
        for p in point:
            coord[0] += p[0]
            coord[1] += p[1]
        coord[0] /= 4
        coord[1] /= 4
        centroid.append(coord)
    
    same_coord = [0]*len(points)

