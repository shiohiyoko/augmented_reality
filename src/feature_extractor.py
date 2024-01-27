import cv2

def FeatureExtractor(target_img, ref_img, detector):
    correspondence = []

    # Detect keypoints and compute descriptors
    keypoints_target, descriptors_target = detector.detectAndCompute(target_img, None)
    keypoints_reference, descriptors_reference = detector.detectAndCompute(ref_img, None)
    FeatureMatcher(keypoints_target, keypoints_reference, descriptors_target, descriptors_reference, correspondence)
    
    return keypoints_target, keypoints_reference, correspondence

def FeatureMatcher(keypoint_target, keypoint_reference, descriptors_target, descriptors_reference, correspondence):

    # Create a Brute-Force matcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(descriptors_target, descriptors_reference)
    # Sort matches by their distance
    correspondence.extend(sorted(matches, key=lambda x: x.distance))