
import numpy as np
import cv2

def homography(img1, img2):
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des2, des1, k=2)

    # Apply ratio test
    good_draw = []
    good_without_list = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance: #0.8 = a value suggested by David G. Lowe.
            good_draw.append([m])
            good_without_list.append(m)

    # Extract location of good matches
    points1 = np.zeros((len(good_without_list), 2), dtype=np.float32)
    points2 = np.zeros((len(good_without_list), 2), dtype=np.float32)

    for i, match in enumerate(good_without_list):
        points1[i, :] = kp2[match.queryIdx].pt
        points2[i, :] = kp1[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width = img2.shape[:2]
    white_img2 = 255- np.zeros(shape=img2.shape, dtype=np.uint8)
    whiteReg = cv2.warpPerspective(white_img2, h, (width, height))
    blank_pixels_mask = np.any(whiteReg != [255, 255, 255], axis=-1)
    im2Reg = cv2.warpPerspective(img2, h, (width, height))
    
    return im2Reg, None, blank_pixels_mask

        

