import cv2
import numpy as np

# Kết nối các khớp theo định dạng COCO
SKELETON = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [1, 5], [5, 6], [6, 7],
    [1, 8], [8, 9], [9, 10],
    [8, 12], [12, 13], [13, 14],
    [0, 15], [0, 16], [15, 17], [16, 18]
]

# Hàm vẽ keypoints + xương
def draw_pose(img, keypoints, kpt_thr=0.3):
    if isinstance(keypoints, list):
        keypoints = np.array(keypoints)

    for i, (x, y, score) in enumerate(keypoints):
        if score > kpt_thr:
            cv2.circle(img, (int(x), int(y)), 3, (0, 255, 0), -1)

    for pair in SKELETON:
        i1, i2 = pair
        if keypoints[i1][2] > kpt_thr and keypoints[i2][2] > kpt_thr:
            pt1 = tuple(map(int, keypoints[i1][:2]))
            pt2 = tuple(map(int, keypoints[i2][:2]))
            cv2.line(img, pt1, pt2, (255, 0, 0), 2)

    return img
