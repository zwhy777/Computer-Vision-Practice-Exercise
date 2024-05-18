import cv2
import numpy as np

# 读取两幅图像
image1 = cv2.imread('image/1.jpg')
image2 = cv2.imread('image/2.jpg')

# 使用SIFT特征检测器和描述符来找到关键点和匹配
sift = cv2.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

# 使用FLANN匹配器来匹配关键点
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(descriptors1, descriptors2, k=2)

# 进行比例测试来获得良好的匹配
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

# 提取匹配点的坐标
src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# 估计
H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# 对第一幅图像应用Homography矩阵
height, width, _ = image2.shape
aligned_image = cv2.warpPerspective(image1, H, (width, height))

# 创建全景图像
panorama = cv2.addWeighted(image2, 0.5, aligned_image, 0.5, 0)

# 显示全景图像
cv2.imshow('Panorama', panorama)
cv2.waitKey(0)
cv2.destroyAllWindows()
