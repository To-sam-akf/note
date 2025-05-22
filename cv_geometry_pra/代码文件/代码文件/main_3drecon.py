import cv2
import numpy as np
import sfm
import matplotlib.pyplot as plt
import camera

def test():
    # 标定矩阵
    K = np.array([[2394, 0, 932], [0, 2398, 628], [0, 0, 1]])

    # 载入图像
    im1 = cv2.imread('./data/alcatraz1.jpg', cv2.IMREAD_GRAYSCALE)
    im2 = cv2.imread('./data/alcatraz2.jpg', cv2.IMREAD_GRAYSCALE)

    # 初始化 SIFT 检测器
    sift = cv2.SIFT_create()

    # 使用sift实例来计算特征点和描述符，TODO
    keypoints1, descriptors1 = sift.detectAndCompute(im1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(im2, None)

    # 检查是否有足够的描述符进行匹配
    if descriptors1 is None or descriptors2 is None or len(descriptors1) == 0 or len(descriptors2) == 0:
        print("Error: Not enough descriptors found in one or both images.")
        return

    # 使用 BFMatcher 进行特征匹配，TODO
    bf = cv2.BFMatcher()
    # 使用 knnMatch 找到每个描述符的 k=2 近邻匹配，用于Lowe's ratio test
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # 过滤匹配，最小距离和次小距离比的阈值为0.6
    good_matches = []
    # TODO
    for m, n in matches:
        if m.distance < 0.6 * n.distance:  # Lowe's ratio test
            good_matches.append(m)

    # 检查是否有足够的良好匹配点
    if len(good_matches) < 8:  # RANSAC for Fundamental/Essential matrix usually needs at least 8 points
        print(
            f"Error: Not enough good matches ({len(good_matches)}) found. At least 8 are usually required for epipolar geometry estimation.")
        return

    # 提取匹配点
    x1 = np.array([keypoints1[m.queryIdx].pt for m in good_matches])
    x2 = np.array([keypoints2[m.trainIdx].pt for m in good_matches])

    # 转换为齐次坐标
    x1 = np.vstack((x1.T, np.ones(x1.shape[0])))
    x2 = np.vstack((x2.T, np.ones(x2.shape[0])))

    # 归一化
    x1n = np.dot(np.linalg.inv(K),x1)
    x2n = np.dot(np.linalg.inv(K),x2)

    # 计算基础矩阵
    # 使用RANSAC 方法估计E
    model = sfm.RansacModel()
    print("Estimating essential matrix...")
    E,inliers = sfm.F_from_ransac(x1n,x2n,model)
    # 计算照相机矩阵（P2 是4 个解的列表）
    P1 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
    P2 = sfm.compute_P_from_essential(E)

    # 选取点在照相机前的解
    best_ind = 0
    max_infront = 0
    best_infront_mask = None
    for i in range(4):
        # 三角剖分正确点，并计算每个照相机的深度
        X = sfm.triangulate(x1n[:, inliers], x2n[:, inliers], P1, P2[i])
        d1 = (P1 @ X)[2]
        d2 = (P2[i] @ X)[2]
        # 判断是否在两个相机前（Z > 0）
        infront_mask = (d1 > 0) & (d2 > 0)
        num_infront = np.sum(infront_mask)

        # 选取使更多点在相机前的解
        if num_infront > max_infront:
            max_infront = num_infront
            best_ind = i
            best_infront_mask = infront_mask

    # 使用最佳 P2 重新三角化，并过滤在所有相机前的点
    X = sfm.triangulate(x1n[:, inliers], x2n[:, inliers], P1, P2[best_ind])
    X = X[:, best_infront_mask]

    # ========== 3D 点云绘图 ==========
    fig = plt.figure()
    ax = fig.add_subplot(212, projection='3d')
    ax.scatter(X[0], X[1], X[2], c='k')
    #ax.set_axis_off()
    plt.title("3D Points")

    # ========== 投影到两个图像平面 ==========
    # 初始化两个相机模型
    cam1 = camera.Camera(P1)
    cam2 = camera.Camera(P2[best_ind])

    # 投影到两个视角
    x1p = cam1.project(X)
    x2p = cam2.project(X)

    # 使用相机内参矩阵 K 归一化投影点
    x1p = K @ x1p
    x2p = K @ x2p

    # 归一化齐次坐标
    x1p /= x1p[2]
    x2p /= x2p[2]

    # 第一个图像投影结果
    ax1 = fig.add_subplot(221)
    ax1.imshow(im1, cmap='gray')
    ax1.plot(x1p[0], x1p[1], 'o', label='Projected')
    ax1.plot(x1[0], x1[1], 'r.', label='Original')
    ax1.axis('off')
    ax1.set_title("Projection on Image 1")
    ax1.legend()

    # 第二个图像投影结果
    ax2 = fig.add_subplot(222)
    ax2.imshow(im2, cmap='gray')
    ax2.plot(x2p[0], x2p[1], 'o', label='Projected')
    ax2.plot(x2[0], x2[1], 'r.', label='Original')
    ax2.axis('off')
    ax2.set_title("Projection on Image 2")
    ax2.legend()

    plt.show()

if __name__ == '__main__':
    test()