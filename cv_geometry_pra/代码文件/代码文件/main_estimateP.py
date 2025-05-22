import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import camera
import sfm


def test():
    # 载入一些图像
    im1 = np.array(Image.open('./data/images/001.jpg'))
    im2 = np.array(Image.open('./data/images/002.jpg'))
    # 载入每个视图的二维点到列表中
    points2D = [np.loadtxt('data/2D/00' + str(i + 1) + '.corners').T for i in range(3)]
    # 载入三维点
    points3D = np.loadtxt('./data/3D/p3d').T
    # 载入对应
    corr = np.genfromtxt('./data/2D/nview-corners', dtype='int', missing_values='*')
    # 载入照相机矩阵到Camera 对象列表中
    P = [camera.Camera(np.loadtxt('./data/2D/00' + str(i + 1) + '.P')) for i in range(3)]

    # 将三维点转换成齐次坐标表示，并投影
    X = np.vstack((points3D, np.ones(points3D.shape[1])))
    x = P[0].project(X)
    # 在视图1 中绘制点
    fig1 = plt.figure()
    plt.imshow(im1)
    plt.plot(points2D[0][0], points2D[0][1], '*')
    plt.axis('off')
    fig2 = plt.figure()
    plt.imshow(im1)
    plt.plot(x[0], x[1], 'r.')
    plt.axis('off')

    corr = corr[:, 0]  # 视图1
    ndx3D = np.where(corr >= 0)[0]  # 丢失的数值为-1
    ndx2D = corr[ndx3D]

    # 选取可见点，并用齐次坐标表示
    x = points2D[0][:, ndx2D]  # 视图1
    x = np.vstack((x, np.ones(x.shape[1])))
    X = points3D[:, ndx3D]
    X = np.vstack((X, np.ones(X.shape[1])))
    # 估计P, 并分解成K, R, t, TODO
    Pest = camera.Camera(sfm.compute_P(x, X))
    # 打印形状
    Kest, Rest, Test = Pest.factor()
    print(Kest.shape, Rest.shape, Test.shape)
    Test = Test.reshape(3,1)  # 将Test从(3,)转换为(3,1)
    print(Kest@np.hstack((Rest, Test)))
    Pest = camera.Camera(Kest @ np.hstack((Rest, Test)))
    # 比较！
    print(Pest.P / Pest.P[2, 3])
    print(P[0].P / P[0].P[2, 3])
    xest = Pest.project(X)
    # 绘制图像
    fig3 = plt.figure()
    plt.imshow(im1)
    plt.plot(x[0], x[1], 'bo')
    plt.plot(xest[0], xest[1], 'r.')
    plt.axis('off')

    plt.show()


if __name__ == '__main__':
    test()
