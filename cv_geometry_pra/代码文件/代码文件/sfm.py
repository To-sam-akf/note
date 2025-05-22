import numpy as np
import matplotlib.pyplot as plt


# def compute_P(x: np.ndarray, X: np.ndarray) -> np.ndarray:
#     """
#     通过2D-3D点对应关系计算相机投影矩阵
#     参数:
#         x: 2D点的齐次坐标
#         X: 3D点的齐次坐标
#     返回:
#         3x4的相机投影矩阵P
#     """
#     n = x.shape[1]
#     if X.shape[1] != n:
#         raise ValueError("Number of points don't match.")
#     # 创建用于计算DLT 解的矩阵
#     M = np.zeros((3 * n, 12 + n))
#     for i in range(n):
#         # TODO 对M进行赋值
#
#     _, _, Vt = np.linalg.svd(M)
#     return Vt[-1, :12].reshape((3, 4))

def compute_P(x: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    通过2D-3D点对应关系计算相机投影矩阵
    该实现基于显式地引入尺度因子lambda_i的DLT方法。
    参数:
        x: 2D点的齐次坐标 (形状为 3xN 的 numpy array，其中N是点的数量)
           x[0, i] 是 u_i, x[1, i] 是 v_i, x[2, i] 是 w_i
        X: 3D点的齐次坐标 (形状为 4xN 的 numpy array，其中N是点的数量)
           X[0, i] 是 X_i, X[1, i] 是 Y_i, X[2, i] 是 Z_i, X[3, i] 是 W_i
    返回:
        3x4的相机投影矩阵P (numpy array)
    """
    n = x.shape[1]  # 获取点的数量
    if X.shape[1] != n:
        raise ValueError("Number of points don't match.")

    # 创建用于计算DLT 解的矩阵 M
    # M的维度是 (3n) x (12 + n)
    # 前12列对应P的12个元素 (p11...p34)，这些是主要的未知数
    # 后n列对应n个尺度因子 lambda_i (lambda_1, ..., lambda_n)，每个点对应一个lambda_i
    M = np.zeros((3 * n, 12 + n))

    for i in range(n):
        # 获取第i个2D点和3D点
        # 2D齐次坐标 (u_i, v_i, w_i)
        u_i, v_i, w_i = x[:, i]
        # 3D齐次坐标 (X_i, Y_i, Z_i, W_i)
        X_i_vec = X[:, i]

        # 构造M的第 3i, 3i+1, 3i+2 行
        # 这些行来源于 P * X_i = lambda_i * x_i，即 P * X_i - lambda_i * x_i = 0
        # 向量化形式为:
        # [X_i^T    0^T    0^T    -u_i e_i^T]
        # [0^T    X_i^T    0^T    -v_i e_i^T]
        # [0^T    0^T    X_i^T    -w_i e_i^T]
        # e_i 是一个n维向量，除了第i个位置是1，其他为0。
        # 这里为了简化，我们直接在M的相应位置赋值。

        # 对应方程: (P_row1 . X_i_vec) - lambda_i * u_i = 0
        # P_row1 的系数是 X_i_vec，对应 M 的 0-3 列
        # lambda_i 的系数是 -u_i，对应 M 的 12+i 列
        M[3 * i, 0:4] = X_i_vec
        M[3 * i, 12 + i] = -u_i

        # 对应方程: (P_row2 . X_i_vec) - lambda_i * v_i = 0
        # P_row2 的系数是 X_i_vec，对应 M 的 4-7 列
        # lambda_i 的系数是 -v_i，对应 M 的 12+i 列
        M[3 * i + 1, 4:8] = X_i_vec
        M[3 * i + 1, 12 + i] = -v_i

        # 对应方程: (P_row3 . X_i_vec) - lambda_i * w_i = 0
        # P_row3 的系数是 X_i_vec，对应 M 的 8-11 列
        # lambda_i 的系数是 -w_i，对应 M 的 12+i 列
        M[3 * i + 2, 8:12] = X_i_vec
        M[3 * i + 2, 12 + i] = -w_i

    # 对矩阵 M 进行奇异值分解 (SVD)
    # M = U S V^T
    # 齐次线性方程 M @ q = 0 的解 q 是 V 的最后一列，
    # 或者说 Vt (V的转置) 的最后一行。
    # 这个解 q 对应于 M 的最小奇异值。
    _, _, Vt = np.linalg.svd(M)

    # 提取 P 的元素并重塑为 3x4 矩阵
    # Vt[-1, :] 是解向量 q，包含了 P 的12个元素和 n 个 lambda_i 元素。
    # 我们只关心 P 的前12个元素，将它们重塑为 3x4 矩阵。
    return Vt[-1, :12].reshape((3, 4))

def triangulate_point(x1: np.ndarray, x2: np.ndarray, P1: np.ndarray, P2: np.ndarray) -> np.ndarray:
    """
    使用最小二乘法对单个点对进行三角化
    参数:
        x1, x2: 两幅图像中对应点的齐次坐标
        P1, P2: 两个相机的投影矩阵
    返回:
        重建的3D点坐标(齐次)
    """
    M = np.zeros((6, 6))
    M[:3, :4] = P1
    M[3:, :4] = P2
    M[:3, 4] = -x1
    M[3:, 5] = -x2

    _, _, Vt = np.linalg.svd(M)
    X = Vt[-1, :4]
    return X / X[3]


def triangulate(x1: np.ndarray, x2: np.ndarray, P1: np.ndarray, P2: np.ndarray) -> np.ndarray:
    """Two-view triangulation of points."""
    n = x1.shape[1]
    if x2.shape[1] != n:
        raise ValueError("Number of points don't match.")

    X = [triangulate_point(x1[:, i], x2[:, i], P1, P2) for i in range(n)]
    return np.array(X).T


def compute_fundamental(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """
    使用8点法计算基本矩阵F
    参数:
        x1, x2: 两幅图像中对应点的齐次坐标
    返回:
        3x3基本矩阵F
    步骤:
    1. 构建线性方程组
    2. 使用SVD求解
    3. 强制秩为2的约束
    4. 归一化
    """
    n = x1.shape[1]
    if x2.shape[1] != n:
        raise ValueError("Number of points don't match.")

    A = np.array([
        [x1[0, i] * x2[0, i], x1[0, i] * x2[1, i], x1[0, i] * x2[2, i],
         x1[1, i] * x2[0, i], x1[1, i] * x2[1, i], x1[1, i] * x2[2, i],
         x1[2, i] * x2[0, i], x1[2, i] * x2[1, i], x1[2, i] * x2[2, i]]
        for i in range(n)
    ])

    _, _, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)

    # Enforce rank 2
    Uf, Sf, Vtf = np.linalg.svd(F)
    Sf[2] = 0
    F = Uf @ np.diag(Sf) @ Vtf

    return F / F[2, 2]


def compute_epipole(F: np.ndarray) -> np.ndarray:
    """
    从基本矩阵计算(右)极点
    参数:
        F: 3x3基本矩阵
    返回:
        极点的齐次坐标
    """
    _, _, Vt = np.linalg.svd(F)
    e = Vt[-1]
    return e / e[2]


def plot_epipolar_line(im: np.ndarray, F: np.ndarray, x: np.ndarray, epipole=None, show_epipole=True):
    """
    在图像中绘制极线和极点
    参数:
        im: 图像
        F: 基本矩阵
        x: 点的齐次坐标
        epipole: 极点坐标(可选)
        show_epipole: 是否显示极点
    """
    m, n = im.shape[:2]
    line = F @ x

    t = np.linspace(0, n, 100)
    lt = np.array([(line[2] + line[0] * tt) / (-line[1]) for tt in t])

    valid = (lt >= 0) & (lt < m)
    plt.plot(t[valid], lt[valid], linewidth=2)

    if show_epipole:
        if epipole is None:
            epipole = compute_epipole(F)
        plt.plot(epipole[0] / epipole[2], epipole[1] / epipole[2], 'r*')


def skew(a: np.ndarray) -> np.ndarray:
    """三维向量转换为反对称矩阵（skew-symmetric matrix）"""
    return np.array([
        [0, -a[2], a[1]],
        [a[2], 0, -a[0]],
        [-a[1], a[0], 0]
    ])


def compute_P_from_fundamental(F: np.ndarray) -> np.ndarray:
    """
    假设P1=[I|0]，从基本矩阵计算第二个相机矩阵
    参数:
        F: 基本矩阵
    返回:
        第二个相机的投影矩阵P2
    """
    e = compute_epipole(F.T)
    Te = skew(e)
    return np.vstack((Te @ F.T, e)).T


def compute_P_from_essential(E: np.ndarray) -> list:
    """
    从本质矩阵计算4个可能的相机矩阵
    参数:
        E: 本质矩阵
    返回:
        4个可能的相机投影矩阵列表
    """
    U, S, Vt = np.linalg.svd(E)
    if np.linalg.det(U @ Vt) < 0:
        Vt = -Vt
    E = U @ np.diag([1, 1, 0]) @ Vt

    W = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])

    P2 = [
        np.vstack((U @ W @ Vt, U[:, 2])).T,
        np.vstack((U @ W @ Vt, -U[:, 2])).T,
        np.vstack((U @ W.T @ Vt, U[:, 2])).T,
        np.vstack((U @ W.T @ Vt, -U[:, 2])).T
    ]

    return P2


def compute_fundamental_normalized(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """标准化的8点法计算基本矩阵的函数"""
    n = x1.shape[1]
    if x2.shape[1] != n:
        raise ValueError("Number of points don't match.")

    x1 = x1 / x1[2]
    mean1 = np.mean(x1[:2], axis=1)
    S1 = np.sqrt(2) / np.std(x1[:2])
    T1 = np.array([[S1, 0, -S1 * mean1[0]],
                   [0, S1, -S1 * mean1[1]],
                   [0, 0, 1]])
    x1 = T1 @ x1

    x2 = x2 / x2[2]
    mean2 = np.mean(x2[:2], axis=1)
    S2 = np.sqrt(2) / np.std(x2[:2])
    T2 = np.array([[S2, 0, -S2 * mean2[0]],
                   [0, S2, -S2 * mean2[1]],
                   [0, 0, 1]])
    x2 = T2 @ x2

    F = compute_fundamental(x1, x2)
    F = T1.T @ F @ T2
    return F / F[2, 2]


class RansacModel:
    """
    用于RANSAC的基本矩阵模型类
    实现了:
    1. fit方法-从数据计算基本矩阵
    2. get_error方法-计算误差
    """

    def __init__(self, debug=False):
        self.debug = debug

    def fit(self, data: np.ndarray) -> np.ndarray:
        data = data.T
        x1, x2 = data[:3, :8], data[3:, :8]
        return compute_fundamental_normalized(x1, x2)

    def get_error(self, data: np.ndarray, F: np.ndarray) -> np.ndarray:
        data = data.T
        x1, x2 = data[:3], data[3:]

        Fx1 = F @ x1
        Fx2 = F.T @ x2
        denom = Fx1[0] ** 2 + Fx1[1] ** 2 + Fx2[0] ** 2 + Fx2[1] ** 2
        err = (np.diag(x1.T @ F @ x2)) ** 2 / denom
        return err


def F_from_ransac(x1: np.ndarray, x2: np.ndarray, model, maxiter=5000, match_threshold=1e-6):
    """
    使用RANSAC算法稳健估计基本矩阵
    参数:
        x1,x2: 对应点坐标
        model: RANSAC模型
        maxiter: 最大迭代次数
        match_threshold: 匹配阈值
    返回:
        基本矩阵和内点索引
    """
    import ransac
    data = np.vstack((x1, x2))
    F, ransac_data = ransac.ransac(data.T, model, 8, maxiter, match_threshold, 20, return_all=True)
    return F, ransac_data['inliers']
