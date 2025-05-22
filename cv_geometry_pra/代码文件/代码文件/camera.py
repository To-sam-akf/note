import numpy as np
from scipy import linalg
from scipy.linalg import qr, expm


class Camera:
    """表示针孔相机的类。"""

    def __init__(self, P: np.ndarray):
        """
        初始化针孔相机，使用投影矩阵 P = K [R | t]。

        参数:
            P (np.ndarray): 3x4 的相机投影矩阵
        """
        self.P = P  # 投影矩阵
        self.K = None  # 内参矩阵
        self.R = None  # 旋转矩阵
        self.t = None  # 平移向量
        self.c = None  # 相机中心

    def project(self, X: np.ndarray) -> np.ndarray:
        """
        将 3D 齐次点 (4xN) 投影到图像平面 (3xN)。

        参数:
            X (np.ndarray): 4xN 的 3D 点矩阵（齐次坐标）

        返回:
            np.ndarray: 3xN 的 2D 投影点矩阵（齐次坐标）
        """
        x = self.P @ X  # 使用投影矩阵进行投影
        x /= x[2]  # 归一化齐次坐标（除以第三行）
        return x

    def factor(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        将投影矩阵分解为 K, R, t，使得 P = K [R | t]。

        返回:
            tuple: (K, R, t)
        """
        # P 的前三列是 M = K @ R
        M = self.P[:, :3]

        # 对 M 进行 RQ 分解，得到 K (上三角矩阵) 和 R (正交矩阵)
        # 注意: scipy.linalg.rq 返回的 K 是上三角矩阵，R 是正交矩阵
        # 这里的 K 对应相机内参，R 对应旋转矩阵
        K, R = linalg.rq(M)

        # --- 符号调整和归一化 ---
        # 1. 确保 K 的对角线元素为正（焦距必须为正）
        #    T 是一个对角矩阵，其对角线元素为 K 对应对角线元素的符号
        T = np.diag(np.sign(np.diag(K)))

        # 2. 如果 T 的行列式为负，这意味着我们在 K 和 R 中引入了一个额外的翻转
        #    通常通过翻转 K 的第二行（y轴）的符号来修正，以确保最终 R 的行列式为 1 (纯旋转)
        if np.linalg.det(T) < 0:
            T[1, 1] *= -1  # 翻转 T 的 (1,1) 元素，使得 det(T) 变为正

        # 3. 将 T 应用到 K 和 R
        #    因为 K * R = (K * T) * (T_inv * R) 且 T 是对角矩阵，T_inv = T
        self.K = K @ T
        self.R = T @ R

        # 4. 归一化 K 矩阵，使其 K[2,2] 元素为 1
        self.K /= self.K[2, 2]

        # --- 计算 t 和 c ---
        # 我们有 P = K [R | t]
        # P 的最后一列是 K @ t
        # P[:, 3] = K @ t
        # 因此 t = inv(K) @ P[:, 3]
        self.t = np.linalg.inv(self.K) @ self.P[:, 3]

        # 相机中心 c 在世界坐标系中的位置
        # 从相机坐标系到世界坐标系变换：X_world = R.T @ X_camera - R.T @ t
        # 相机中心是当 X_camera = [0, 0, 0].T 时，X_world 的值
        # 0 = R @ c + t  =>  R @ c = -t  =>  c = R.T @ -t
        self.c = -self.R.T @ self.t

        return self.K, self.R, self.t

    def center(self) -> np.ndarray:
        """
        计算并返回相机在世界坐标系中的中心。

        返回:
            np.ndarray: 3D 相机中心
        """
        if self.c is None:  # 如果相机中心未计算
            self.factor()  # 分解投影矩阵
            self.c = -self.R.T @ self.t  # 计算相机中心
        return self.c


def rotation_matrix(a: np.ndarray) -> np.ndarray:
    """
    创建一个 4x4 的 3D 旋转矩阵，用于绕轴向量 a 旋转。

    参数:
        a (np.ndarray): 旋转轴向量 (3,)

    返回:
        np.ndarray: 4x4 的旋转矩阵
    """
    R = np.eye(4)  # 初始化为单位矩阵
    skew = np.array([  # 计算旋转轴的反对称矩阵
        [0, -a[2], a[1]],
        [a[2], 0, -a[0]],
        [-a[1], a[0], 0]
    ])
    R[:3, :3] = expm(skew)  # 使用矩阵指数计算旋转矩阵
    return R


def rq(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    对矩阵进行 RQ 分解。

    参数:
        A (np.ndarray): 待分解的矩阵（通常为 3x3）

    返回:
        tuple: (R, Q)，使得 A = R @ Q
    """
    Q, R = qr(np.flipud(A).T)  # 对矩阵的倒置转置进行 QR 分解
    R = np.flipud(R.T)  # 恢复 R 的形状
    Q = Q.T  # 恢复 Q 的形状
    return R[:, ::-1], Q[::-1, :]  # 调整列和行的顺序