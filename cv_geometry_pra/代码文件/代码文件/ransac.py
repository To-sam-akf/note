import numpy as np


def ransac(data, model, n, k, t, d, debug=False, return_all=False):
    """
    使用RANSAC算法拟合模型参数
    参数:
        data: 输入数据
        model: 拟合模型类
        n: 最小样本数
        k: 最大迭代次数
        t: 阈值，用于判断点是否为内点
        d: 内点数量阈值，超过此值则重新估计模型
        debug: 是否打印调试信息
        return_all: 是否返回所有信息
    返回:
        bestfit: 最佳拟合模型
        {'inliers': best_inlier_idxs}: 内点索引(当return_all=True时)
    """
    iterations = 0  # 迭代计数器
    bestfit = None  # 最佳拟合模型
    besterr = np.inf  # 最小误差
    best_inlier_idxs = None  # 最佳内点索引

    while iterations < k:
        # 随机选择数据点
        maybe_idxs, test_idxs = random_partition(n, data.shape[0])
        maybeinliers = data[maybe_idxs, :]  # 候选内点
        test_points = data[test_idxs]  # 测试点

        # TODO: 需要实现以下步骤
        maybemodel = model.fit(maybeinliers)  # 使用候选内点拟合模型
        test_err = model.get_error(test_points, maybemodel)  # 计算测试点误差
        also_idxs = test_idxs[test_err < t]  # 筛选新的内点
        alsoinliers = data[also_idxs, :]  # 新增内点

        if debug:  # 打印调试信息
            print('test_err.min()', test_err.min())
            print('test_err.max()', test_err.max())
            print('np.mean(test_err)', np.mean(test_err))
            print(f'iteration {iterations}: len(alsoinliers) = {len(alsoinliers)}')

        # 如果内点数量足够多，重新估计模型
        if len(alsoinliers) > d:
            # TODO: 使用所有内点重新估计模型并计算误差
            bettermodel = model.fit(np.concatenate((maybeinliers, alsoinliers)))
            thiserr = model.get_error(np.concatenate((maybeinliers, alsoinliers)), bettermodel).sum()

            if thiserr < besterr:  # 更新最佳结果
                bestfit = bettermodel
                besterr = thiserr
                best_inlier_idxs = np.concatenate((maybe_idxs, also_idxs))
        iterations += 1
        print(f"Iteration {iterations}: besterr = {besterr}")

    if bestfit is None:
        raise ValueError("did not meet fit acceptance criteria")

    # 返回结果
    if return_all:
        return bestfit, {'inliers': best_inlier_idxs}
    else:
        return bestfit


def random_partition(n, n_data):
    """
    随机划分数据集
    参数:
        n: 选择的数据点数量
        n_data: 总数据点数量
    返回:
        idxs1: 选中的n个点的索引
        idxs2: 剩余点的索引
    """
    all_idxs = np.arange(n_data)
    np.random.shuffle(all_idxs)  # 随机打乱索引
    idxs1 = all_idxs[:n]
    idxs2 = all_idxs[n:]
    return idxs1, idxs2


class LinearLeastSquaresModel:
    """
    最小二乘线性拟合模型类
    用于测试RANSAC算法
    """

    def __init__(self, input_columns, output_columns, debug=False):
        """
        初始化模型
        参数:
            input_columns: 输入数据的列索引
            output_columns: 输出数据的列索引
            debug: 是否打印调试信息
        """
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.debug = debug

    def fit(self, data):
        """使用最小二乘法拟合数据"""
        A = np.vstack([data[:, i] for i in self.input_columns]).T
        B = np.vstack([data[:, i] for i in self.output_columns]).T
        x, resids, rank, s = np.linalg.lstsq(A, B, rcond=None)
        return x

    def get_error(self, data, model):
        """计算每个数据点的拟合误差"""
        A = np.vstack([data[:, i] for i in self.input_columns]).T
        B = np.vstack([data[:, i] for i in self.output_columns]).T
        B_fit = np.dot(A, model)
        err_per_point = np.sum((B - B_fit) ** 2, axis=1)
        return err_per_point


def test():
    """
    测试函数：生成带有噪声和离群点的数据，
    并使用RANSAC算法进行拟合
    """
    # 生成测试数据
    n_samples = 500  # 样本数
    n_inputs = 1  # 输入维度
    n_outputs = 1  # 输出维度

    # 生成完美数据
    A_exact = 20 * np.random.random((n_samples, n_inputs))
    perfect_fit = 60 * np.random.normal(size=(n_inputs, n_outputs))
    B_exact = np.dot(A_exact, perfect_fit)

    # 添加噪声
    A_noisy = A_exact + np.random.normal(size=A_exact.shape)
    B_noisy = B_exact + np.random.normal(size=B_exact.shape)

    # 添加离群点
    n_outliers = 100
    all_idxs = np.arange(A_noisy.shape[0])
    np.random.shuffle(all_idxs)
    outlier_idxs = all_idxs[:n_outliers]
    A_noisy[outlier_idxs] = 20 * np.random.random((n_outliers, n_inputs))
    B_noisy[outlier_idxs] = 50 * np.random.normal(size=(n_outliers, n_outputs))

    # 准备数据和模型
    all_data = np.hstack((A_noisy, B_noisy))
    input_columns = list(range(n_inputs))
    output_columns = [n_inputs + i for i in range(n_outputs)]
    model = LinearLeastSquaresModel(input_columns, output_columns, debug=True)

    # 使用普通最小二乘法拟合
    linear_fit, resids, rank, s = np.linalg.lstsq(
        all_data[:, input_columns], all_data[:, output_columns], rcond=None)

    # 使用RANSAC算法拟合
    ransac_fit, ransac_data = ransac(
        all_data, model, 5, 5000, 7e4, 50, debug=True, return_all=True)

    # 可视化结果
    import matplotlib.pyplot as plt

    sort_idxs = np.argsort(A_exact[:, 0])
    A_col0_sorted = A_exact[sort_idxs]

    plt.plot(A_noisy[:, 0], B_noisy[:, 0], 'k.', label='data')  # 所有数据点
    plt.plot(A_noisy[ransac_data['inliers'], 0],
             B_noisy[ransac_data['inliers'], 0], 'bx',
             label='RANSAC data')  # RANSAC内点
    plt.plot(A_col0_sorted[:, 0],
             np.dot(A_col0_sorted, ransac_fit)[:, 0],
             label='RANSAC fit')  # RANSAC拟合结果
    plt.plot(A_col0_sorted[:, 0],
             np.dot(A_col0_sorted, perfect_fit)[:, 0],
             label='exact system')  # 真实模型
    plt.plot(A_col0_sorted[:, 0],
             np.dot(A_col0_sorted, linear_fit)[:, 0],
             label='linear fit')  # 普通最小二乘拟合
    plt.legend()
    plt.show()