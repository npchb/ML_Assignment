"""作业二：绘制线性回归损失函数 J(w, b) 的等值线图。

任务：
  1. 实现 compute_loss(w, b, x, y)。
  2. 实现 normal_equation_solution(x, y) 求出最优的 (w, b)。
  3. 在 (W, B) 网格上构建损失矩阵 Z（使用向量化写法）。

在完成作业一之后运行：python draw_loss_contour.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "linear_regression_data.csv"
FIG_PATH = "loss_contour.png"


def load_data(csv_path=CSV_PATH):
    df = pd.read_csv(csv_path)
    return df["x"].to_numpy(), df["y"].to_numpy()


def compute_loss(w, b, x, y):
    """计算 J(w, b) = (1 / 2m) * sum((w*x + b - y)^2)。

    Parameters
    ----------
    w, b : float
        当前模型参数（斜率和截距）。
    x, y : np.ndarray，形状为 (m,)
        训练输入和目标值。

    Returns
    -------
    float
        带 1/(2m) 系数的均方误差损失。
    """
    # TODO：实现损失函数
    raise NotImplementedError


def normal_equation_solution(x, y):
    """用正规方程求解 [w, b]。

    Returns
    -------
    (w, b) : 由两个 float 组成的元组
        最小二乘意义下的最优斜率和截距。
    """
    # TODO：构造设计矩阵并求解正规方程
    raise NotImplementedError


def draw_contour(x, y, save_path=FIG_PATH):
    w_opt, b_opt = normal_equation_solution(x, y)
    print(f"Normal equation solution: w={w_opt:.4f}, b={b_opt:.4f}")
    print(f"Loss at optimum: J={compute_loss(w_opt, b_opt, x, y):.4f}")

    # 以正规方程的解为中心构造等值线网格。
    w_range = 1.5
    b_range = 1.5
    w_grid = np.linspace(w_opt - w_range, w_opt + w_range, 200)
    b_grid = np.linspace(b_opt - b_range, b_opt + b_range, 200)
    W, B = np.meshgrid(w_grid, b_grid)

    # TODO：计算每个网格点上的 Z[i, j] = J(W[i, j], B[i, j], x, y)。
    #       请勿使用 Python 的 for 循环；请使用 NumPy 的广播机制。
    Z = None  # 用你的代码替换此处

    fig, ax = plt.subplots(figsize=(7, 6))
    contour = ax.contour(W, B, Z, levels=20, cmap="viridis")
    ax.clabel(contour, inline=True, fontsize=8, fmt="%.2f")
    ax.plot(w_opt, b_opt, "r*", markersize=14, label=f"Minimum (w={w_opt:.2f}, b={b_opt:.2f})")
    ax.set_xlabel("w (slope)")
    ax.set_ylabel("b (intercept)")
    ax.set_title(r"Loss Contour $J(w, b) = \frac{1}{2m}\sum(w x + b - y)^2$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved contour plot to {save_path}")

    return w_opt, b_opt


def main():
    x, y = load_data()
    draw_contour(x, y)


if __name__ == "__main__":
    main()
