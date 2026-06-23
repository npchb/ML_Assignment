"""作业三：用梯度下降求解线性回归，并在损失等值线上可视化迭代轨迹。

任务：
  1. 实现 compute_loss(w, b, x, y)。
  2. 实现 compute_gradients(w, b, x, y) —— 计算 J 对 w 和 b 的偏导数。
  3. 完成 gradient_descent() 中的参数更新循环。
  4. 构建损失矩阵 Z（与作业二相同）。

在完成作业一之后运行：python linear_gradient_descent.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data") 
#os.makedirs(DATA_DIR,exist_ok=True)
CSV_PATH = os.path.join(DATA_DIR,"linear_regression_data.csv")
FIG_PATH = os.path.join(DATA_DIR,"gradient_descent_contour.png")

# 梯度下降超参数
LEARNING_RATE = 0.05
N_ITERATIONS = 80
INIT_W = -1.0
INIT_B = 0.0


def load_data(csv_path=CSV_PATH):
    df = pd.read_csv(csv_path)
    return df["x"].to_numpy(), df["y"].to_numpy()


def compute_loss(w, b, x, y):
    """计算 J(w, b) = (1 / 2m) * sum((w*x + b - y)^2)。"""
    # TODO：实现损失函数
    return (np.sum((np.dot(x,w)+b-y)**2))/(2*x.shape[0])


def normal_equation_solution(x, y):
    """[w, b] 的闭式最小二乘解。

    在作业三中作为参考解提供，用于绘制梯度下降的对比图。
    """
    m = len(x)
    X = np.column_stack([x, np.ones(m)])
    theta = np.linalg.lstsq(X, y, rcond=None)[0]
    return theta[0], theta[1]


def compute_gradients(w, b, x, y):
    """返回偏导数 (dJ/dw, dJ/db)。"""
    # TODO：自行推导并实现 J 对 w 和 b 的梯度
    err=np.dot(x,w)+b-y

    db=np.sum(err)/x.shape[0]
    dw=x.T@(err)/x.shape[0]
    return dw,db


def gradient_descent(x, y, lr=LEARNING_RATE, n_iterations=N_ITERATIONS, init_w=INIT_W, init_b=INIT_B):
    w, b = init_w, init_b
    history = [(w, b, compute_loss(w, b, x, y))]

    for _ in range(n_iterations):
        # TODO：计算 (w, b) 处的梯度，并执行一步梯度下降
        dw,db=compute_gradients(w,b,x,y)
        w=w-lr*dw
        b=b-lr*db
        history.append((w, b, compute_loss(w, b, x, y)))

    return w, b, history


def draw_contour_with_path(x, y, history, save_path=FIG_PATH):
    w_opt, b_opt = normal_equation_solution(x, y)

    w_range = 1.5
    b_range = 1.5
    w_grid = np.linspace(w_opt - w_range, w_opt + w_range, 200)
    b_grid = np.linspace(b_opt - b_range, b_opt + b_range, 200)
    W, B = np.meshgrid(w_grid, b_grid)

    # TODO：在 (W, B) 上计算损失矩阵 Z，方法与作业二相同。
    #       请勿使用 Python 的 for 循环；请使用 NumPy 的广播机制。
    Z = (1/2)*np.mean((W[:,:,None]*x[None,None,:]+B[:,:,None]-y[None,None,:])**2,axis=2)

    w_path = [h[0] for h in history]
    b_path = [h[1] for h in history]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.contour(W, B, Z, levels=20, cmap="viridis", alpha=0.7)
    ax.plot(w_path, b_path, "r-o", markersize=4, linewidth=1.5, label="Gradient descent path")
    ax.plot(w_path[0], b_path[0], "go", markersize=10, label=f"Start (w={w_path[0]:.2f}, b={b_path[0]:.2f})")
    ax.plot(w_path[-1], b_path[-1], "r*", markersize=14, label=f"Final (w={w_path[-1]:.2f}, b={b_path[-1]:.2f})")
    ax.plot(w_opt, b_opt, "b*", markersize=14, label=f"Normal eq. (w={w_opt:.2f}, b={b_opt:.2f})")
    ax.set_xlabel("w (slope)")
    ax.set_ylabel("b (intercept)")
    ax.set_title("Gradient Descent on Loss Contour")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    x, y = load_data()
    w_final, b_final, history = gradient_descent(x, y)
    w_opt, b_opt = normal_equation_solution(x, y)

    print(f"Normal equation solution: w={w_opt:.4f}, b={b_opt:.4f}")
    print(f"Final loss: J={history[-1][2]:.4f}")
    print(f"Optimal loss: J={compute_loss(w_opt, b_opt, x, y):.4f}")

    draw_contour_with_path(x, y, history)
    print(f"Saved figure to {FIG_PATH}")


if __name__ == "__main__":
    main()
