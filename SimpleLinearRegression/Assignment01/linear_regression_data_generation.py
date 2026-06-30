"""作业一：生成带噪声的线性数据并保存到 CSV 文件。

任务：完成 generate_data()，根据直线方程和噪声生成 y 值。
运行方式：python linear_regression_data_generation.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 真实直线参数：y = W_TRUE * x + B_TRUE
W_TRUE = 2.5
B_TRUE = 1.0
NOISE_STD = 1.5
N_POINTS = 100
RANDOM_SEED = 42

# CSV_PATH = "linear_regression_data.csv"
# FIG_PATH = "linear_regression_data.png"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data") 
os.makedirs(DATA_DIR,exist_ok=True)
CSV_PATH = os.path.join(DATA_DIR, "linear_regression_data.csv")
FIG_PATH = os.path.join(DATA_DIR, "linear_regression_data.png")

def generate_data(n_points=N_POINTS, w_true=W_TRUE, b_true=B_TRUE, noise_std=NOISE_STD, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)

    # TODO：
    #   1. 从 [-5, 5] 区间均匀采样 `n_points` 个 x 值。
    #   2. 采样均值为 0、标准差为 `noise_std` 的高斯噪声，共 `n_points` 个。
    #   3. 计算 y = w_true * x + b_true + noise。

    x =rng.uniform(-5,5,n_points)
    noise=rng.normal(0,noise_std,n_points)
    y =w_true*x+b_true+noise

    return x, y


def plot_data(x, y, save_path=FIG_PATH):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, alpha=0.7, edgecolors="k", linewidths=0.4, label="Noisy samples")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Linear Regression Data (Noisy Points)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def save_csv(x, y, csv_path=CSV_PATH):
    df = pd.DataFrame({"x": x, "y": y})
    df.to_csv(csv_path, index=False)


def main():
    x, y = generate_data()
    save_csv(x, y)
    plot_data(x, y)
    print(f"Saved {len(x)} points to {CSV_PATH}")
    print(f"Saved scatter plot to {FIG_PATH}")
    print(f"True parameters: w={W_TRUE}, b={B_TRUE}")


if __name__ == "__main__":
    main()
