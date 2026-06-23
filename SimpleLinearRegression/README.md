# 线性回归作业说明

本作业通过三个循序渐进的小任务，带领大家从零开始实现一元线性回归：
**生成数据 → 绘制损失等值线 → 用梯度下降求解参数**。
完成全部三份代码后，你将得到一份完整的线性回归学习轨迹。

---

## 一、考察范围

| 知识点 | 对应作业 |
| --- | --- |
| 一元线性回归模型 `y = w·x + b` | 作业一 |
| 高斯噪声与随机数生成 | 作业一 |
| 均方误差（MSE）损失函数 `J(w, b)` | 作业二、作业三 |
| 最小二乘法 / 正规方程（闭式解） | 作业二 |
| 多元函数的偏导数与梯度推导 | 作业三 |
| 批量梯度下降（Batch GD）的参数更新 | 作业三 |
| NumPy 向量化运算与广播机制 | 作业二、作业三 |
| matplotlib 等值线绘图 | 作业二、作业三 |

> 本作业**不考察**：多元线性回归、正则化（Ridge / Lasso）、特征归一化、随机 / 小批量梯度下降。请大家专注于上述基础内容。

---

## 二、编程要求

### 1. 通用要求（适用于所有作业）

- **Python 版本**：3.8 及以上。
- **依赖库**：见 `requirements.txt`（`numpy`、`pandas`、`matplotlib`）。请先执行 `pip install -r requirements.txt`。
- **NumPy 优先**：所有数值计算必须使用 NumPy 完成，**严禁使用 Python 原生 `for` 循环**逐元素计算损失或梯度（绘图循环除外）。
- **向量化**：涉及网格（mesh）的计算请使用 NumPy 的**广播机制**完成。
- **不要修改**文件中已提供的函数签名、`main()` 流程以及绘图相关代码；只在标有 `# TODO` 或 `raise NotImplementedError` 的位置填写代码。

### 2. 各作业具体要求

#### 作业一 `linear_regression_data_generation.py`
- 在 `generate_data()` 中完成：
  1. 从 `[-5, 5]` 区间均匀采样 `N_POINTS` 个 `x`；
  2. 采样均值为 0、标准差为 `NOISE_STD` 的高斯噪声；
  3. 按真实模型 `y = W_TRUE · x + B_TRUE + 噪声` 计算 `y`。
- 运行后应生成 `linear_regression_data.csv`（100 行）与散点图 `linear_regression_data.png`。

#### 作业二 `draw_loss_contour.py`
- 在 `compute_loss(w, b, x, y)` 中实现 \( J(w,b)=\frac{1}{2m}\sum(wx+b-y)^2 \)。
- 在 `normal_equation_solution(x, y)` 中构造设计矩阵 \( X=[x,\ \mathbf{1}] \) 并通过最小二乘求解最优 `w, b`。
- 在 `draw_contour()` 中用广播机制一次性计算整个损失网格 `Z`，**不得使用双重 `for` 循环**。
- 运行后应生成等值线图 `loss_contour.png`，图上应标出最小值点。

#### 作业三 `linear_gradient_descent.py`
- 实现 `compute_loss`（与作业二相同）。
- **自行推导并实现** `compute_gradients(w, b, x, y)`，返回 \( \partial J/\partial w \) 与 \( \partial J/\partial b \)。
- 完成 `gradient_descent()` 中每一步的参数更新：\( w \leftarrow w - \eta\,\partial J/\partial w \)，\( b \leftarrow b - \eta\,\partial J/\partial b \)。
- 用广播机制构建损失网格 `Z`（与作业二相同）。
- `normal_equation_solution` 已在本作业中提供，**无需实现**，仅用于对比。
- 运行后应生成 `gradient_descent_contour.png`，图上应同时显示梯度下降路径、起点、终点与正规方程的最优解。

---

## 三、提交要求

1. 提交文件包含三份已完成的 `.py` 脚本。
3. 每份脚本必须能**独立运行**：先运行作业一，再运行作业二、作业三。
4. 运行后生成的 `.csv` 和 `.png` 可一并提交，便于助教核对结果。

---

## 四、评分标准（参考）

| 项目 | 分值 |
| --- | --- |
| 作业一：数据生成正确（与真实参数一致） | 20% |
| 作业二：损失函数与正规方程实现正确 | 30% |
| 作业二：损失网格 `Z` 使用向量化写法 | 10% |
| 作业三：梯度推导与实现正确 | 25% |
| 作业三：梯度下降收敛至最优解附近 | 10% |
| 代码规范（命名、注释、无 `for` 循环计算损失） | 5% |

> **收敛判据**：作业三最终损失 `J` 应与正规方程给出的最优损失相差小于 `0.05`。

---

## 五、运行顺序

```bash
pip install -r requirements.txt               # 0. pip相关库，可忽略
python linear_regression_data_generation.py   # 1. 先生成数据
python draw_loss_contour.py                    # 2. 绘制损失等值线
python linear_gradient_descent.py              # 3. 梯度下降求解
```

完成上述三步后，当前目录下应出现：

- `linear_regression_data.csv`
- `linear_regression_data.png`
- `loss_contour.png`
- `gradient_descent_contour.png`

祝大家学习愉快！如有疑问，请在课程群中提问。
