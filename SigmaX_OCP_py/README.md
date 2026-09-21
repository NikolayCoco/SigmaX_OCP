# SigmaX 最优控制问题梯度法求解工具箱（Python 版）

这个仓库是我自己的 MATLAB 最优控制工具箱 [`SigmaX_OCP`](../SigmaX_OCP)
的 **Python 重写版** —— 同一份工作，用另一种语言重新实现一遍。

一个自研的**直接法梯度求解器**，用于求解时间最优的含 $\sigma_x$ 驱动的量子比特重置问题
（Lindblad 主方程 + 庞特里亚金极值原理）。它把「**待求的最优控制问题**」抽象成一个接口，
把「**求解算法**」做成通用可复用组件 —— 同一个求解器无需任何修改，即可求解另一个
满足该接口的问题。

> **文档导航**（`docs/`）：
>
> | 文件 | 内容 | 什么时候读 |
> |---|---|---|
> | [`docs/05-看懂Python代码.md`](docs/05-看懂Python代码.md) | **Python 语法逐条对照 MATLAB**，用本项目代码当教材 | **看 Python 代码卡住时（最先读这个）** |
> | [`docs/01-两小时入门.md`](docs/01-两小时入门.md) | 学习路线、简历写法、面试问答 | 要把它写进简历时 |
> | [`docs/02-逐行讲解-核心.md`](docs/02-逐行讲解-核心.md) | 抽象接口 / 线搜索 / 梯度求解器 | 想深入核心算法时 |
> | [`docs/03-逐行讲解-物理.md`](docs/03-逐行讲解-物理.md) | 量子比特物理模块 | 要核对物理实现时 |
> | [`docs/04-逐行讲解-框架.md`](docs/04-逐行讲解-框架.md) | 第二个问题 / 延拓 / 可视化 / 自检 | 想了解整体架构时 |
>
> 算法与物理的**原理推导**见 MATLAB 工程的 `descriptor/main.tex`（xelatex 编译）。

---

## 0. 这个项目是什么

`SigmaX_OCP` 是我自己写的一个 MATLAB 最优控制工具箱（约 1100 行）。
为了把它迁到 Python 生态、同时给自己的 Python 学习一个真实项目，
我按 **1:1 的结构**把它重写成了 Python：

| MATLAB | Python | 行数 |
|---|---|---|
| `src/+ocp/AbstractOCP.m` | `sigmax_ocp/ocp/abstract_ocp.py` | 66 → 150 |
| `src/+ocp/QubitReset.m` | `sigmax_ocp/ocp/qubit_reset.py` | 335 → 520 |
| `src/+ocp/ControlExample.m` | `sigmax_ocp/ocp/control_example.py` | 96 → 160 |
| `src/+solver/GradientSolver.m` | `sigmax_ocp/solver/gradient_solver.py` | 214 → 330 |
| `src/+solver/LineSearch.m` | `sigmax_ocp/solver/line_search.py` | 83 → 150 |
| `src/+vis/Visualizer.m` | `sigmax_ocp/vis/visualizer.py` | 185 → 260 |
| `src/+continuation/HomotopyContinuation.m` | `sigmax_ocp/continuation/homotopy_continuation.py` | 85 → 170 |
| `demo/demo_*.m` × 4 | `demo/demo_*.py` × 4 | — |
| `tests/validate_suite.m` | `tests/validate_suite.py` | 77 → 130 |

**数值层面与 MATLAB 完全对齐**（见文末"验证结果"）。

---

## 1. 环境要求

- **Python 3.9 及以上**（本机在 3.13.15 上验证通过）
- **numpy**（本机 2.5.3）
- **matplotlib**（本机 3.11.2，仅绘图/动画需要）
- **Pillow**（随 matplotlib 自动安装，导出 GIF 需要）

安装依赖：

```powershell
python -m pip install numpy matplotlib
# 国内网络可换镜像加速：
python -m pip install numpy matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> ⚠️ **numpy 2.x 的重要变化**：`np.trapz` 在 numpy 2.0 已被**移除**，改名为
> `np.trapezoid`。MATLAB 的 `trapz(tg, y)` 在 Python 里必须写成
> `np.trapezoid(y, tg)`（注意**参数顺序也反过来了**：先函数值、后坐标）。
> 本项目在 `AbstractOCP.trapezoid()` 里做了统一封装，全项目都走它。

---

## 2. 快速开始

```powershell
# 在项目根目录（本文件所在目录）下：
python tests/validate_suite.py     # 自检，确认重写正确
python demo/demo_gradient.py       # 梯度法求解 + 结果绘图
python demo/demo_prolong.py        # 同伦延拓求解（tau 或 kappa）
python demo/demo_animation.py      # 边求解边录制动图 GIF
python demo/demo_example1.py       # 换一个完全不同的可替换问题
python demo/demo_qutip.py          # QuTiP 入门 + 用矩阵指数交叉验证积分器（需 pip install qutip）
python tests/verify_qutip_dynamics.py  # QuTiP 动力学版 vs 手写 RK4 版（精度/性能对比）
```

最小可用示例：

```python
import sys; sys.path.insert(0, r"路径\SigmaX_OCP_py")

from sigmax_ocp import ocp, solver

# 1) 建模型（要解决的 OCP）
model = ocp.QubitReset(tau=0.15, N=100, gamma=10, lambda_max=[3, 2])

# 2) 配置求解器
s = solver.GradientSolver(model)
s.direction = "gradient"
s.line_search = "armijo"
s.max_iter = 2000
s.tol_grad = 1e-5

# 3) 求解
res = s.solve()
print(f"J = {res.J:.6e}, converged = {res.converged} ({res.reason})")
```

> 💡 **强烈建议**：调用 `s.solve()` 时显式传入时间格点，例如
> `tg = model.time_grid(model.N); res = s.solve(tg=tg)`。
> 原因见第 7 节「重写中发现的 MATLAB 版缺陷」。

---

## 3. 项目结构

```
SigmaX_OCP_py/
  sigmax_ocp/                      ← 等价于 MATLAB 的 src/
    __init__.py
    ocp/                           ← 等价于 +ocp
      abstract_ocp.py              抽象接口（可替换问题的契约）
      qubit_reset.py               σx 量子比特重置问题（核心，手写 RK4 动力学）
      qubit_reset_qutip.py         同一问题，动力学改用 QuTiP（可选，需 pip install qutip）
      control_example.py           另一个可替换问题（descriptor 例1）
    solver/                        ← 等价于 +solver
      gradient_solver.py           直接法梯度求解器
      line_search.py               非精确步长搜索
    vis/                           ← 等价于 +vis
      visualizer.py                迭代四联图 + 动画（GIF/MP4）
    continuation/                  ← 等价于 +continuation
      homotopy_continuation.py     同伦延拓外壳
  demo/                            demo_*.py（4 个示例脚本）
  tests/                           validate_suite.py（自检）
  docs/                            中文讲解文档
```

Python 没有 MATLAB 的 `+package` 目录语法，所以用**子包**来对应：
`src/+ocp/` → `sigmax_ocp/ocp/`。调用方式由 `ocp.QubitReset(...)`
变为 `from sigmax_ocp import ocp; ocp.QubitReset(...)`。

---

## 4. 问题接口（`AbstractOCP`）

任何最优控制问题只需继承 `AbstractOCP` 并实现下列方法，就能被同一个求解器求解：

| 方法 | 含义 | 返回形状 |
|---|---|---|
| `time_grid(N)` | 时间格点 | `(N,)` |
| `initial_state()` | 初态 | `(n_state,)` |
| `initial_control(tg)` | 初始控制的猜测 | `(n_control, N)` |
| `cost_gradient(tg, U)` | 成本 `J`、状态 `X`、协态 `Mu`、梯度 `G` | `float, (n_state,N), (n_state,N), (n_control,N)` |
| `project(U)` | 把控制投影到可行域 | `(n_control, N)` |
| `control_bounds()` | 控制上下界 `(Ulo, Uhi)` | 各 `(n_control,)` |

属性：`n_state`、`n_control`、`name`、`tau`、`homotopy_value`、`homotopy_name`。

> **可替换性验证**：`ocp.ControlExample`（descriptor 例1）是一个标量状态/控制、
> 带运行成本、与量子比特完全无关的问题，但 `GradientSolver` **零修改**即可求解它。
> `demo/demo_example1.py` 演示了这一点。

MATLAB 与 Python 的写法对照：

```matlab
% MATLAB：抽象类 + 抽象属性
classdef (Abstract) AbstractOCP < handle
    properties (Abstract)
        nState
    end
    methods (Abstract)
        tg = timeGrid(obj, N)
    end
end
```

```python
# Python：ABC + @abstractmethod（不实现就实例化不了）
from abc import ABC, abstractmethod

class AbstractOCP(ABC):
    n_state: int = 0

    @abstractmethod
    def time_grid(self, N=None): ...
```

---

## 5. 求解器参数（`GradientSolver`）

```python
s = solver.GradientSolver(model)
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `direction` | `"gradient"` | 方向优化：`gradient`/`adagrad`/`rmsprop`/`momentum`/`nestorov`/`adam` |
| `line_search` | `"armijo"` | 步长准则：`simple`/`armijo`/`wolfe` |
| `interpolation` | `"bisection"` | 步长折半方式 |
| `use_projection` | `True` | 每步把控制投影回可行域（比罚函数稳定） |
| `kkt_active_set` | `False` | 投影梯度/KKT 活动集：在到达边界的控制分量处把向外推的梯度置零 |
| `max_iter` | `1000` | 最大迭代 |
| `min_iter` | `1` | 最小迭代 |
| `tol_grad` | `1e-4` | 梯度范数收敛阈值 |
| `tol_rel` | `1e-6` | 成本相对变化阈值 |
| `beta1`/`beta2` | `0.9`/`0.999` | 动量/二阶矩衰减（momentum/adam 用） |
| `rho`/`sigma` | `0.1`/`0.4` | Armijo/Wolfe 参数 |
| `alpha0`/`alpha_min`/`alpha_max` | `0.5`/`1e-8`/`1.0` | 初始/最小/最大步长 |
| `N` | `100` | 时间离散点数（默认网格） |
| `observer` | `None` | 可选回调 `observer(iter, J, X, Mu, U, tg)`，每轮迭代调用（用于可视化） |
| `verbosity` | `1` | 打印进度 |

### 5.1 求解结果 `res`

```python
res.U             # (n_control, N)  最优控制
res.J             # 最优成本
res.X, res.Mu     # (n_state, N)    末次状态/协态
res.G             # (n_control, N)  最优控制处的梯度
res.hist.J        # 每轮成本（list）
res.hist.gnorm    # 每轮梯度范数（list）
res.hist.alpha    # 每轮所用步长（list，与 hist.J 下标对齐）
res.converged     # 是否收敛（bool）
res.reason        # 收敛原因（str）
res.iterations    # 实际迭代数
```

### 5.2 选择方向优化（跨越鞍点/高原区）

纯梯度法收敛慢，在高原区易停滞；历史累计方向能显著改善。本机实测
（`tau=0.15, N=50, gamma=10`，60 步）：

| direction | 初始 J | 60 步后 J |
|---|---|---|
| `gradient` | 3.1568e-01 | 1.7508e-01 |
| `adam` | 3.1568e-01 | **5.2739e-02** |

---

## 6. 同伦延拓（prolong）

`HomotopyContinuation` 是**独立于具体问题**的外壳：它沿一条参数路径延拓求解，
每一步以上一步的解作为热启动，从而解决**初值敏感 / 梯度消失**。
两种延拓族由 `model.homotopy_name` 选择：

```python
import numpy as np
from sigmax_ocp import ocp, continuation

# 方式一：时域延拓（tau 从短到长）
model = ocp.QubitReset(tau=1.0, N=80, homotopy_name="tau")
c = continuation.HomotopyContinuation(model, np.linspace(0.09, 1.0, 12))
c.solver.direction = "gradient"
c.solver.max_iter = 800
r = c.run()

# 方式二：成本权重延拓（kappa 从 0 到 1，含正则化运行项）
model = ocp.QubitReset(tau=0.15, N=80, homotopy_name="kappa")
c = continuation.HomotopyContinuation(model, np.linspace(0.2, 1, 8))
r = c.run()

print(r.U, r.J)                          # 最终解
print([st.J for st in r.results])        # 各步成本
```

---

## 7. 重写中发现的 MATLAB 版缺陷（重要）

> **说明**：MATLAB 版和本 Python 版出自**同一作者**。所以这一节记录的其实是
> **一次对自己旧代码的复查** —— 这也说明"重写"本身就是一种有效的代码审查手段：
> 逐行重写会逼你把每一处隐含假设都显式化，原来被掩盖的问题自然就浮出来了。

### 7.1 ∂p₀/∂θ 的一处公式错误 —— **唯一影响数值正确性的缺陷**

> 这一条是通过**数值微分逐点核对**发现的（见 `tests/gradient_check.py`），
> 也是全部缺陷中唯一会让结果算错的一个。

`QubitReset.matVecTheta`（Python 版为 `_mat_vec_theta`）里，向量 $\vb*{p}_0$
对 $\theta$ 的偏导**第二行**有一处几何因子写错：

| 数学上应为 | 规范写法 | MATLAB 版写法 | 判定 |
|---|---|---|---|
| $\frac{d}{d\theta}[\sin\theta(\cos\theta-2)]=\cos^2\theta-2\cos\theta-\sin^2\theta$ | `c²-2c-s²` | `s.^2-c.^2-2.*c` | ❌ **错** |
| $\frac{d}{d\theta}[\sin\theta(\cos\theta+2)]=\cos^2\theta+2\cos\theta-\sin^2\theta$ | `c²+2c-s²` | `c.^2-s.^2+2.*c` | ✅ 恰好等价 |

两项形式看似对称，但**只有第二项在加法交换下恰好等价**；第一项差了
$-2(\cos^2\theta-\sin^2\theta)$。

**为什么长期没被发现**：这处偏差只影响 $\partial J/\partial\theta$ 分量的量级，
不改变下降方向的大方向，所以求解器照样收敛，只是收敛得更差。

**修复效果**（本机实测，N=50、60 步）：

| 指标 | 修复前 | 修复后 | 变化 |
|---|---|---|---|
| 纯梯度法最终成本 | 1.7508e-01 | **1.1402e-01** | **↓ 35%** |
| Adam 最终成本 | 5.2739e-02 | 5.2581e-02 | ↓ 0.3% |

纯梯度法对梯度精度最敏感，因此改善最明显；Adam 的自适应步长掩盖了大部分误差。

> ✅ **MATLAB 版已同步修复**：`src/+ocp/QubitReset.m` 里的 `matVecTheta`
> （Gp 几何因子）与 `initialState`（忽略 `pInitial`）两处都已改。
> 两边算法因此重新完全一致，**"逐项对齐"继续成立** —— 只是对齐到的是
> **修复后**的数值。
>
> **可验证的预测**：在 MATLAB R2024a 上重跑
> `solverDecreases(ocp.QubitReset('tau',0.15,'N',50), 60, 'qubit')`，
> 应当得到 `Jf ≈ 1.1402e-01`（与 Python 侧相同），而不是原来的 `1.7508e-01`。

### 7.2 `solve()` 用的是求解器的 `N`，不是模型的 `N`

MATLAB 版：

```matlab
function res = solve(obj, U0, tg)
    if nargin < 3
        tg = model.timeGrid(obj.N);   % ← obj 是 solver，用的是 solver.N
    end
```

`GradientSolver.N` 默认 100，与 `model.N` 是两个**互相独立**的属性。
于是 MATLAB 版 `demo/demo_example1.m` 里：

```matlab
model = ocp.ControlExample('N', 200);
tg    = model.timeGrid(model.N);   % 200 个点
res   = s.solve();                 % 内部却按 solver.N = 100 个点求解
plot(tg, res.U, ...)               % ← 200 对 100，MATLAB 直接报错
```

**MATLAB 版这个 demo 是跑不通的**（会报 `Vectors must be the same length`）。

Python 版的处理：
- **核心库保持与 MATLAB 完全相同的语义**（用 `solver.N`），这样数值才能逐项对齐；
- 在各 demo 里**显式传入 `tg`**（`s.solve(tg=tg)`），把这个坑规避掉；
- 在 `gradient_solver.py` 对应位置留了醒目注释。

> 你自己写代码时请记住：**`s.solve()` 要么显式传 `tg`，要么把 `s.N` 与 `model.N` 对齐。**

### 7.3 `hist.alpha` 与 `hist.J` 长度不一致

MATLAB 版每轮都记录 `hist.J`，但只在"没被 `break` 掉"的那一轮才记录
`hist.alpha`，导致两者长度差 1（或更多）。Python 版改为**下标对齐**：
`hist.alpha[k]` 就是产生 `hist.J[k]` 的那一轮所用步长；若该轮提前收敛退出
（没走线搜索），则该位置为 `None`。这只影响记录方式，不影响数值。

### 7.4 收敛原因的判定

MATLAB 用 `if iter == obj.maxIter, reason = 'max iteration'; end`，
在"恰好第 `maxIter` 轮才收敛"时会误报为 `max iteration`。
Python 版用 `for ... else` 判定"是否跑满迭代"，语义更准确。

### 7.5 MATLAB `round` vs Python `round`

`QubitReset.initial_control` 里用 `round(n/2)` 决定抛物段与常数段的拼接位置。
MATLAB 的 `round` 遇 `.5` **向远离 0 的方向进位**，而 Python 内置 `round` 是
**银行家舍入**（`round(2.5) == 2`）。当 `N` 为奇数时两者结果不同，会改变初始控制。
Python 版用 `_matlab_round()` 显式模拟 MATLAB 行为。

> 顺带一提：`ControlExample.cost_gradient` 的 MATLAB 版里定义了
> `fx = @(x) -2*x;` 却从未使用（死代码），Python 版已略去。

---

## 8. 迭代可视化 / 动画

```python
from sigmax_ocp import ocp, solver, vis

model = ocp.QubitReset(tau=1.5, N=60, gamma=10)
tg = model.time_grid(model.N)

v = vis.Visualizer(animate=True, format="gif",
                   video_file="iteration.gif", fps=12, show_live=False)
v.init(model, tg)

s = solver.GradientSolver(model)
s.max_iter = 150
s.observer = lambda it, J, X, Mu, U, tgrid: v.snapshot(it, J, X, Mu, U, tgrid)
res = s.solve(tg=tg)
v.finish()
```

- `format` 换成 `"mp4"` 即导出 MP4（**需要系统里有 ffmpeg**，否则会给出明确提示）；
- 四联图：状态、协态、控制、成本历史（对数轴）；曲线数量自动适配
  `n_state` / `n_control`；
- **与 MATLAB 的差异**：MATLAB 默认 `showLive = true`（弹出现场图窗），
  Python 版默认 `show_live = False`，这样在无图形界面的环境下也能正常导出动画。
  需要现场窗口时显式传 `show_live=True`。
- Python 版用纯 Agg 后端渲染帧，因此**不需要 GUI** 也能导出 GIF。

---

## 9. 自检

项目带**两个**自检脚本，分工不同：

```powershell
python tests/validate_suite.py     # 整体性自检（约 10 秒）
python tests/gradient_check.py     # 数值微分逐点核对（约 30 秒）
```

### 9.1 `validate_suite.py` —— 整体性自检

检查：①梯度是否为下降方向；②求解器成本是否下降；③Adam 方向是否更优；
④同伦延拓是否可运行。本机实测输出：

```
=== gradient OCP self check (Python) ===
[1] gradient is descent dir    : qubit=PASS example=PASS
   [qubit] gradient: J0=3.1568e-01 -> Jf=1.1402e-01 (iter 60)
   [example] gradient: J0=4.5539e+00 -> Jf=4.4937e+00 (iter 9)
[2] solver decreases cost      : qubit=PASS example=PASS
   [qubit] adam: J0=3.1568e-01 -> Jf=5.2581e-02 (iter 60)
[3] adam direction reduces cost: PASS
   [homotopy] final J=5.2388e-02 over 4 steps
[4] homotopy continuation runs : PASS
=== done (all PASS) ===
```

### 9.2 `gradient_check.py` —— 数值微分逐点核对

这是发现第 7.1 节那处公式错误的工具。它用**中心差分**逐点核对解析偏导：

```
[1] 解析偏导 vs 中心差分
    dP_th    相对误差=3.058e-10  OK
    dp0_th   相对误差=2.953e-10  OK     ← 修复前这里是 6.080e-01
    dP_la    相对误差=6.552e-09  OK
    dp0_la   相对误差=4.603e-09  OK
[2] 极坐标雅可比 vs 中心差分 ... 全部 OK
[3] 梯度检验：残差按 O(Δt) 一阶收敛（连续伴随方法的固有精度）
=== 全部通过 ===
```

> **为什么 [1] 项这么关键**：$\partial\vb*{P}/\partial\theta$、
> $\partial\vb*{p}_0/\partial\theta$、$\partial\vb*{P}/\partial\Lambda$、
> $\partial\vb*{p}_0/\partial\Lambda$ 这四个偏导**在 `descriptor/main.tex` 里
> 并没有展开推导**，是在代码里手推的 —— 正是最需要数值验证的地方。
>
> **为什么 [3] 项判据是"收敛阶"而不是固定阈值**：解析梯度来自**连续伴随
> 方程**，而有限差分给出的是**离散目标**的真实导数，两者相差 O(Δt)
> （解析式用采样点 `k` 的协态，离散伴随严格应当用 `k+1` 处的协态）。
> 这是方法层面的固有差异，不是实现错误。
> 实测残差 N=40→320 依次为 1.40e-01 → 1.54e-02，阶数 p≈1.02。

---

## 10. 验证结果：Python vs MATLAB

### 10.1 修复 7.1 的公式错误**之前**（重写完成时的状态）

| 检查项 | MATLAB R2024a 实测 | Python 版（修复前） | 结论 |
|---|---|---|---|
| [1] 梯度是下降方向 | qubit=PASS example=PASS | qubit=PASS example=PASS | ✅ 一致 |
| [2] 求解器成本下降（qubit） | 0.316 → 0.175（60步） | 3.1568e-01 → 1.7508e-01（60步） | ✅ 4 位有效数字一致 |
| [2] 求解器成本下降（example） | 4.554 → 4.494 | 4.5539e+00 → 4.4937e+00 | ✅ 一致 |
| [3] Adam 方向优化 | 0.316 → 0.053 | 3.1568e-01 → 5.2739e-02 | ✅ 一致 |
| [4] 同伦延拓运行 | kappa 4 步 J=0.052 | kappa 4 步 J=5.2388e-02 | ✅ 一致 |

**这说明重写是忠实的** —— 忠实到连那处公式错误一起复现了，所以两边"一致地错"。

> 这里有一个值得记住的教训：**只靠"和 MATLAB 对比"是发现不了 MATLAB 本身
> 错误的**。真正抓出 7.1 那个 bug 的，是**独立于任何参考实现的数值微分核对**。

### 10.2 修复**之后**

| 检查项 | MATLAB 版（修复前记录） | Python 版（修复后） | 结论 |
|---|---|---|---|
| [1] 梯度是下降方向 | PASS | PASS | ✅ |
| [2] 求解器成本下降（qubit） | 0.316 → 0.175 | 3.1568e-01 → **1.1402e-01** | ⬆ 见 10.3 |
| [2] 求解器成本下降（example） | 4.554 → 4.494 | 4.5539e+00 → 4.4937e+00 | ✅ 一致 |
| [3] Adam 方向优化 | 0.316 → 0.053 | 3.1568e-01 → 5.2581e-02 | ✅ 一致 |
| [4] 同伦延拓运行 | kappa 4 步 J=0.052 | kappa 4 步 J=5.2388e-02 | ✅ 一致 |

**差异只出现在"纯梯度法"那一行**，原因很清楚：它是唯一对梯度精度敏感的项 ——
`example` 问题不使用 $\partial\vb*{p}_0/\partial\theta$，Adam 有自适应步长兜底，
同伦延拓那 4 步以正则化项为主。

### 10.3 两边同步修复后：对齐依然成立

MATLAB 源码（`src/+ocp/QubitReset.m`）已同步修复同一处错误，
所以**两边算法重新完全一致**，"逐项对齐"继续成立 —— 只是对齐到修复后的数值。

| 检查项 | MATLAB 版（修复后，**待你验证**） | Python 版（修复后，已实测） |
|---|---|---|
| [2] 求解器成本下降（qubit，60步） | **预期 1.1402e-01** | 1.1402e-01 |
| [3] Adam 方向优化（60步） | 预期 5.2581e-02 | 5.2581e-02 |

> **可验证的预测**：在 MATLAB R2024a 上重跑 `tests/validate_suite.m`，
> 若 `[qubit] gradient` 那一行输出 `J0=3.1568e-01 -> Jf=1.1402e-01`，
> 就说明两处修改都正确、且两个实现重新严格对齐。
> （本机没有 MATLAB，所以 MATLAB 侧的这一列是预测值而非实测值 ——
> 这一点必须说清楚。）

### 性能参考（Python，本机实测）

单次 `cost_gradient` 调用耗时（等价于一次迭代的主要开销）：

| N | 单次耗时 |
|---|---|
| 100 | 9.9 ms |
| 200 | 19.5 ms |
| 500 | 48.8 ms |

据此：`N=100` 跑 1000 次迭代约 10 秒；`N=500` 的 14 步延拓（MATLAB 版规模）
需要十几分钟。想提速的话，**真正的瓶颈在 `_mat_vec` 里的标量 numpy 运算**
（不是 RK4 的函数调用开销 —— 实测把 RK4 内联只快 3%）。
进一步优化应当把 `_mat_vec` 向量化，或用 numba/numpy 批处理整个时间序列。

---

## 11. 待办 / 备注

- **MATLAB 工程的 `_source_original/`**：原始研究代码备份（含 CasADi/牛顿法的
  `code_Gradient_AD`），本次重写**未包含**；若想复用牛顿法（二阶）版本，需要额外
  安装 CasADi 并自行重写。
- **罚函数法**：descriptor 里作为原理讲解；实现上默认用更稳定的**投影梯度法**
  （`use_projection`）。
- 若要解决**另一个**最优控制问题：继承 `AbstractOCP` 即可，无需改任何求解器代码
  （见 `ocp/control_example.py` 与 `demo/demo_example1.py`）。
