# 主方程与 QuTiP 的对应关系

> 本文档记录 **QubitReset 的 $P$、$\vec{p}_0$ 与标准 Lindblad 形式的对应分析**。
> 分析脚本：[`tests/analysis_p_matrix.py`](../tests/analysis_p_matrix.py)、
> [`tests/verify_lindblad_form.py`](../tests/verify_lindblad_form.py)

---

## 0. 为什么需要这个分析

QuTiP 的 `mesolve` 解的是**标准 Lindblad 方程**：

$$
\frac{d\rho}{dt} = -i[H,\rho] + \sum_k \left(L_k\rho L_k^\dagger - \tfrac12\{L_k^\dagger L_k,\rho\}\right)
$$

而 `QubitReset` 给出的是**离散化后的 3 维实数 ODE**：

$$
\frac{d\vec{p}}{dt} = \vb*{P}(\theta,\Lambda)\,\vec{p} + \vec{p}_0(\theta,\Lambda)
$$

要把前者用上，就得知道后者对应哪些 $H$ 和 $L_k$。

**分析方法**：任何线性生成元都能唯一分解成对称 + 反对称两部分：

$$
\vb*{P} = \underbrace{\vb*{P}_{\text{sym}}}_{\text{耗散}} + \underbrace{\vb*{P}_{\text{asym}}}_{\text{幺正演化 } -i[H,\cdot]}
$$

- **反对称部分**只可能来自 Hamiltonian（旋转），可以直接反解出 $H$；
- **对称部分**只能来自 Lindblad 的耗散项，可以拿去匹配 $\{L_k\}$。

---

## 1. 结论一：Hamiltonian 完全确定 ✅

从 $\vb*{P}_{\text{asym}} = [\vec{h}]_\times$ 反解（$[h]_\times$ 是"叉乘矩阵"）：

```python
hx = P_asym[2, 1];  hy = P_asym[0, 2];  hz = P_asym[1, 0]
```

实测（多个 $(\theta,\Lambda)$ 点，误差 **0.00e+00**）：

$$
\vec{h} = (-2\lambda_z,\; -2\lambda_x,\; 0)
\qquad\Longrightarrow\qquad
\boxed{\;H = \tfrac12\vec{h}\cdot\vec{\sigma} = -\lambda_z\,\sigma_x - \lambda_x\,\sigma_y\;}
$$

**注意两点**：

1. $h_z = 0$：**没有 $\sigma_z$ 项**，即这个坐标系里没有失谐项。
2. $\lambda_x$ 配的是 $\sigma_y$、$\lambda_z$ 配的是 $\sigma_x$ ——
   两路驱动在 Bloch 球上**正交但互换了轴**（具体取决于 $\ket{e}$、$\ket{r}$ 的定义
   和所用旋转坐标系）。

---

## 2. 结论二：耗散是"随 $\theta$ 旋转的横向算符" ✅

把对称部分的横向 $2\times2$ 块写成

$$
\vb*{S}_{xy} = -P\cdot\vb*{I} + Q\begin{pmatrix}\cos2\theta & \sin2\theta\\ \sin2\theta & -\cos2\theta\end{pmatrix}
$$

实测 **$Q - R \approx 10^{-15}$**（$R$ 是交叉项幅度）——
即各向异性部分**严格是旋转形式**，不是"固定的 $\sigma_z$ 退相位"。

这对应一个**随驱动方向 $\theta$ 旋转的横向塌缩算符**：

$$
L_{\text{rot}} = \cos\theta\,\sigma_x + \sin\theta\,\sigma_y
\quad\text{（或等价地 }\sin\theta\,\sigma_x - \cos\theta\,\sigma_y\text{，取决于 }Q\text{ 的符号）}
$$

**物理意义**：浴的耦合方向**由驱动方向决定**，而不是固定沿 $z$ 轴。

---

## 3. 结论三：齐次部分 $P$ 能精确写成 Lindblad 形式 ✅

用四个通道（率 $a,b,c,d \ge 0$）：

| 通道 | 算符 | 率 | 作用 |
|---|---|---|---|
| 旋转横向 | $\cos\theta\,\sigma_x + \sin\theta\,\sigma_y$ | $a$ | 随驱动方向旋转的退相位 |
| 固定退相位 | $\sigma_z$ | $b$ | 常规纯退相位 |
| 弛豫 | $\sigma_-$ | $c$ | $\ket{e}\to\ket{r}$ |
| 激发 | $\sigma_+$ | $d$ | $\ket{r}\to\ket{e}$ |

由四个条件定出四个率（$S_{22}$ 是 $z$ 方向的对角元、$\vec{p}_0$ 的 $z$ 分量给非齐次条件）：

$$
a = |Q|,\qquad
c+d = -S_{22} - 2a,\qquad
c-d = (\vec{p}_0)_z,\qquad
a + 2b + \tfrac{c+d}{2} = P
$$

**实测结果（用 QuTiP 的算符代数重建后逐元素对比）**：

```
θ=0.3  Λ=1.0   |P_重建 - P_原| = 7.105e-15   OK
θ=0.9  Λ=2.0   |P_重建 - P_原| = 1.421e-14   OK
θ=1.2  Λ=0.5   |P_重建 - P_原| = 5.684e-14   OK
θ=0.75 Λ=3.0   |P_重建 - P_原| = 5.684e-14   OK
θ=0.15 Λ=0.8   |P_重建 - P_原| = 3.553e-15   OK
```

**结论：$P$ 确实是严格的 Lindblad 生成元**，误差在浮点精度量级。

---

## 4. 结论四：非齐次项 $\vec{p}_0$ 的横向分量 —— **来源已找到** ✅

### 4.1 现象

这是分析中唯一"对不上"的地方：

```
θ=0.3, Λ=1.0
    你的 p0 = [+3.924060, -2.812712, -0.295520]
    重建 p0 = [+0.000000, +0.000000, -0.295520]
                 ↑ 横向分量对不上（重建恒为 0）
                                    ↑ z 分量完美匹配
```

**$z$ 分量（$=-\lambda_x$）匹配**（来自 $\sigma_\pm$ 通道的非齐次贡献），
但 **$x$、$y$ 分量对不上**。

### 4.2 为什么"标准 Lindblad"给不出来

$\sigma_x,\sigma_y,\sigma_z,\sigma_\pm$ 这些算符都是**正规的**（$LL^\dagger = L^\dagger L$），
其耗散稳态只能落在 $z$ 轴上（$|0\rangle$、$|1\rangle$ 或完全混态），
所以非齐次项**只能出现在 $z$ 方向**。

### 4.3 物理来源：**久期近似是在"绝热参考系"里做的**

`QubitReset` 的主方程出自
**Wang & Xu, Phys. Rev. A 104, 032201 (2021)**
（[arXiv:2105.00040](https://arxiv.org/abs/2105.00040)，"Nonadiabatic evolution and
thermodynamics of a time-dependent open quantum system"）。

该文的方法要点（Section II）：

1. 在**绝热参考系**中推导：用瞬时本征态 $|n(t)\rangle$ 定义变换
   $U(t)=\sum_n|\psi_n(t)\rangle\langle n(t_0)|\otimes e^{-iH_Bt}$；
2. 变换后的系统哈密顿量
   $\tilde H_S(t)=\sum_{n\neq m}\alpha_{nm}(t)|n(t_0)\rangle\langle m(t_0)|$，
   其中 $\alpha_{nm}(t)=-ie^{-i\phi_{nm}(t)}\langle n(t)|\dot m(t)\rangle$
   —— **它只包含非绝热跃迁**，绝热极限下可忽略；
3. **关键的一句**：*"We should keep all the rotating wave and
   counter-rotating wave terms in the system–bath coupling $V$ to ensure
   the secular approximation is carried out correctly in the adiabatic
   frame of reference."*
   —— 必须保留**所有旋波与反旋波项**，久期近似才在绝热框架中正确；
4. 用 Nakajima–Zwanzig 投影算符方法导出 TDQME，
   使**非绝热跃迁与耗散在同一框架下解耦处理**。

**这就解释了两个观察结果**：

| 我的数值发现 | 物理来源 |
|---|---|
| 耗散算符**随 $\theta$ 旋转**（结论二） | 久期近似在**绝热框架**里做；瞬时本征态随驱动方向 $\theta$ 旋转 |
| 非齐次项有**横向分量**（结论四） | 浴把系统驱向"**绝热框架下的热态**"；换回实验室系就是一个**随 $\theta$ 倾斜的稳态** |

也就是说：**"横向分量"不是错误，而是"绝热框架 + 部分久期近似"的直接后果** ——
标准实验室系 Lindblad 之所以给不出来，正是因为它把那些反旋波项丢掉了。

### 4.4 对 QuTiP 使用的影响

**`qutip.mesolve` 默认对应的是"实验室系标准 Lindblad"**，
所以它**不能直接**解你这个方程 —— 这不是实现问题，而是两种主方程的框架不同。

要在 QuTiP 里复现，有两条路（见第 5 节）。

---

## 4.5 关于 $\theta\to\pi/2$ 附近的"负率"

第 3 节那套四通道分解在 $\theta\to\pi/2$（纯 $\lambda_x$ 驱动）附近会给出负的率
（例如 $\theta=1.4,\Lambda=2.5$ 时 $c<0$）。

**这不等于主方程本身有问题** —— Lindblad 生成元的**通道分解不唯一**，
我固定选了"旋转横向 + $\sigma_z$ + $\sigma_\pm$"这一组，
在部分参数区域它不再物理，但换一组分解仍可保持所有率非负。

真正值得追的线索是：**在绝热框架下，正确的分解应当直接用瞬时本征态的跃迁算符**，
那样对称性和正定性都会自然保持。这需要按第 4.3 节的框架重新表述。

---

## 5. 所以怎么用 QuTiP 跑你的方程

### 路线 A：直接用 ODE 求解器（**现在就能跑**）

你的方程 $\dot{\vec{p}} = \vb*{P}\vec{p}+\vec{p}_0$ 本身是完整的线性 ODE，
不需要 Lindblad 形式就能求解。见 [`demo/demo_qutip.py`](../demo/demo_qutip.py)
第 [2] 部分：已用分段矩阵指数验证你的 RK4 是**四阶收敛**（实测阶数 4.16 / 4.09 / 4.04）。

### 路线 B：用 `mesolve` —— **标准 `mesolve` 不适合**

原本的设想是"补一个非正规算符，然后用 `mesolve`"。**知道物理来源后，这条路要修正**：

`qutip.mesolve` 默认对应的是**实验室系标准 Lindblad 主方程**，
而你的方程是**绝热参考系中的 TDQME**（见 4.3 节）。两者框架不同，
横向非齐次项正是这个差异的体现，**不是"缺了一个算符"那么简单**。

硬套 `mesolve` 的做法是：用一组**非正规塌缩算符**去凑出相同的 $\vb*{P}$ 和 $\vec{p}_0$。
数学上可以做到（例如 $L=\sigma_x+i\sigma_z$ 会产生 $\sigma_y$ 方向的非齐次项），
但那样得到的算符**没有物理意义**，只是等价重写。

### 路线 C：按论文框架重建（**推荐，如果要用 QuTiP**）

既然主方程的来源清楚了，正确的做法是**按论文的框架重建**：

```python
# 1) 实验室系系统哈密顿量（我们已反解出它的形式）
H_S = -lam_z(t) * qt.sigmax() - lam_x(t) * qt.sigmay()

# 2) 对角化，得到瞬时本征态 |n(t)>  -> 数值上逐时刻算
evals, evecs = H_S.eigenstates()

# 3) 在绝热框架里写耗散：塌缩算符用"瞬时本征态之间的跃迁"
#    L_{nm}(t) = |n(t)><m(t)|   （n ≠ m）
#    这一点是论文的关键：久期近似在绝热框架中做，
#    所有旋波/反旋波项都要保留
c_ops = [qt.QobjEvo(...) for ...]   # 瞬时本征态跃迁算符

# 4) 同时别漏掉非绝热项：tilde_H_S(t) 含 <n|d m/dt> 的耦合
```

**这条路的实际价值**：
1. 它才是"用 QuTiP 跑你这个主方程"的**正确**方式；
2. 可以拿它与你现在的 $\vb*{P},\vec{p}_0$ 实现**互相对照**，
   等于给整条推导链加一道独立验证；
3. 瞬时本征态的构造可以用 QuTiP 的 `Qobj.eigenstates()` 直接做。

### 路线 D：用 QuTiP 复现论文的 LZ 例子（**最省事、收益明确**）

论文里分析了**耗散 Landau–Zener 模型**作为例子。用 QuTiP 复现那个例子，
可以：
- 确认你对论文方法的理解正确；
- 得到一个"标准答案"来对照你自己的实现；
- 顺便熟悉 QuTiP 的 `QobjEvo`、时间依赖求解等用法。

---

## 6. 这个分析的副产品

1. **验证了 $H$ 的形式**：$H = -\lambda_z\sigma_x - \lambda_x\sigma_y$（零误差），
   这是从你的 $P$ 严格反解出来的，可以作为独立核对。
2. **发现耗散算符随驱动方向旋转**：这不是额外假设，
   而是"久期近似在绝热框架中做"的**必然结果**。
3. **$\vec{p}_0$ 的横向分量**：同样由绝热框架的稳态倾斜导致 ——
   数值分析与论文方法自洽。
4. **由此确立了 QuTiP 的正确用法**：不该硬套标准 `mesolve`，
   而应按绝热框架重建（路线 C），或用论文的 LZ 例子做对照验证（路线 D）。
5. **边界情况**：$\theta \to \pi/2$（纯 $\lambda_x$ 驱动）附近，
   上面那套四通道分解会出现负的率（如 $\theta=1.4,\Lambda=2.5$ 时 $c<0$）——
   说明该参数区域需要换一组通道分解，或提示模型在该区域有近似失效。

---

## 7. 已实现：QuTiP 动力学版

项目里现在有两个动力学实现，**物理完全相同**（用的是同一个 $\vb*{P}$、$\vec p_0$），
只是积分方式不同：

| 类 | 位置 | 动力学 |
|---|---|---|
| `QubitReset` | `sigmax_ocp/ocp/qubit_reset.py` | 手写 RK4（固定步长） |
| `QubitResetQuTiP` | `sigmax_ocp/ocp/qubit_reset_qutip.py` | QuTiP 的 `MESolver` + superoperator |

用法完全一致：

```python
from sigmax_ocp.ocp.qubit_reset_qutip import QubitResetQuTiP
model = QubitResetQuTiP(tau=0.15, N=100, gamma=10, lambda_max=[3, 2])
res = solver.GradientSolver(model).solve(tg=model.time_grid(model.N))
```

### 7.1 实现思路

绕开"必须凑出 $H$ 和 $c\_ops$"这一限制：**直接把已有的 Bloch 方程嵌成 Liouvillian**。

Bloch 基取正交归一的 $\{I,\sigma_x,\sigma_y,\sigma_z\}/\sqrt2$。因为方程带非齐次项 $\vec p_0$，
正好对应 4×4 生成元的第 0 列：

$$
M = \begin{pmatrix} 0 & \mathbf 0 \\ \vec p_0 & \vb*{P} \end{pmatrix}
\qquad\text{（第 0 行全零 = 迹守恒）}
$$

再提升成作用在 $\rho$ 上的 superoperator（16×16）：

$$
L = \sum_{ij} M_{ij}\,|B_i\rangle\langle B_j|
$$

交给 `qt.MESolver` 积分。协态方程的生成元是

$$
M_{\text{adj}} = \begin{pmatrix} 0 & 0 \\ 0 & \vb*{P}^{\mathsf T} \end{pmatrix}
$$

（换变量 $s=-t$ 后正向积分，与原 RK4 实现的技巧一致。）

### 7.2 踩过的两个坑

**坑一：QuTiP 默认的 `adams` 会留下 $10^{-6}$ 的误差平台。**

adams 是多步法，在极短区间上需要"预热"；而本问题每段内系数是常数，
属于"光滑、短区间"的典型情形 —— 高阶显式 RK 一步就能到位。
改用 **`dop853`**：

```python
qt.MESolver(L, options={"method": "dop853"})
```

| N | RK4 误差 | QuTiP（dop853） |
|---|---|---|
| 20 | 9.350e-06 | **5.922e-12** |
| 40 | 4.475e-07 | **1.843e-14** |
| 100 | 9.660e-09 | **3.259e-16** |

（基准是分段矩阵指数给出的严格解。）

**坑二：`solver.options = {...}` 直接赋 dict 是无效的。**

QuTiP 5 的 `options` 是 `_SolverOptions` 对象，赋 dict 会被**静默忽略**，
参数根本不生效（这正是"误差平台"一开始没被消掉的原因）。
必须用 `MESolver(L, options={...})` 或 `options.update({...})`。

### 7.3 性能与选择

| N | RK4 | QuTiP | 倍数 |
|---|---|---|---|
| 20 | 0.0013 s | 0.0078 s | 5.9× |
| 100 | 0.0050 s | 0.0482 s | 9.7× |

**结论**：

- **要精度** → `QubitResetQuTiP`（高 5～7 个数量级，且随 N 仍在改善）
- **要速度** → `QubitReset`（RK4）
- 两者在位形空间内一致到 $10^{-6}$，优化轨迹最终 $J$ 差 $1.1\times10^{-6}$

验证脚本：[`tests/verify_qutip_dynamics.py`](../tests/verify_qutip_dynamics.py)。
