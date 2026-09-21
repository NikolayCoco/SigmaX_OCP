# -*- coding: utf-8 -*-
"""qubit_reset_qutip.py —— 用 QuTiP 实现动力学的版本。

设计原则
--------
**不改动** `QubitReset`（它已经过数值验证）。本模块新增一个子类
`QubitResetQuTiP`，只**重写动力学部分**：

    _forward_integrate   前向：状态演化
    _backward_integrate  后向：协态演化

其余（契约实现、物理量、梯度的解析偏导）全部继承，保证物理不变。

为什么不能直接用 `qt.mesolve(H, rho, ...)`
-------------------------------------------
`qt.mesolve` 的常规用法要求模型写成**标准 Lindblad 形式**
（Hamiltonian + 塌缩算符）。而 `QubitReset` 的方程来自
**绝热参考系中的 TDQME**（Wang & Xu, PRA 104, 032201 (2021)），
其非齐次项带有横向分量（见 `docs/06-主方程与QuTiP的对应.md`），
**不是**实验室系标准 Lindblad。

本实现的做法：绕开"必须凑出 H 和 c_ops"这一限制，
**直接把已有的 Bloch 方程嵌成 Liouvillian superoperator**，
再交给 QuTiP 的求解器积分。这样：

  * 物理模型完全不变（用的还是同一个 P、p0）；
  * 动力学（积分）真正交给 QuTiP；
  * 结果应当与原来的 RK4 一致（可逐点对比验证）。

数学对照
--------
Bloch 基取正交归一的 {I, σx, σy, σz}/√2。密度矩阵写作

    ρ = (1/√2)·B₀ + Σ_i (r_i/√2)·B_i,   B_i = σ_i/√2

于是 `dr/dt = P r + p0` 等价于在算子空间里的 4×4 生成元

    M = [[0,   0    ],
         [p0,  P    ]]        （第 0 行全零 = 迹守恒）

再把 M 提升成作用在 ρ 上的 superoperator（16×16）：

    L = Σ_{ij} M[i,j] · |B_i⟩⟨B_j|      （Frobenius 内积意义下）

协态方程 `dμ/dt = -Aᵀμ` 没有非齐次项，其生成元是

    M_adj = [[0, 0], [0, Pᵀ]]

对应到 superoperator 之后，用 s = -t 的变量替换即可正向积分
（与原 RK4 实现的技巧一致）。
"""

from __future__ import annotations

import numpy as np

from .qubit_reset import QubitReset

try:                                    # 允许在未装 QuTiP 时 import 本模块
    import qutip as qt
    _HAS_QUTIP = True
except Exception:                       # pragma: no cover
    qt = None
    _HAS_QUTIP = False


def _require_qutip() -> None:
    if not _HAS_QUTIP:
        raise ImportError(
            "本模块需要 QuTiP。请先安装：\n"
            "    python -m pip install qutip\n"
            "（或 python -m pip install qutip -i https://pypi.tuna.tsinghua.edu.cn/simple）"
        )


# ======================================================================
# 算子基与 superoperator 构造
# ======================================================================
def _operator_basis():
    """正交归一的算子基 {I, σx, σy, σz}/√2（tr(B_i† B_j) = δ_ij）。"""
    s = 1.0 / np.sqrt(2.0)
    return [qt.qeye(2) * s, qt.sigmax() * s, qt.sigmay() * s, qt.sigmaz() * s]


def _bloch_to_superoperator(M: np.ndarray, basis) -> np.ndarray:
    """4×4 的 Bloch 生成元 -> 16×16 的 superoperator 矩阵。

    L[ρ] = Σ_k (M c)_k B_k，其中 ρ = Σ_k c_k B_k。
    用向量化写出来就是  L = Σ_{ij} M[i,j] · vec(B_i) vec(B_j)†。
    （Frobenius 向量化，按列展平。）
    """
    vecs = [b.full().flatten(order="F") for b in basis]
    L = np.zeros((4, 4), dtype=complex)
    for i in range(4):
        for j in range(4):
            if M[i, j] != 0.0:
                L += M[i, j] * np.outer(vecs[i], vecs[j].conj())
    return L


# ======================================================================
# 主类
# ======================================================================
class QubitResetQuTiP(QubitReset):
    """`QubitReset` 的 QuTiP 动力学版本。

    用法与 `QubitReset` 完全一致：

        from sigmax_ocp.ocp.qubit_reset_qutip import QubitResetQuTiP
        model = QubitResetQuTiP(tau=0.15, N=100, gamma=10, lambda_max=[3, 2])

        from sigmax_ocp import solver
        s = solver.GradientSolver(model)
        res = s.solve(tg=model.time_grid(model.N))

    为了可复现/可调，本类提供两个额外开关：

        integrate_method : 'mesolve' | 'mesolver'  （用 QuTiP 的哪个求解器）
        solver_atol/rtol : 传给 QuTiP 求解器的容差
    """

    name = "time-optimal qubit reset (sigma_x, QuTiP dynamics)"

    #: QuTiP 求解器的积分方法。
    #:
    #: ⚠️ **不要用 QuTiP 的默认值 `'adams'`。** adams 是多步法，在极短区间上
    #: 需要"预热"，在默认容差（rtol=1e-6）下会留下约 1e-6 的误差平台。
    #: 而本问题的每一段内系数是**常数**，属于"光滑、短区间"的典型情形，
    #: 高阶显式 RK 一步就能达到极高精度：
    #:
    #:     实测整条轨迹（N=40）相对分段矩阵指数（严格解）的误差：
    #:         adams  rtol=1e-6  ->  2.344e-06
    #:         dop853 rtol=1e-6  ->  1.843e-14     ← 默认容差下就够
    #:         dop853 rtol=1e-13 ->  7.772e-16
    #:
    #: 作为对比，原 RK4 在 N=40 时的截断误差是 4.475e-07。
    integrate_method: str = "dop853"

    #: 传给 QuTiP 求解器的容差（None = 用该方法的默认值）
    solver_atol: float | None = None
    solver_rtol: float | None = None

    def __init__(self, **kwargs) -> None:
        _require_qutip()
        super().__init__(**kwargs)
        self._basis = _operator_basis()

    # ==================================================================
    # 与 Bloch 向量的相互转换
    # ==================================================================
    def _rho_from_bloch(self, r: np.ndarray):
        """布洛赫向量 -> 密度矩阵（Qobj）。"""
        return (qt.qeye(2) + r[0] * qt.sigmax() + r[1] * qt.sigmay()
                + r[2] * qt.sigmaz()) / 2.0

    @staticmethod
    def _bloch_from_rho(rho) -> np.ndarray:
        """密度矩阵（Qobj）-> 布洛赫向量。"""
        return np.array([
            (rho * qt.sigmax()).tr().real,
            (rho * qt.sigmay()).tr().real,
            (rho * qt.sigmaz()).tr().real,
        ])

    # ==================================================================
    # Liouvillian 构造
    # ==================================================================
    def _bloch_generator(self, theta: float, Lam: float, adjoint: bool = False) -> np.ndarray:
        """该采样点的 4×4 Bloch 生成元。

        adjoint=False：前向，dr/dt = P r + p0   ->  M = [[0,0],[p0,P]]
        adjoint=True ：协态，dμ/ds = Pᵀ μ       ->  M = [[0,0],[0, Pᵀ]]
        """
        A, b = self._mat_vec(theta, Lam)

        M = np.zeros((4, 4), dtype=float)
        if adjoint:
            M[1:4, 1:4] = A.T
        else:
            M[1:4, 1:4] = A
            M[1:4, 0] = b
        return M

    def _superoperator(self, theta: float, Lam: float, adjoint: bool = False):
        """该采样点的 Liouvillian（Qobj superoperator）。"""
        M = self._bloch_generator(theta, Lam, adjoint=adjoint)
        Lmat = _bloch_to_superoperator(M, self._basis)
        return qt.Qobj(Lmat, dims=[[[2], [2]], [[2], [2]]])

    def _make_solver(self, L):
        """按 `integrate_method` / 容差建 QuTiP 求解器。

        ⚠️ 注意：QuTiP 5 的 `MESolver.options` 是一个 `_SolverOptions` 对象，
        **直接赋 dict 是无效的**（会被静默忽略，参数根本不生效）。
        必须在构造时用 `options={...}` 传入，或用 `options.update({...})`。
        这一点踩过坑：早先版本用 `solver.options = {...}` 赋值，
        结果一直在用默认的 adams + rtol=1e-6，留下了 1e-6 的误差平台。
        """
        opts = {"method": self.integrate_method}
        if self.solver_atol is not None:
            opts["atol"] = self.solver_atol
        if self.solver_rtol is not None:
            opts["rtol"] = self.solver_rtol
        return qt.MESolver(L, options=opts)

    # ==================================================================
    # 重写动力学：前向
    # ==================================================================
    def _forward_integrate(self, tg, theta, Lam, X0):
        """前向积分：用 QuTiP 逐段（控制分段常数）演化。

        与基类 RK4 版本保持同样的分段约定：第 k 段用 (theta[k], Lam[k])。
        """
        N = tg.size
        X = np.zeros((self.n_state, N))
        X[:, 0] = X0

        rho = self._rho_from_bloch(np.asarray(X0, dtype=float))
        for k in range(N - 1):
            L = self._superoperator(theta[k], Lam[k], adjoint=False)
            solver = self._make_solver(L)
            seg = solver.run(rho, [float(tg[k]), float(tg[k + 1])])
            rho = seg.states[-1]
            X[:, k + 1] = self._bloch_from_rho(rho)
        return X

    # ==================================================================
    # 重写动力学：后向（协态）
    # ==================================================================
    def _backward_integrate(self, tg, theta, Lam, Mu0):
        """后向积分协态方程 dμ/dt = -Aᵀμ。

        做法与基类一致：换变量 s = -t，把"向后"变成"向前"，右端取 Aᵀ。
        这里把 Aᵀ 也嵌成 Liouvillian，交给 QuTiP 积分。
        """
        N = tg.size
        Mu = np.zeros((self.n_state, N))
        Mu[:, N - 1] = Mu0

        mu = self._rho_from_bloch(np.asarray(Mu0, dtype=float))
        for k in range(N - 1, 0, -1):
            dt = float(tg[k] - tg[k - 1])
            L = self._superoperator(theta[k - 1], Lam[k - 1], adjoint=True)
            solver = self._make_solver(L)
            seg = solver.run(mu, [0.0, dt])
            mu = seg.states[-1]
            Mu[:, k - 1] = self._bloch_from_rho(mu)
        return Mu
