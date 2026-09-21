# -*- coding: utf-8 -*-
"""GradientSolver —— AbstractOCP 的直接法（direct method）梯度求解器。

MATLAB 版 ``src/+solver/GradientSolver.m`` 的 Python 重写。

求解器只在（离散化的）控制上迭代，并且只通过 AbstractOCP 接口与模型对话，
因此**问题完全可替换**。可调的算法开关：

    direction       字符串  'gradient'（默认）, 'adagrad', 'rmsprop',
                            'momentum'（经典动量）, 'nestorov', 'adam'
                            —— 历史累积方向法
    line_search     字符串  'simple', 'armijo', 'wolfe'
    interpolation   字符串  'bisection'
    use_projection  布尔    每轮把控制投影回盒约束
    kkt_active_set  布尔    在起作用边界处把向外的梯度置零
                            （投影梯度 / KKT 活动集处理路径约束）

用法：
    s = GradientSolver(model)
    s.direction = 'adam'
    res = s.solve()
    # 查看 res.U, res.J, res.hist

与 MATLAB 的差异（有意修正，不影响数值）
----------------------------------------
1. ``hist.alpha`` 在 MATLAB 版里比 ``hist.J`` 短一项（因为只在没有 break 的那一轮
   才追加）。本重写让两者**下标对齐**：``hist.alpha[k]`` 就是产生 ``hist.J[k]``
   那一轮所用的步长；若该轮因收敛提前退出（没走线搜索），则为 ``None``。
2. 收敛原因用 Python 的 ``for ... else`` 判定"是否跑满迭代"，比 MATLAB 里
   ``if iter == obj.maxIter`` 更准确（后者在"恰好第 maxIter 轮收敛"时会误报）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from ..ocp.abstract_ocp import AbstractOCP
from .line_search import LineSearch, LineSearchOptions


# ======================================================================
# 结果容器：对应 MATLAB 里的两个 struct（res 和 hist）
# ======================================================================
@dataclass
class History:
    """迭代历史。对应 MATLAB 的 ``hist`` struct，字段名保持一致。"""

    J: list[float] = field(default_factory=list)              # 每轮成本
    gnorm: list[float] = field(default_factory=list)          # 每轮梯度范数
    alpha: list[Optional[float]] = field(default_factory=list)  # 每轮所用步长

    def __len__(self) -> int:
        return len(self.J)


@dataclass
class SolveResult:
    """求解结果。对应 MATLAB 的 ``res`` struct，字段名保持一致。"""

    U: np.ndarray              # n_control x N  最优控制
    J: float                   # 最优成本
    X: np.ndarray              # n_state x N    最优状态
    Mu: np.ndarray             # n_state x N    最优协态
    G: np.ndarray              # n_control x N  最优控制处的梯度
    hist: History              # 迭代历史
    converged: bool            # 是否收敛
    reason: str                # 收敛原因
    iterations: int            # 实际迭代数

    def __repr__(self) -> str:
        return (
            f"<SolveResult J={self.J:.6e} converged={self.converged} "
            f"reason='{self.reason}' iterations={self.iterations}>"
        )


class GradientSolver:
    """对任意 AbstractOCP 生效的梯度类求解器。"""

    def __init__(self, model: AbstractOCP | None = None, **kwargs) -> None:
        # ---- 对应 MATLAB properties 里的默认值 ----
        self.model: AbstractOCP | None = model
        self.direction: str = "gradient"
        self.line_search: str = "armijo"
        self.interpolation: str = "bisection"
        self.use_projection: bool = True
        self.kkt_active_set: bool = False
        self.max_iter: int = 1000
        self.min_iter: int = 1
        self.tol_grad: float = 1e-4
        self.tol_rel: float = 1e-6
        self.beta1: float = 0.9
        self.beta2: float = 0.999
        self.eps: float = 1e-8
        self.alpha0: float = 0.5
        self.alpha_min: float = 1e-8
        self.alpha_max: float = 1.0
        self.rho: float = 0.1
        self.sigma: float = 0.4
        self.N: int = 100
        self.verbosity: int = 1
        # 可选回调： observer(iter, J, X, Mu, U, tg)，每轮迭代调用一次（用于可视化）
        self.observer: Optional[Callable] = None

        # ---- 对应 MATLAB properties (Access = private) ----
        self._m: Optional[np.ndarray] = None   # 一阶矩（速度）
        self._v: Optional[np.ndarray] = None   # 二阶矩（梯度平方累积）
        self._k: int = 0                       # 迭代计数（Adam 偏差校正用）

        self._ls = LineSearch()                # 无状态的线搜索器
        self.set(**kwargs)

    # ------------------------------------------------------------------
    def set(self, **kwargs) -> "GradientSolver":
        """批量设置属性（对应 MATLAB 的 ``set(obj, varargin{:})``）。"""
        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise AttributeError(f"GradientSolver 没有属性 '{key}'")
            setattr(self, key, value)
        return self

    # ==================================================================
    # 主循环
    # ==================================================================
    def solve(
        self,
        U0: np.ndarray | None = None,
        tg: np.ndarray | None = None,
    ) -> SolveResult:
        """跑梯度迭代并返回结果。

        参数
        ----
        U0 : 可选的初始控制（n_control x N）；不给则用 model.initial_control
        tg : 可选的时间格点；不给则用 model.time_grid(self.N)
        """
        model = self.model
        if model is None:
            raise ValueError("GradientSolver 还没有绑定 model")

        # MATLAB: if nargin < 3, tg = model.timeGrid(obj.N); end
        #
        # ⚠ 注意：这里用的是**求解器自己的** self.N，而不是 model.N（与 MATLAB 语义一致）。
        #   两者默认都是 100，但如果你改了 model.N 却没同步改 solver.N，solve() 内部
        #   会按 solver.N 离散，而外面画图可能按 model.N 取格点，长度不一致就会出错。
        #   MATLAB 版 MATLAB 的 demo_example1.m 正是栽在这里（model.N=200、solver.N=100，
        #   在 plot 处报 "Vectors must be the same length"）。本重写的各 demo 通过
        #   显式传 tg 规避了这个坑。
        if tg is None:
            tg = model.time_grid(self.N)
        # MATLAB: if nargin < 2 || isempty(U0), U0 = model.initialControl(tg); end
        if U0 is None:
            U0 = model.initial_control(tg)

        # MATLAB: obj.k = 0; obj.m = []; obj.v = [];   （重置累积量）
        self._k = 0
        self._m = None
        self._v = None

        U = model.project(U0) if self.use_projection else np.array(U0, dtype=float)

        hist = History()
        converged = False
        reason = ""

        # MATLAB: bestJ = inf; bestU = U; bestX = []; bestMu = [];
        best_J = np.inf
        best_U = U
        best_X = None
        best_Mu = None
        best_G = None

        it = 0
        for it in range(1, int(self.max_iter) + 1):
            # ---- 每轮先算成本 / 状态 / 协态 / 梯度 ----
            J, X, Mu, G = model.cost_gradient(tg, U)

            # ---- KKT / 活动集（投影梯度） ----
            if self.use_projection and self.kkt_active_set:
                G = self._apply_active_set(U, G)

            # ---- 记录历史最优 ----
            if J < best_J:
                best_J, best_U, best_X, best_Mu, best_G = J, U, X, Mu, G

            # ---- 可视化回调 ----
            if self.observer is not None:
                self.observer(it, J, X, Mu, U, tg)

            # ---- 梯度范数（L2，在时间上积分） ----
            # MATLAB: gnorm = sqrt(trapz(tg, sum(G.^2, 1)));
            gnorm = np.sqrt(AbstractOCP.trapezoid(tg, np.sum(G**2, axis=0)))
            hist.J.append(float(J))
            hist.gnorm.append(float(gnorm))
            hist.alpha.append(None)      # 先占位，稍后回填本轮真实步长

            # ---- 收敛判据 1：梯度足够小 ----
            if it >= self.min_iter and gnorm < self.tol_grad:
                converged = True
                reason = "gradient small"
                break

            # ---- 方向与投影步 ----
            ddir = self._compute_direction(model, tg, U, G)
            dphi0 = AbstractOCP.trapezoid(tg, np.sum(G * ddir, axis=0))

            if dphi0 < 0:
                alpha = self._line_search_step(model, tg, U, ddir, G)
            else:
                alpha = 0.0      # 不是下降方向：不动

            U_old = U
            if self.use_projection:
                U = model.project(U + alpha * ddir)
            else:
                U = U + alpha * ddir
            hist.alpha[-1] = float(alpha)     # 回填（见文件头"与 MATLAB 的差异"）

            # ---- 收敛判据 2：步长足够小 ----
            delta = U - U_old
            step_norm = np.sqrt(AbstractOCP.trapezoid(tg, np.sum(delta**2, axis=0)))
            if it >= self.min_iter and step_norm < self.tol_grad:
                converged = True
                reason = "step small"
                break

            # ---- 收敛判据 3：成本相对变化足够小 ----
            if (
                it >= self.min_iter
                and it > 1
                and abs(hist.J[-1] - hist.J[-2]) < self.tol_rel * abs(hist.J[-2])
            ):
                converged = True
                reason = "cost unchanged"
                break

            # ---- 进度打印 ----
            if self.verbosity > 0 and (it % 50 == 0 or it == 1):
                print(
                    f"[{self.direction}] iter {it:4d}  J = {J:.6e}  "
                    f"|grad| = {gnorm:.3e}  alpha = {alpha:.3e}"
                )
        else:
            # for-else：循环完整跑完（没有 break）才会执行到这里
            reason = "max iteration"

        return SolveResult(
            U=best_U,
            J=float(best_J),
            X=best_X,
            Mu=best_Mu,
            G=best_G,
            hist=hist,
            converged=converged,
            reason=reason,
            iterations=it,
        )

    # ==================================================================
    # 私有辅助（对应 MATLAB 的 methods (Access = private)）
    # ==================================================================
    def _apply_active_set(self, U: np.ndarray, G: np.ndarray) -> np.ndarray:
        """在起作用边界处，把向外的梯度分量置零。

        对应 MATLAB 的 ``applyActiveSet``。lo/hi 是 (n_control,)，U 是
        (n_control, N)，所以这里用 [:, None] 把上下界升成列向量来广播。
        """
        lo, hi = self.model.control_bounds()
        at_up = U >= hi[:, None] - 1e-12
        at_lo = U <= lo[:, None] + 1e-12
        outward = (at_up & (G < -1e-12)) | (at_lo & (G > 1e-12))
        G = G.copy()
        G[outward] = 0.0
        return G

    def _compute_direction(
        self, model: AbstractOCP, tg: np.ndarray, U: np.ndarray, G: np.ndarray
    ) -> np.ndarray:
        """由 grad J 算出历史累积的下降方向。对应 MATLAB 的 ``computeDirection``。"""
        method = self.direction.lower()
        self._k += 1

        if method in ("grad", "gradient"):
            ddir = -G

        elif method == "adagrad":
            if self._v is None:
                self._v = np.zeros_like(G)
            self._v += G**2
            ddir = -G / (np.sqrt(self._v) + self.eps)

        elif method == "rmsprop":
            if self._v is None:
                self._v = np.zeros_like(G)
            self._v = self.beta2 * self._v + (1.0 - self.beta2) * G**2
            ddir = -G / (np.sqrt(self._v) + self.eps)

        elif method in ("momentum", "cm"):
            if self._m is None:
                self._m = np.zeros_like(G)
            self._m = self.beta1 * self._m + (1.0 - self.beta1) * G
            ddir = -self._m

        elif method in ("nestorov", "nag"):
            if self._m is None:
                self._m = np.zeros_like(G)
            U_la = U + self.beta1 * self._m              # 前瞻点
            _, _, _, G_la = model.cost_gradient(tg, U_la)  # 需要多算一次梯度
            self._m = self.beta1 * self._m + (1.0 - self.beta1) * G_la
            ddir = -self._m

        elif method == "adam":
            if self._m is None:
                self._m = np.zeros_like(G)
            if self._v is None:
                self._v = np.zeros_like(G)
            self._m = self.beta1 * self._m + (1.0 - self.beta1) * G
            self._v = self.beta2 * self._v + (1.0 - self.beta2) * G**2
            m_hat = self._m / (1.0 - self.beta1**self._k)   # 偏差校正
            v_hat = self._v / (1.0 - self.beta2**self._k)
            ddir = -m_hat / (np.sqrt(v_hat) + self.eps)

        else:
            raise ValueError(f"GradientSolver: 未知的 direction '{self.direction}'")

        return ddir

    def _line_search_step(
        self,
        model: AbstractOCP,
        tg: np.ndarray,
        U: np.ndarray,
        ddir: np.ndarray,
        G: np.ndarray,
    ) -> float:
        """用求解器参数包装 LineSearch。对应 MATLAB 的 ``lineSearchStep``。"""
        opts = LineSearchOptions(
            rho=self.rho,
            sigma=self.sigma,
            alpha0=self.alpha0,
            alpha_min=self.alpha_min,
            alpha_max=self.alpha_max,
        )
        return self._ls.search(
            self.line_search, self.interpolation, model, tg, U, ddir, G, opts
        )
