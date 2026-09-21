# -*- coding: utf-8 -*-
"""QubitReset —— σx/σz 驱动的时间最优量子比特重置问题。

MATLAB 版 ``src/+ocp/QubitReset.m`` 的 Python 重写。

这是梯度工具箱实际要解的那个具体最优控制问题。它实现了 AbstractOCP 契约，
所以求解器与物理完全解耦。

模型（受洛伦兹谱密度耗散浴作用的量子比特 Lindblad 主方程，退相干率 gamma，
截止频率 omega_c）：

    控制   u(t) = [lambda_x(t); lambda_z(t)]            （驱动幅度）
    theta(t)  = atan2(lambda_x, lambda_z),  Lambda(t) = |u(t)|
    状态   p(t) = [p_e; p_r; p_i]                       （3 x N）
    d p / dt = P(theta, Lambda) p + p0(theta, Lambda)

成本：
    J = |p(tau)|                              （σx 重置的 Mayer 终端项）
    可选  J = kappa|p(tau)| + (1-kappa)/tau * ∫ (u/lambda_max) dt
    其中第二种（带正则化运行项）形式用于对延拓权重 kappa 做同伦（见
    HomotopyContinuation）。

路径约束： 0 <= lambda_x <= lambda_x_max, 0 <= lambda_z <= lambda_z_max，
用**投影**来落实（比罚函数稳定；罚函数法在原理文档里讨论）。

只使用本类做模型时，只需要它的物理。要解别的问题，请改为继承 AbstractOCP。

下标对照的重要提醒
------------------
MATLAB 是 1-based，Python 是 0-based。本文件里所有循环都已整体平移，
凡是涉及"前一点/后一点"的地方都加了注释，这是重写时最容易出错的地方。
"""

from __future__ import annotations

import numpy as np

from .abstract_ocp import AbstractOCP


def _matlab_round(x: float) -> int:
    """MATLAB 的 ``round``：四舍五入，遇 .5 向远离 0 的方向进位。

    Python 内置 ``round`` 是"银行家舍入"（round(2.5) == 2），与原代码不一致，
    会改变 ``initial_control`` 里抛物线/常数段的拼接位置，所以必须显式模拟。
    """
    return int(np.floor(x + 0.5)) if x >= 0 else -int(np.floor(-x + 0.5))


class QubitReset(AbstractOCP):
    """时间最优量子比特重置问题（σx 驱动）。"""

    n_state = 3
    n_control = 2
    name = "time-optimal qubit reset (sigma_x)"

    def __init__(self, **kwargs) -> None:
        super().__init__()
        # ---- 对应 MATLAB 构造函数里的默认值 ----
        self.omega_c: float = 1.0                  # 谱密度截止频率
        self.gamma: float = 10.0                   # 退相干 / 弛豫速率
        self.lambda_max = np.array([3.0, 2.0])     # 控制上界 (2,)，对应 lambdaMax (2,1)
        self.p_initial = np.array([0.5, 0.0, 0.0])  # 初态布居 (3,)
        self.tau: float = 0.15                     # 时间区间长度
        self.N: int = 100                          # 控制采样点数
        self.homotopy_name: str = "kappa"          # 延拓权重（0 -> 1）
        self.homotopy_value: float = 1.0
        self.set(**kwargs)

    def set(self, **kwargs) -> "QubitReset":
        """按关键字批量设置属性（对应 MATLAB 的 ``set(obj, varargin{:})``）。"""
        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise AttributeError(
                    f"{type(self).__name__} 没有属性 '{key}'；"
                    f"可用属性: {sorted(k for k in vars(self) if not k.startswith('_'))}"
                )
            # 列表/元组自动转 numpy 数组，方便 QubitReset(lambda_max=[3, 2])
            if key in ("lambda_max", "p_initial"):
                value = np.asarray(value, dtype=float)
            setattr(self, key, value)
        return self

    # ==================================================================
    # AbstractOCP 契约
    # ==================================================================

    def time_grid(self, N: int | None = None) -> np.ndarray:
        """[0, tau] 上均匀分布 N 个点。MATLAB: linspace(0, obj.tau, N)"""
        if N is None:
            N = self.N
        return np.linspace(0.0, self.tau, int(N))

    def initial_state(self) -> np.ndarray:
        """初态 p(0)。

        ⚠️ **此处修正了 MATLAB 版的一处内部不一致**：

        MATLAB 的 ``initialState`` 硬编码返回 ``[0.5; 0; 0]``，**没有**读取
        ``pInitial``；而 ``costGradient`` 前向积分时用的却是 ``pInitial``。
        结果是：只要你改了 ``pInitial``，两者就会打架 ——
        ``costGradient`` 按新初态演化，``initialState`` 却仍报旧初态。

        这里改为返回 ``self.p_initial``。因为 ``p_initial`` 的默认值本来
        就是 ``[0.5, 0, 0]``，所以这个修正**不改变默认情形下的任何数值**
        （自检与各 demo 的结果完全不变），只在你自定义初态时才体现出来。

        ``.copy()`` 是必要的：避免调用方修改返回值时污染模型自身的初态。
        """
        return self.p_initial.copy()

    def initial_control(self, tg: np.ndarray) -> np.ndarray:
        """初始控制猜测：物理上合理的抛物 ramp（热启动）。

        MATLAB:
            ux  = 4*b(1)/tau^2 * tg .* (tau - tg);
            uz1 = 4*b(2)/tau^2 * tg .* (tau - tg);
            uz2 = b(2)*ones(1,n);
            uz  = [uz1(1:round(n/2))  uz2(round(n/2)+1:end)];
            U   = [ux; uz];
        """
        n = tg.size
        b = self.lambda_max
        ux = 4.0 * b[0] / self.tau**2 * tg * (self.tau - tg)
        uz1 = 4.0 * b[1] / self.tau**2 * tg * (self.tau - tg)
        uz2 = b[1] * np.ones(n)
        half = _matlab_round(n / 2.0)
        # MATLAB 1:half  -> Python [:half]；MATLAB half+1:end -> Python [half:]
        uz = np.concatenate([uz1[:half], uz2[half:]])
        return np.vstack([ux, uz])

    def control_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """0 <= u <= lambda_max。MATLAB: Ulo = zeros(2,1); Uhi = obj.lambdaMax"""
        return np.zeros(2), self.lambda_max.copy()

    def project(self, U: np.ndarray) -> np.ndarray:
        """把控制投影到盒约束 [0, lambda_max]。

        MATLAB: Uproj = max(lo, min(hi, U));
        上下界是 (n_control,)，U 是 (n_control, N)，用 [:, None] 升维广播。
        """
        lo, hi = self.control_bounds()
        return np.clip(U, lo[:, None], hi[:, None])

    def cost_gradient(
        self, tg: np.ndarray, U: np.ndarray
    ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """成本、状态、协态、以及成本对控制的梯度。"""
        # ---- 控制 -> (theta, Lambda) ----
        theta, Lam = self._control_transform(U)
        kappa = self.homotopy_value

        # ---- 前向积分状态 ----
        X = self._forward_integrate(tg, theta, Lam, self.p_initial)

        # ---- 终端成本与自然边界条件 ----
        # MATLAB: pT = X(:, end);  nrm = sqrt(sum(pT.^2));
        pT = X[:, -1]
        nrm = float(np.sqrt(np.sum(pT**2)))
        if nrm < 1e-14:
            nrm = 1e-14
        J_ter = nrm
        mu0 = -kappa * pT / nrm              # 末端协态（自然边界条件）
        Mu = self._backward_integrate(tg, theta, Lam, mu0)

        # ---- 梯度：G = grad_u J（注意这里的负号） ----
        G = -self._gradient_at(X, Mu, theta, Lam)

        if kappa < 1.0:
            # 正则化运行成本（对 kappa 做同伦时启用）
            # MATLAB: Lr = (1-kappa)/tau .* sum(U ./ lambdaMax, 1);
            lm = self.lambda_max[:, None]
            Lr = (1.0 - kappa) / self.tau * np.sum(U / lm, axis=0)
            J = kappa * J_ter + AbstractOCP.trapezoid(tg, Lr)
            # MATLAB: dLr = (1-kappa)/tau ./ lambdaMax;   -> (n_control,1)
            dLr = ((1.0 - kappa) / self.tau / self.lambda_max)[:, None]
            G = G + dLr
        else:
            J = J_ter

        return float(J), X, Mu, G

    def set_homotopy(self, v: float) -> None:
        """分派延拓参数。

        'tau'   -> 改变时间区间长度（在时间上延拓）
        'kappa' -> 改变终端成本权重 / 正则化强度
       （MATLAB 的 ``setHomotopy``）
        """
        if self.homotopy_name.lower() == "tau":
            self.tau = v
        self.homotopy_value = v

    def solve_reference(self) -> "SolveResult":  # noqa: F821  (仅用于类型提示)
        """用默认配置求解一次（对应 MATLAB 的 ``solveReference``）。"""
        from ..solver.gradient_solver import GradientSolver   # 延迟导入，避免循环

        s = GradientSolver(self)
        s.direction = "gradient"
        s.line_search = "armijo"
        return s.solve(self.initial_control(self.time_grid(self.N)))

    # ==================================================================
    # 控制变换
    # ==================================================================
    @staticmethod
    def _control_transform(U: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(lambda_x, lambda_z) -> (theta, Lambda)，极坐标变换。

        注意 theta = atan2(lambda_x, lambda_z)，即"从 z 轴量起"。
        """
        ux = U[0, :]
        uz = U[1, :]
        theta = np.arctan2(ux, uz)
        Lam = np.sqrt(ux**2 + uz**2)
        return theta, Lam

    @staticmethod
    def _transform_jacobian(
        theta: np.ndarray, Lam: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """极坐标变换的雅可比（逐点）。

        MATLAB 的 ``transformJacobian``：极坐标在 Lambda = 0 处奇异，
        按原始推导的做法，在那里把雅可比置零以避免 0/0。
        """
        invL = 1.0 / np.maximum(Lam, 1e-12)
        tx = np.cos(theta) * invL      # d theta  / d lambda_x
        tz = -np.sin(theta) * invL     # d theta  / d lambda_z
        lx = np.sin(theta)             # d Lambda / d lambda_x
        lz = np.cos(theta)             # d Lambda / d lambda_z
        small = Lam < 1e-9
        tx[small] = 0.0
        tz[small] = 0.0
        return tx, tz, lx, lz

    # ==================================================================
    # 积分器（RK4，控制分段常数）
    # ==================================================================
    def _forward_integrate(
        self, tg: np.ndarray, theta: np.ndarray, Lam: np.ndarray, X0: np.ndarray
    ) -> np.ndarray:
        """前向积分 d p/dt = P p + p0。

        MATLAB: for k = 1:N-1 ... X(:,k+1) = rk4(f, X(:,k), dt)
        Python: for k in range(N-1) ... X[:, k+1]
        """
        N = tg.size
        X = np.zeros((self.n_state, N))
        X[:, 0] = X0
        for k in range(N - 1):
            dt = tg[k + 1] - tg[k]
            A, b = self._mat_vec(theta[k], Lam[k])
            X[:, k + 1] = self._rk4(lambda x, A=A, b=b: A @ x + b, X[:, k], dt)
        return X

    def _backward_integrate(
        self, tg: np.ndarray, theta: np.ndarray, Lam: np.ndarray, Mu0: np.ndarray
    ) -> np.ndarray:
        """后向积分协态方程 d mu / dt = -A' mu（等价于在 s = -t 上向前积分）。

        MATLAB: for k = N:-1:2 ... [A,~] = matVec(theta(k-1), Lambda(k-1))
                                      Mu(:,k-1) = rk4(@(mu) A'*mu, Mu(:,k), dt)
        Python: for k in range(N-1, 0, -1)   （k 即 MATLAB 的 k-1，整体平移 1）
        """
        N = tg.size
        Mu = np.zeros((self.n_state, N))
        Mu[:, N - 1] = Mu0
        for k in range(N - 1, 0, -1):
            dt = tg[k] - tg[k - 1]
            A, _ = self._mat_vec(theta[k - 1], Lam[k - 1])
            Mu[:, k - 1] = self._rk4(lambda mu, A=A: A.T @ mu, Mu[:, k], dt)
        return Mu

    @staticmethod
    def _rk4(f, x: np.ndarray, dt: float) -> np.ndarray:
        """经典四阶 Runge-Kutta 单步（向量版）。"""
        k1 = f(x)
        k2 = f(x + 0.5 * dt * k1)
        k3 = f(x + 0.5 * dt * k2)
        k4 = f(x + dt * k3)
        return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    # ==================================================================
    # 梯度
    # ==================================================================
    def _gradient_at(
        self, X: np.ndarray, Mu: np.ndarray, theta: np.ndarray, Lam: np.ndarray
    ) -> np.ndarray:
        """组装 grad_u J。

        MATLAB 的 ``gradientAt`` 是逐点循环：

            for k = 1:N
                Hth(k) = Mu(:,k)' * ( Pth(:,:,k)*X(:,k) + p0th(:,k) );
                Hla(k) = Mu(:,k)' * ( Pla(:,:,k)*X(:,k) + p0la(:,k) );
            end

        这里用 einsum 一次性完成同样的求和（等价，但快得多；同伦延拓要调用
        它几百次）：
            '<i,k>,<i,k>->k'           = sum_i Mu[i,k] * p0[i,k]
            '<i,k>,<i,j,k>,<j,k>->k'   = sum_i sum_j Mu[i,k]*P[i,j,k]*X[j,k]
        """
        Pth, p0th = self._mat_vec_theta(theta, Lam)      # (3,3,N), (3,N)
        Pla, p0la = self._mat_vec_lambda(theta, Lam)     # (3,3,N), (3,N)

        Hth = np.einsum("ik,ik->k", Mu, p0th) + np.einsum("ik,ijk,jk->k", Mu, Pth, X)
        Hla = np.einsum("ik,ik->k", Mu, p0la) + np.einsum("ik,ijk,jk->k", Mu, Pla, X)

        tx, tz, lx, lz = self._transform_jacobian(theta, Lam)
        # MATLAB: G = [ tx.*Hth + lx.*Hla ;  tz.*Hth + lz.*Hla ];
        return np.vstack([tx * Hth + lx * Hla, tz * Hth + lz * Hla])

    # ==================================================================
    # 物理：全部是控制变换 (theta, Lambda) 的函数
    # 表达式转录自解析推导（见 descriptor）。约定 d p/dt = P(theta,Lambda) p + p0
    # ==================================================================

    def _spec(self, omega):
        """谱密度 J(omega)（洛伦兹型）。"""
        wc = self.omega_c
        return np.pi * omega * wc**2 / (omega**2 + wc**2)

    def _dspec(self, omega):
        """d J / d omega。"""
        wc = self.omega_c
        return np.pi * wc**2 * (wc**2 - omega**2) / (omega**2 + wc**2) ** 2

    # ---------------- Gamma_z ----------------
    def _gz(self, th, la):
        return np.pi * np.sin(th) ** 2

    def _gz_th(self, th, la):
        return 2.0 * np.pi * np.sin(th) * np.cos(th)

    def _gz_la(self, th, la):
        # MATLAB: 0*th  —— 保留同样的写法，标量与数组都能正确广播
        return 0.0 * th

    # ---------------- Gamma_+ ----------------
    # 注意：la = 0 时 1/tanh(la) = Inf，而 spec(2*la) = 0，于是 0*Inf = NaN；
    # MATLAB 版正是靠"把 NaN 置 0，再加上 (la==0) 的解析极限项"来处理这个奇点。
    # 因此这里必须用 errstate 让 numpy 安静地产生 NaN，而不是抛错。
    def _gp(self, th, la):
        with np.errstate(invalid="ignore", divide="ignore"):
            re = 0.5 * self._spec(2 * la) * (1.0 / np.tanh(la) + 1.0) * np.cos(th) ** 2
        re = np.where(np.isnan(re), 0.0, re)
        re = re + np.where(np.asarray(la) == 0, 1.0, 0.0) * (0.5 * np.pi * np.cos(th) ** 2)
        return re

    def _gp_th(self, th, la):
        with np.errstate(invalid="ignore", divide="ignore"):
            re = -self._spec(2 * la) * (1.0 / np.tanh(la) + 1.0) * np.cos(th) * np.sin(th)
        re = np.where(np.isnan(re), 0.0, re)
        re = re + np.where(np.asarray(la) == 0, 1.0, 0.0) * (-np.pi * np.cos(th) * np.sin(th))
        return re

    def _gp_la(self, th, la):
        with np.errstate(invalid="ignore", divide="ignore"):
            re = (
                self._dspec(2 * la) * (1.0 / np.tanh(la) + 1.0)
                - 0.5 * self._spec(2 * la) * (1.0 / np.sinh(la)) ** 2
            ) * np.cos(th) ** 2
        re = np.where(np.isnan(re), 0.0, re)
        re = re + np.where(np.asarray(la) == 0, 1.0, 0.0) * (self._dspec(0) * np.cos(th) ** 2)
        return re

    # ---------------- Gamma_- ----------------
    def _gm(self, th, la):
        with np.errstate(invalid="ignore", divide="ignore"):
            re = 0.5 * self._spec(2 * la) * (1.0 / np.tanh(la) - 1.0) * np.cos(th) ** 2
        re = np.where(np.isnan(re), 0.0, re)
        re = re + np.where(np.asarray(la) == 0, 1.0, 0.0) * (0.5 * np.pi * np.cos(th) ** 2)
        return re

    def _gm_th(self, th, la):
        with np.errstate(invalid="ignore", divide="ignore"):
            re = -self._spec(2 * la) * (1.0 / np.tanh(la) - 1.0) * np.cos(th) * np.sin(th)
        re = np.where(np.isnan(re), 0.0, re)
        re = re + np.where(np.asarray(la) == 0, 1.0, 0.0) * (-np.pi * np.cos(th) * np.sin(th))
        return re

    def _gm_la(self, th, la):
        with np.errstate(invalid="ignore", divide="ignore"):
            re = (
                self._dspec(2 * la) * (1.0 / np.tanh(la) - 1.0)
                - 0.5 * self._spec(2 * la) * (1.0 / np.sinh(la)) ** 2
            ) * np.cos(th) ** 2
        re = np.where(np.isnan(re), 0.0, re)
        re = re + np.where(np.asarray(la) == 0, 1.0, 0.0) * (-self._dspec(0) * np.cos(th) ** 2)
        return re

    # ---------------- 矩阵 P 与向量 p0（单点） ----------------
    def _mat_vec(self, th, la) -> tuple[np.ndarray, np.ndarray]:
        """在某一个采样点上给出 P (3x3) 与 p0 (3,)。"""
        g = self.gamma
        Gz = self._gz(th, la)
        Gp = self._gp(th, la)
        Gm = self._gm(th, la)
        c = np.cos(th)
        s = np.sin(th)

        A = np.zeros((3, 3))
        A[0, 0] = -g * (4 * Gz * s**2 + (Gp + Gm) * (1 + c**2))
        A[0, 1] = g * (4 * Gz - Gp - Gm) * c * s
        A[0, 2] = -2 * la * s
        A[1, 0] = g * (4 * Gz - Gp - Gm) * c * s
        A[1, 1] = -g * (4 * Gz * c**2 + (Gp + Gm) * (1 + s**2))
        A[1, 2] = 2 * la * c
        A[2, 0] = 2 * la * s
        A[2, 1] = -2 * la * c
        A[2, 2] = -g * (4 * Gz + Gp + Gm)

        b = np.array(
            [
                g * (2 * Gz * s**2 + 0.5 * (Gp * (1 - c) ** 2 + Gm * (1 + c) ** 2)),
                -g * (2 * Gz * c * s - 0.5 * (Gp * s * (c - 2) + Gm * s * (c + 2))),
                -la * s,
            ]
        )
        # MATLAB: A(isnan(A)) = 0; b(isnan(b)) = 0;
        A = np.where(np.isnan(A), 0.0, A)
        b = np.where(np.isnan(b), 0.0, b)
        return np.asarray(A, dtype=float), np.asarray(b, dtype=float)

    # ---------------- dP/dtheta 与 dp0/dtheta（全部采样点） ----------------
    def _mat_vec_theta(self, th, la) -> tuple[np.ndarray, np.ndarray]:
        """给出 dP/dtheta (3,3,N) 与 dp0/dtheta (3,N)。

        MATLAB 里用 ``A(1,1,:) = ...`` 这样的三维赋值，Python 用 ``A[0,0,:] = ...``。
        """
        g = self.gamma
        Gz = self._gz(th, la)
        Gzt = self._gz_th(th, la)
        Gp = self._gp(th, la)
        Gpt = self._gp_th(th, la)
        Gm = self._gm(th, la)
        Gmt = self._gm_th(th, la)
        c = np.cos(th)
        s = np.sin(th)

        N = np.size(th)
        A = np.zeros((3, 3, N))
        A[0, 0, :] = (
            -g * (4 * Gzt * s**2 + (Gpt + Gmt) * (1 + c**2))
            - g * (4 * Gz * 2 * s * c + (Gp + Gm) * (-2 * s * c))
        )
        A[0, 1, :] = g * (4 * Gzt - Gpt - Gmt) * c * s + g * (4 * Gz - Gp - Gm) * (c**2 - s**2)
        A[0, 2, :] = -2 * la * c
        A[1, 0, :] = g * (4 * Gzt - Gpt - Gmt) * c * s + g * (4 * Gz - Gp - Gm) * (c**2 - s**2)
        A[1, 1, :] = (
            -g * (4 * Gzt * c**2 + (Gpt + Gmt) * (1 + s**2))
            - g * (4 * Gz * (-2 * s * c) + (Gp + Gm) * (2 * s * c))
        )
        A[1, 2, :] = -2 * la * s
        A[2, 0, :] = 2 * la * c
        A[2, 1, :] = 2 * la * s
        A[2, 2, :] = -g * (4 * Gzt + Gpt + Gmt)

        b = np.vstack(
            [
                g * (2 * Gzt * s**2 + 0.5 * (Gpt * (1 - c) ** 2 + Gmt * (1 + c) ** 2))
                + g * (2 * Gz * (2 * s * c) + 0.5 * (Gp * 2 * (1 - c) * s + Gm * 2 * (1 + c) * (-s))),
                # ⚠ 此处修正了 MATLAB 版的一个真实错误（数值微分核对发现，见
                #   tests/gradient_check.py 的 [1] 项）：
                #     d/dθ[sinθ(cosθ-2)] = cos²θ - 2cosθ - sin²θ  → 因子 (c²-2c-s²)
                #   MATLAB 版误写为 (s²-c²-2c)，两者相差 -2(c²-s²)。
                #   而 d/dθ[sinθ(cosθ+2)] = cos²θ + 2cosθ - sin²θ，MATLAB 版写的
                #   (c²-s²+2c) 与之等价（加法交换），所以那一项是对的、无需改动。
                #   下面统一写成规范形式，方便逐项对照。
                -g * (2 * Gzt * c * s - 0.5 * (Gpt * s * (c - 2) + Gmt * s * (c + 2)))
                - g * (2 * Gz * (c**2 - s**2)
                       - 0.5 * (Gp * (c**2 - 2 * c - s**2) + Gm * (c**2 + 2 * c - s**2))),
                -la * c,
            ]
        )
        A = np.where(np.isnan(A), 0.0, A)
        b = np.where(np.isnan(b), 0.0, b)
        return A, b

    # ---------------- dP/dLambda 与 dp0/dLambda（全部采样点） ----------------
    def _mat_vec_lambda(self, th, la) -> tuple[np.ndarray, np.ndarray]:
        """给出 dP/dLambda (3,3,N) 与 dp0/dLambda (3,N)。"""
        g = self.gamma
        Gz = self._gz(th, la)
        Gzl = self._gz_la(th, la)
        Gp = self._gp(th, la)
        Gpl = self._gp_la(th, la)
        Gm = self._gm(th, la)
        Gml = self._gm_la(th, la)
        c = np.cos(th)
        s = np.sin(th)

        N = np.size(th)
        A = np.zeros((3, 3, N))
        A[0, 0, :] = -g * (4 * Gzl * s**2 + (Gpl + Gml) * (1 + c**2))
        A[0, 1, :] = g * (4 * Gzl - Gpl - Gml) * c * s
        A[0, 2, :] = -2 * s
        A[1, 0, :] = g * (4 * Gzl - Gpl - Gml) * c * s
        A[1, 1, :] = -g * (4 * Gzl * c**2 + (Gpl + Gml) * (1 + s**2))
        A[1, 2, :] = 2 * c
        A[2, 0, :] = 2 * s
        A[2, 1, :] = -2 * c
        A[2, 2, :] = -g * (4 * Gzl + Gpl + Gml)

        b = np.vstack(
            [
                g * (2 * Gzl * s**2 + 0.5 * (Gpl * (1 - c) ** 2 + Gml * (1 + c) ** 2)),
                -g * (2 * Gzl * c * s - 0.5 * (Gpl * s * (c - 2) + Gml * s * (c + 2))),
                -s,
            ]
        )
        A = np.where(np.isnan(A), 0.0, A)
        b = np.where(np.isnan(b), 0.0, b)
        return A, b
