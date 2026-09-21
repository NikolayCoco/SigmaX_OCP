# -*- coding: utf-8 -*-
"""ControlExample —— 第二个、与量子比特完全无关的最优控制问题。

MATLAB 版 ``src/+ocp/ControlExample.m`` 的 Python 重写。

这是原理文档（descriptor）里的例 1：

    最小化  J = ∫_0^1 0.5 ( x^2 + u^2 ) dt
    约束    x(0) = 10,  x' = -x^2 + u,  终端状态自由

它刻意和量子比特重置问题不同（标量状态/控制、带运行成本，而不是 Mayer
型终端成本）。但因为实现了**同一个** AbstractOCP 契约，**同一个**
GradientSolver 不需要任何修改就能求解它 —— 这就是"求解器是抽象工具、
问题可以替换"的证明。

协态方程：lambda' = 2 x lambda + x,  lambda(1) = 0
梯度：    grad_u J = u - lambda
"""

from __future__ import annotations

import numpy as np

from .abstract_ocp import AbstractOCP


class ControlExample(AbstractOCP):
    """descriptor 例 1：min ∫ 0.5(x²+u²) dt，s.t. x' = -x² + u，x(0)=10。"""

    n_state = 1
    n_control = 1
    name = "descriptor example 1 (x'=-x^2+u)"

    def __init__(self, **kwargs) -> None:
        super().__init__()
        # MATLAB: properties 里的默认值 N = 200，tau 继承自基类（= 1）
        self.N = 200
        self.homotopy_name = ""     # 这个问题不做延拓
        self.set(**kwargs)

    # ------------------------------------------------------------------
    # 与 MATLAB 的 ``set(obj, varargin{:})`` 对应：
    #     model.set(N=200)
    # 等价于 MATLAB 的
    #     model.set('N', 200)
    # ------------------------------------------------------------------
    def set(self, **kwargs) -> "ControlExample":
        """按关键字批量设置属性；属性名不存在时立即报错（防止拼错静默失效）。"""
        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise AttributeError(
                    f"{type(self).__name__} 没有属性 '{key}'；"
                    f"可用属性: {sorted(k for k in vars(self) if not k.startswith('_'))}"
                )
            setattr(self, key, value)
        return self

    # ==================================================================
    # AbstractOCP 契约的实现
    # ==================================================================

    def time_grid(self, N: int | None = None) -> np.ndarray:
        """[0, tau] 上均匀分布 N 个点。MATLAB: linspace(0, obj.tau, N)"""
        if N is None:
            N = self.N
        return np.linspace(0.0, self.tau, int(N))

    def initial_state(self) -> np.ndarray:
        """MATLAB: X0 = 10;  这里返回 shape (1,) 以保持 (n_state,) 的约定。"""
        return np.array([10.0])

    def initial_control(self, tg: np.ndarray) -> np.ndarray:
        """MATLAB: U = zeros(1, numel(tg))"""
        return np.zeros((1, tg.size))

    def control_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """无界控制（±inf）。"""
        return np.array([-np.inf]), np.array([np.inf])

    def project(self, U: np.ndarray) -> np.ndarray:
        """无约束问题：投影是恒等映射。"""
        return U

    def cost_gradient(
        self, tg: np.ndarray, U: np.ndarray
    ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """返回 (J, X, Mu, G)。

        对应 MATLAB 的 costGradient。逐行对照见下方注释。
        """
        N = tg.size

        # MATLAB 里还定义了 fx = @(x) -2*x，但从未使用（死代码），此处略去。

        # ---- 前向积分状态（RK4，控制分段常数） ----
        # MATLAB:
        #   X = zeros(1, N); X(1) = obj.initialState();
        #   for k = 1:N-1
        #       dt = tg(k+1) - tg(k);
        #       X(k+1) = rk4(@(x) f(x, U(k)), X(k), dt);
        #   end
        # Python 下标整体 -1：k=1..N-1  ->  k=0..N-2（range(N-1)）
        X = np.zeros(N)
        X[0] = self.initial_state()[0]
        for k in range(N - 1):
            dt = tg[k + 1] - tg[k]
            X[k + 1] = self._rk4(lambda x, u=U[0, k]: self._f(x, u), X[k], dt)

        # ---- 后向积分协态（终端自然边界条件 lambda(1) = 0） ----
        # MATLAB:
        #   Mu = zeros(1,N); Mu(end) = 0;
        #   for k = N:-1:2
        #       dt = tg(k) - tg(k-1);
        #       Mu(k-1) = rk4(@(la) -(dlam(X(k-1), la)), Mu(k), dt);
        #   end
        # Python: k 从 N 递减到 2  ->  range(N-1, 0, -1)
        Mu = np.zeros(N)
        Mu[N - 1] = 0.0
        for k in range(N - 1, 0, -1):
            dt = tg[k] - tg[k - 1]
            x_prev = X[k - 1]
            Mu[k - 1] = self._rk4(lambda la, x=x_prev: -self._dlam(x, la), Mu[k], dt)

        # ---- 成本与梯度 ----
        # MATLAB: J = trapz(tg, 0.5*(X.^2 + U.^2));  G = U - Mu;
        J = AbstractOCP.trapezoid(tg, 0.5 * (X**2 + U[0, :] ** 2))
        G = U - Mu[None, :]        # (1,N) - (1,N) -> (n_control, N)

        return J, X[None, :], Mu[None, :], G

    # ==================================================================
    # 动力学与 RK4（MATLAB 的私有方法）
    # ==================================================================

    @staticmethod
    def _f(x: float, u: float) -> float:
        """x' = -x^2 + u"""
        return -x * x + u

    @staticmethod
    def _dlam(x: float, lam: float) -> float:
        """协态方程右端：lambda' = 2 x lambda + x"""
        return 2.0 * x * lam + x

    @staticmethod
    def _rk4(f, x: float, dt: float) -> float:
        """经典四阶 Runge-Kutta 单步；与 MATLAB 的 ``rk4`` 完全一致。"""
        k1 = f(x)
        k2 = f(x + 0.5 * dt * k1)
        k3 = f(x + 0.5 * dt * k2)
        k4 = f(x + dt * k3)
        return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
