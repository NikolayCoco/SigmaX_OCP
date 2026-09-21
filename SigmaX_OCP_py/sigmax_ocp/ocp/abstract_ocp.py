# -*- coding: utf-8 -*-
"""AbstractOCP —— 离散化最优控制问题的抽象契约。

这是 MATLAB 版 ``src/+ocp/AbstractOCP.m`` 的 Python 重写。

求解器（梯度工具箱）关于一个具体问题**所需要知道的全部信息**，都由这个接口
表达出来：

    n_state / n_control   状态、控制的维数
    tau                   时间区间长度（horizon time）
    time_grid(N)          时间格点            -> shape (N,)
    initial_state()       初态                -> shape (n_state,)
    initial_control(tg)   初始控制猜测        -> shape (n_control, N)
    cost_gradient(tg, U)  成本、状态、协态、成本对控制的梯度
    project(U)            把控制投影回可行域
    control_bounds()      控制上下界          -> (Ulo, Uhi)，各 shape (n_control,)

某个具体问题的物理全部藏在这些方法背后。想求解**另一个**最优控制问题，只需
继承本类并实现这些方法，求解器一行都不用改——这就是"问题可替换"的含义。

同伦 / 延拓（homotopy / continuation）
--------------------------------------
标量延拓参数通过 ``homotopy_value`` 暴露（默认 1，表示"原问题"）。想让自己的
问题支持沿一条参数路径延拓（例如时间长度 tau、或终端成本权重 kappa）的子类，
应当在 ``cost_gradient`` / ``time_grid`` 里读取 ``self.homotopy_value``，这样
通用的 HomotopyContinuation 外壳就能驱动它。``set_homotopy`` 只负责存值。

MATLAB 与 Python 的对照
-----------------------
    MATLAB                                Python
    ----------------------------------    ------------------------------------
    classdef (Abstract) X < handle        class X(AbstractOCP)  （abc.ABC）
    properties (Abstract)                 @property + @abstractmethod
    obj.method(arg)                       self.method(arg)   （self 显式传递）
    1-based 下标 a(1), a(k+1)             0-based 下标 a[0], a[k+1]
    nState x N 矩阵                       np.ndarray，shape = (n_state, N)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class AbstractOCP(ABC):
    """所有（离散化）最优控制问题的抽象基类。"""

    # ------------------------------------------------------------------
    # 子类必须给出的基本信息。
    # MATLAB 里这些写作 properties (Abstract)，Python 里用类属性 + 抽象 property
    # 来达到同样的"不实现就实例化不了"的效果。
    # ------------------------------------------------------------------
    n_state: int = 0      # 状态维数
    n_control: int = 0    # 控制维数
    name: str = "unnamed OCP"

    def __init__(self) -> None:
        # MATLAB: properties 里带默认值的普通属性
        self.tau: float = 1.0            # 时间区间长度（可被延拓修改）
        self.homotopy_value: float = 1.0  # 延拓参数（默认 1 = 原问题）
        self.homotopy_name: str = ""      # 延拓参数的说明，例如 'tau' / 'kappa'

    # ==================================================================
    # 抽象方法：子类必须实现，否则实例化时直接报错。
    # MATLAB 用 methods (Abstract) 达成同样目的。
    # ==================================================================

    @abstractmethod
    def time_grid(self, N: int | None = None) -> np.ndarray:
        """返回 [0, tau] 上的时间格点，shape (N,)。"""

    @abstractmethod
    def initial_state(self) -> np.ndarray:
        """返回初态，shape (n_state,)。"""

    @abstractmethod
    def initial_control(self, tg: np.ndarray) -> np.ndarray:
        """返回初始控制的猜测，shape (n_control, N)。"""

    @abstractmethod
    def cost_gradient(
        self, tg: np.ndarray, U: np.ndarray
    ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """返回 (J, X, Mu, G)。

        J    标量成本
        X    状态轨迹，shape (n_state, N)
        Mu   协态轨迹，shape (n_state, N)
        G    成本对控制样本的梯度，shape (n_control, N)
        """

    @abstractmethod
    def project(self, U: np.ndarray) -> np.ndarray:
        """把控制投影到可行盒约束上。"""

    @abstractmethod
    def control_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """返回控制的下界与上界 (Ulo, Uhi)，各 shape (n_control,)。"""

    # ==================================================================
    # 具体方法：所有子类直接继承，不必重写。
    # ==================================================================

    def set_homotopy(self, v: float) -> None:
        """存下当前的延拓参数值。

        对应 MATLAB 的 ``setHomotopy``。子类可以重写它，以便在参数变化时
        顺带改别的东西（例如 QubitReset 在 'tau' 延拓时同步改 tau）。
        """
        self.homotopy_value = v

    def print_info(self) -> None:
        """打印问题的简要说明（对应 MATLAB 的 ``printInfo``）。"""
        print(f"OCP  : {self.name}")
        print(f"state dim   = {self.n_state}")
        print(f"control dim = {self.n_control}")
        print(f"horizon tau = {self.tau:.4g}")
        if self.homotopy_name:
            print(f"homotopy    = {self.homotopy_name} (value {self.homotopy_value:.4g})")

    # ------------------------------------------------------------------
    # 小工具：MATLAB 的 trapz 在 numpy 2.x 里改名了。
    #   numpy 1.x : np.trapz
    #   numpy 2.x : np.trapezoid   （np.trapz 已被移除）
    # 这里统一成一个方法，全项目都用它，避免版本差异带来的坑。
    # ------------------------------------------------------------------
    @staticmethod
    def trapezoid(tg: np.ndarray, y: np.ndarray) -> float:
        """梯形法积分 ∫ y dt，等价于 MATLAB 的 ``trapz(tg, y)``。

        ``y`` 允许是 (n, N)；此时对最后一维（时间轴）积分，返回标量。
        """
        return float(np.trapezoid(y, tg, axis=-1))

    def __repr__(self) -> str:
        return f"<{type(self).__name__} '{self.name}' n_state={self.n_state} n_control={self.n_control} tau={self.tau:g}>"
