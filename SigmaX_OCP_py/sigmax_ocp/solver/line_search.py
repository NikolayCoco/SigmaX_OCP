# -*- coding: utf-8 -*-
"""LineSearch —— 适用于 AbstractOCP 的非精确（inexact）步长准则。

MATLAB 版 ``src/+solver/LineSearch.m`` 的 Python 重写。

给定一个下降方向 d（即 phi'(0) = <grad J, d> < 0），线搜索返回一个步长 alpha，
使成本"充分下降"。支持的准则：

    'simple'   纯回溯（只看充分下降条件）
    'armijo'   Armijo-Goldstein：充分下降 + 曲率下界
    'wolfe'    Wolfe-Powell：充分下降 + dphi >= sigma*dphi0

步长被拒绝时的折半方式：'bisection'（把步长减半）。

线搜索只需要 AbstractOCP 接口（cost_gradient），所以**一个 LineSearch 对象可以
服务任何问题**，这正是"求解器与问题解耦"的体现。

例：
    ls = LineSearch()
    alpha = ls.search('armijo', 'bisection', model, tg, U, d, G, opts)

MATLAB 对照
-----------
    obj.ls.search(...)        这里把 opts 由 struct 换成 dataclass
    phi = @(a) ...            这里用嵌套函数 / 局部方法
    strcmpi(rule,'wolfe')     这里用 rule.lower() == 'wolfe'
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..ocp.abstract_ocp import AbstractOCP


@dataclass
class LineSearchOptions:
    """线搜索参数。对应 MATLAB 里那个临时 struct ``s``。"""

    rho: float = 0.1          # 充分下降参数 c1
    sigma: float = 0.4        # 曲率参数 c2（Wolfe 用）
    alpha0: float = 0.5       # 初始试探步长
    alpha_min: float = 1e-8   # 最小步长
    alpha_max: float = 1.0    # 最大步长


class LineSearch:
    """无状态的线搜索器（因此可以被任意多个求解器共享）。"""

    #: 回溯的最大次数，对应 MATLAB 里的 ``maxIter = 100``
    MAX_ITER = 100

    def search(
        self,
        rule: str,
        interp: str,
        model: AbstractOCP,
        tg: np.ndarray,
        U: np.ndarray,
        d: np.ndarray,
        G0: np.ndarray,
        opts: LineSearchOptions,
    ) -> float:
        """为方向 d 返回一个步长 alpha。

        参数
        ----
        rule    : 'simple' | 'armijo' | 'wolfe'
        interp  : 'bisection'（折半）
        model   : AbstractOCP 实例
        tg      : 时间格点，shape (N,)
        U       : 当前控制，shape (n_control, N)
        d       : 下降方向，shape (n_control, N)
        G0      : U 处的梯度 grad J（本实现只在 dphi0 里用 d，G0 保留以对齐
                  MATLAB 的函数签名，便于逐行对照）
        opts    : LineSearchOptions

        注意：MATLAB 版里在这里声明 ``phi``/``dph`` 两个匿名函数，Python 用
        更轻量的 ``functools.partial`` 风格闭包（下面 _phi / _dphi）。
        """
        c1 = opts.rho      # 充分下降参数
        c2 = opts.sigma    # 曲率参数（Wolfe 用）

        # MATLAB: phi0 = phi(0); dphi0 = dph(0);
        phi0 = self._phi(model, tg, U, d, 0.0)
        dphi0 = self._dphi(model, tg, U, d, 0.0)

        # 若方向不是下降方向（dphi0 >= 0），直接返回 0 步长。
        if dphi0 >= 0:
            return 0.0

        alpha = opts.alpha0
        it = 0

        # ---- 阶段 1：充分下降条件（Armijo 上界） ----
        # MATLAB: while phi(alpha) > phi0 + c1*alpha*dphi0
        while self._phi(model, tg, U, d, alpha) > phi0 + c1 * alpha * dphi0:
            alpha = self._reduce(interp, alpha, opts.alpha_min)
            if alpha <= opts.alpha_min or it > self.MAX_ITER:
                alpha = opts.alpha_min
                break
            it += 1

        # ---- 阶段 2（仅 Wolfe）：步长不要太小（曲率条件） ----
        if rule.lower() == "wolfe" and self._dphi(model, tg, U, d, alpha) < c2 * dphi0:
            it = 0
            while self._dphi(model, tg, U, d, alpha) < c2 * dphi0:
                alpha = min(2.0 * alpha, opts.alpha_max)
                if alpha >= opts.alpha_max or it > self.MAX_ITER:
                    break
                it += 1

        return alpha

    # ==================================================================
    # 内部工具
    # ==================================================================

    @staticmethod
    def _reduce(interp: str, alpha: float, alpha_min: float) -> float:
        """把被拒绝的步长往 0 收缩。

        对应 MATLAB 的 ``reduce``。MATLAB 版里 switch 的两个分支取值相同（都是
        0.5*alpha），这里保留同样的结构，方便以后换插值方式（例如二次插值）。
        """
        key = (interp or "").lower()
        if key in ("bi", "bisection", "quad", "quadratic"):
            a = 0.5 * alpha      # 折半 / 简单的二次收缩
        else:
            a = 0.5 * alpha
        return max(a, alpha_min)

    @staticmethod
    def _phi(
        model: AbstractOCP, tg: np.ndarray, U: np.ndarray, d: np.ndarray, alpha: float
    ) -> float:
        """沿方向 d 走 alpha 后的成本 phi(alpha)。"""
        J, _, _, _ = model.cost_gradient(tg, U + alpha * d)
        return float(J)

    @staticmethod
    def _dphi(
        model: AbstractOCP, tg: np.ndarray, U: np.ndarray, d: np.ndarray, alpha: float
    ) -> float:
        """phi'(alpha) = <grad J(U + alpha*d), d>。

        MATLAB: val = trapz(tg, sum(G .* d, 1));
        注意 MATLAB 的 trapz(X, Y) 是 (坐标, 函数值)，numpy 的
        np.trapezoid(y, x) 顺序相反 —— 统一走 AbstractOCP.trapezoid 封装。
        """
        _, _, _, G = model.cost_gradient(tg, U + alpha * d)
        return AbstractOCP.trapezoid(tg, np.sum(G * d, axis=0))
