# -*- coding: utf-8 -*-
"""solver 子包 —— 求解算法（对应 MATLAB 的 ``+solver`` 包）。

    GradientSolver   直接法梯度求解器
    LineSearch       非精确步长搜索
"""

from .gradient_solver import GradientSolver, History, SolveResult
from .line_search import LineSearch, LineSearchOptions

__all__ = [
    "GradientSolver",
    "History",
    "SolveResult",
    "LineSearch",
    "LineSearchOptions",
]
