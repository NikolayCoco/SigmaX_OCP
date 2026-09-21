# -*- coding: utf-8 -*-
"""SigmaX OCP —— 梯度法最优控制工具箱（Python 版）。

这是 MATLAB 工程 ``SigmaX_OCP`` 的 Python 版本，结构与 MATLAB 的包一一对应：

    MATLAB                          Python
    ----------------------------    --------------------------------
    src/+ocp/AbstractOCP.m          sigmax_ocp/ocp/abstract_ocp.py
    src/+ocp/QubitReset.m           sigmax_ocp/ocp/qubit_reset.py
    src/+ocp/ControlExample.m       sigmax_ocp/ocp/control_example.py
    src/+solver/GradientSolver.m    sigmax_ocp/solver/gradient_solver.py
    src/+solver/LineSearch.m        sigmax_ocp/solver/line_search.py
    src/+vis/Visualizer.m           sigmax_ocp/vis/visualizer.py
    src/+continuation/Homotopy...   sigmax_ocp/continuation/homotopy_continuation.py

快速开始：
    from sigmax_ocp import ocp, solver

    model = ocp.QubitReset(tau=0.15, N=100, gamma=10, lambda_max=[3, 2])
    s = solver.GradientSolver(model)
    s.direction = "gradient"
    s.line_search = "armijo"
    s.max_iter = 2000

    res = s.solve()
    print(res.J, res.converged, res.reason)
"""

from . import continuation, ocp, solver, vis
from .continuation import ContinuationResult, ContinuationStep, HomotopyContinuation
from .ocp import AbstractOCP, ControlExample, QubitReset
from .solver import GradientSolver, History, LineSearch, LineSearchOptions, SolveResult
from .vis import Visualizer

__version__ = "1.0.0"

__all__ = [
    # 子包
    "ocp",
    "solver",
    "vis",
    "continuation",
    # 问题
    "AbstractOCP",
    "QubitReset",
    "ControlExample",
    # 求解器
    "GradientSolver",
    "LineSearch",
    "LineSearchOptions",
    "SolveResult",
    "History",
    # 可视化与延拓
    "Visualizer",
    "HomotopyContinuation",
    "ContinuationResult",
    "ContinuationStep",
]
