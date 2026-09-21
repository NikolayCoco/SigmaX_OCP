# -*- coding: utf-8 -*-
"""HomotopyContinuation —— 抽象的延拓（prolongation）外壳。

MATLAB 版 ``src/+continuation/HomotopyContinuation.m`` 的 Python 重写。

延拓 / 同伦连续法求解困难最优控制问题的思路：把问题嵌入一族单参数问题，
让参数从"容易的问题"走到"我们真正要求解的问题"，每一步用上一步的解做
热启动（warm start）：

    p_0（容易） -> p_1 -> ... -> p_end = 原问题

通用的延拓参数就是 ``model.homotopy_value``。对量子比特重置问题，支持两个
物理上有意义的延拓族：

    * 沿时间长度 tau 延拓（homotopy_name = 'tau'）
      —— 把时间窗口缩短，使终端成本在起点处的梯度不再消失
         （即"初始位置梯度消失"问题）。
    * 沿终端成本权重 kappa 延拓（0 -> 1，同时加入正则化运行项）
      （homotopy_name = 'kappa'）—— 见原理文档的"正则化 / 延拓"一节。

这个外壳**完全与具体问题无关**：它只调用 set_homotopy 和底层的
GradientSolver，所以任何支持延拓的 AbstractOCP 都能复用它。

例：
    c = HomotopyContinuation(model, np.linspace(0.09, 1.0, 12))
    r = c.run()
    r.U, r.J                     # 最终解
    [step.J for step in r.results]   # 各步成本
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..ocp.abstract_ocp import AbstractOCP
from ..solver.gradient_solver import GradientSolver


@dataclass
class ContinuationStep:
    """延拓路径上的一步。对应 MATLAB 里 results(i) 那个 struct。"""

    U: np.ndarray          # 该步求得的控制
    J: float               # 该步的成本
    params: float          # 该步的延拓参数值（MATLAB 里叫 params）
    converged: bool        # 该步是否收敛


@dataclass
class ContinuationResult:
    """延拓运行结果。对应 MATLAB 的 ``res`` struct。"""

    results: list[ContinuationStep]
    path: np.ndarray
    tgrids: list[np.ndarray]
    U: np.ndarray
    J: float

    def __repr__(self) -> str:
        return f"<ContinuationResult steps={len(self.results)} final J={self.J:.6e}>"


class HomotopyContinuation:
    """沿一条参数路径延拓求解的最外层驱动。"""

    def __init__(
        self,
        model: AbstractOCP,
        path,
        solver: Optional[GradientSolver] = None,
        **solver_kwargs,
    ) -> None:
        """构造。

        参数
        ----
        model : AbstractOCP
        path  : 延拓参数序列（建议升序）
        solver: 可选的现成 GradientSolver；不给就新建一个
        其余关键字参数都会转交给新建的 GradientSolver（对应 MATLAB 的
                ``solver.GradientSolver(model, varargin{:})``）

        对应 MATLAB 里那段 ``varargin`` 分派：
            if isempty(varargin)              -> 用模型自带 N 和默认求解器
            elseif isa(varargin{1},'GradientSolver') -> 直接用传入的求解器
            else                              -> 用参数新建求解器
        """
        self.model = model
        self.path = np.asarray(path, dtype=float)
        self.verbose = True

        if solver is not None:
            self.solver = solver
        else:
            self.solver = GradientSolver(model, **solver_kwargs)

        # 模型默认 N：每一步都用它重建时间格点
        # MATLAB: if isprop(model, 'N'), obj.N = model.N; else, obj.N = 100; end
        self.N: int = int(getattr(model, "N", 100))

        # MATLAB: if obj.model.homotopyValue ~= path(1), obj.model.setHomotopy(path(1)); end
        if self.path.size and self.model.homotopy_value != self.path[0]:
            self.model.set_homotopy(float(self.path[0]))

    def run(self, U0: Optional[np.ndarray] = None) -> ContinuationResult:
        """沿参数路径行进，并在每一步求解。

        热启动逻辑：每一步把上一步的解 ``sol.U`` 作为下一步的初值。
        """
        path = self.path
        nP = path.size
        tg_cache: list[np.ndarray] = []
        results: list[ContinuationStep] = []
        U = U0

        for i in range(nP):
            # 1) 先设延拓参数（对 'tau' 族，这一步会改变 model.tau）
            self.model.set_homotopy(float(path[i]))
            # 2) 再重建时间格点，从而"接住"被改掉的 horizon
            tg = self.model.time_grid(self.N)
            tg_cache.append(tg)

            # 3) 第一步（或外部未给初值时）用模型自带的初始猜测
            if U is None:
                U = self.model.initial_control(tg)

            if self.verbose:
                print(
                    f"== prolong step {i + 1}/{nP} : "
                    f"{self.model.homotopy_name} = {path[i]:.4g} =="
                )

            # 4) 求解（预热启动）
            sol = self.solver.solve(U, tg)
            results.append(
                ContinuationStep(
                    U=sol.U, J=sol.J, params=float(path[i]), converged=sol.converged
                )
            )
            U = sol.U      # 作为下一步的热启动

        return ContinuationResult(
            results=results,
            path=path,
            tgrids=tg_cache,
            U=results[-1].U,
            J=results[-1].J,
        )
