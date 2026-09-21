# -*- coding: utf-8 -*-
"""validate_suite.py —— 梯度 OCP 工具箱的自检。

对应 MATLAB 的 ``tests/validate_suite.m``。

运行：
    python tests/validate_suite.py

检查四项：
    [1] 解析梯度是下降方向（量子比特问题 + 标量例子）
    [2] 求解器确实让成本下降（两个问题）
    [3] 历史累积方向（Adam）也能降低成本
    [4] 同伦延拓外壳可以跑通

MATLAB 版在本机的实测输出（README 第 8 节）：
    [1] 梯度是下降方向      : qubit=PASS example=PASS
    [2] 求解器成本下降       : qubit 0.316→0.175(60步)  example 4.554→4.494
    [3] Adam 方向优化        : qubit 0.316→0.053
    [4] 同伦延拓运行         : kappa 4 步 J=0.052
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np    # noqa: E402

from sigmax_ocp import continuation, ocp, solver    # noqa: E402


def yesno(v: bool) -> str:
    return "PASS" if v else "FAIL"


def descent_check(m, tg, U) -> bool:
    """检查解析梯度是不是下降方向（走一小步，成本应当下降）。"""
    J0, _, _, G = m.cost_gradient(tg, U)
    U2 = m.project(U - 1e-3 * G)
    J2, _, _, _ = m.cost_gradient(tg, U2)
    return bool(np.isfinite(J2) and J2 < J0)


def solver_decreases(m, iters: int, tag: str, direction: str = "gradient") -> bool:
    """检查求解器是否让成本下降。"""
    s = solver.GradientSolver(m)
    s.direction = direction
    s.max_iter = iters
    s.verbosity = 0
    s.line_search = "armijo"
    res = s.solve()
    if len(res.hist.J) < 2:
        return False
    J0, Jf = res.hist.J[0], res.hist.J[-1]
    print(
        f"   [{tag}] {direction}: J0={J0:.4e} -> Jf={Jf:.4e} (iter {res.iterations})"
    )
    return bool(np.isfinite(Jf) and Jf < J0)


def homotopy_runs() -> bool:
    """检查同伦延拓能否跑通。"""
    m = ocp.QubitReset(tau=0.15, N=40, homotopy_name="kappa")
    path = np.linspace(0.2, 1.0, 4)
    c = continuation.HomotopyContinuation(m, path)
    c.solver.max_iter = 25
    c.solver.verbosity = 0
    c.verbose = False
    r = c.run()
    print(f"   [homotopy] final J={r.J:.4e} over {path.size} steps")
    return bool(np.isfinite(r.J))


def main() -> int:
    print("=== gradient OCP self check (Python) ===")

    # ---- 1. 解析梯度是下降方向 ----
    m = ocp.QubitReset(tau=0.15, N=60, gamma=10)
    tg = m.time_grid(m.N)
    # MATLAB: U = repmat(0.6*m.lambdaMax, 1, m.N);  严格正的控制
    U = np.tile((0.6 * m.lambda_max)[:, None], (1, m.N))
    ok1 = descent_check(m, tg, U)

    m2 = ocp.ControlExample(N=60)
    tg2 = m2.time_grid(m2.N)
    ok1b = descent_check(m2, tg2, 0.3 * np.ones((1, m2.N)))
    print(f"[1] gradient is descent dir    : qubit={yesno(ok1)} example={yesno(ok1b)}")

    # ---- 2. 求解器让成本下降 ----
    ok2a = solver_decreases(ocp.QubitReset(tau=0.15, N=50), 60, "qubit")
    ok2b = solver_decreases(ocp.ControlExample(N=50), 60, "example")
    print(f"[2] solver decreases cost      : qubit={yesno(ok2a)} example={yesno(ok2b)}")

    # ---- 3. Adam 方向也降低成本 ----
    ok3 = solver_decreases(ocp.QubitReset(tau=0.15, N=50), 60, "qubit", "adam")
    print(f"[3] adam direction reduces cost: {yesno(ok3)}")

    # ---- 4. 同伦延拓 ----
    ok4 = homotopy_runs()
    print(f"[4] homotopy continuation runs : {yesno(ok4)}")

    allok = all([ok1, ok1b, ok2a, ok2b, ok3, ok4])
    print(f"=== done {'(all PASS)' if allok else '(there are FAILures)'} ===")
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
