# -*- coding: utf-8 -*-
"""demo_example1.py —— 用**同一个**梯度求解器求解一个**完全不同**的最优控制问题，
证明"问题是可以替换的"。

对应 MATLAB 的 ``demo/demo_example1.m``。

问题（descriptor 例 1）：

    min  J = ∫_0^1 0.5 ( x^2 + u^2 ) dt
    s.t. x(0) = 10,  x' = -x^2 + u,  终端状态自由

模型是 ``ocp.ControlExample``。**求解器一行代码都没改。**

运行：
    python demo/demo_example1.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt    # noqa: E402

from sigmax_ocp import ocp, solver    # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)


def main() -> None:
    model = ocp.ControlExample(N=200)
    tg = model.time_grid(model.N)

    # 与量子比特问题用的是同一个求解器类
    s = solver.GradientSolver(model)
    s.direction = "gradient"
    s.line_search = "armijo"
    s.max_iter = 500
    s.tol_grad = 1e-6

    print(f"===== solving {model.name} =====")
    # ⚠ 必须显式传 tg！
    # MATLAB 版这里写的是 s.solve()，而 solve 内部用的是 **solver 自己的 N**
    # （默认 100），不是 model.N（这里是 200）。于是 res.U / res.X 只有 100 个点，
    # 而外层的 tg 有 200 个点 —— MATLAB 版 demo_example1.m 会在下面 plot 的地方直接报
    # "Vectors must be the same length"。Python 版在这里把这个坑修掉了。
    res = s.solve(tg=tg)
    print(
        f"iter={res.iterations}  J={res.J:.6e}  "
        f"converged={res.converged} ({res.reason})"
    )

    # 解析参考：协态 lambda(1)=0，最优控制 u = lambda，且 x' = -x^2 + u
    # （descriptor 里推出第一次迭代 u^(1) = 0.5*((10t+1)/11)^2 - 1）
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), facecolor="white")

    axes[0].plot(tg, res.U[0], "b-", linewidth=1.5)
    axes[0].set_xlabel("time")
    axes[0].set_ylabel("control u")
    axes[0].set_title("control")
    axes[0].grid(True)

    axes[1].plot(tg, res.X[0], "r-", linewidth=1.5)
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("state x")
    axes[1].set_title("state")
    axes[1].grid(True)

    fig.tight_layout()
    outfile = OUT / "demo_example1.png"
    fig.savefig(outfile, dpi=120)
    print(f"图片已保存: {outfile}")


if __name__ == "__main__":
    main()
