# -*- coding: utf-8 -*-
"""demo_gradient.py —— 用梯度工具箱求解时间最优量子比特重置问题，并绘图。

对应 MATLAB 的 ``demo/demo_gradient.m``。

运行：
    python demo/demo_gradient.py

图片输出到 ``demo/output/``。
"""

from __future__ import annotations

import pathlib
import sys
import time

# 把项目根目录加进 sys.path
# （对应 MATLAB 的 addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'src'))）
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")            # 无界面后端：只产出图片文件，不弹窗
import matplotlib.pyplot as plt    # noqa: E402

from sigmax_ocp import ocp, solver    # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)


def main() -> None:
    # ------------------------------------------------------------------
    # 1. 建模型（要解决的那个 OCP）
    # ------------------------------------------------------------------
    model = ocp.QubitReset(tau=0.15, N=100, gamma=10, omega_c=1, lambda_max=[3, 2])

    # ------------------------------------------------------------------
    # 2. 配置求解器
    #    direction : 'gradient' | 'adagrad' | 'rmsprop' | 'momentum'
    #                | 'nestorov' | 'adam'
    #    line_search: 'simple'  | 'armijo'  | 'wolfe'
    # ------------------------------------------------------------------
    s = solver.GradientSolver(model)
    s.direction = "adam"
    s.line_search = "armijo"
    s.max_iter = 2000
    s.tol_grad = 1e-5
    s.verbosity = 1

    # ------------------------------------------------------------------
    # 3. 求解
    # ------------------------------------------------------------------
    print(f"\n===== solving {model.name} with {s.direction} =====")
    tg = model.time_grid(model.N)
    t0 = time.time()
    res = s.solve(tg=tg)      # 显式传 tg，确保求解用的格点与画图用的完全一致
    elapsed = time.time() - t0
    print(
        f"elapsed {elapsed:.2f} s  |  converged={res.converged} ({res.reason})  "
        f"|  iter={res.iterations}"
    )

    # ------------------------------------------------------------------
    # 4. 画状态 / 控制 / 成本历史
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.2), facecolor="white")

    # (a) 状态：三个布居
    ax = axes[0]
    ax.plot(tg, res.X[0], "r-", linewidth=1.5, label="p_e")
    ax.plot(tg, res.X[1], "b-", linewidth=1.5, label="p_r")
    ax.plot(tg, res.X[2], "m-", linewidth=1.5, label="p_i")
    ax.set_xlabel("time")
    ax.set_ylabel("population")
    ax.set_title("state")
    ax.grid(True)
    ax.legend(loc="best")

    # (b) 控制：lambda_x / lambda_z
    ax = axes[1]
    ax.plot(tg, res.U[0], "r-", linewidth=1.5, label=r"$\lambda_x$")
    ax.plot(tg, res.U[1], "b-", linewidth=1.5, label=r"$\lambda_z$")
    ax.set_xlabel("time")
    ax.set_ylabel("control")
    ax.set_title(f"control (direction: {s.direction})")
    ax.grid(True)
    ax.legend(loc="best")

    # (c) 成本历史（对数纵轴）
    ax = axes[2]
    ax.semilogy(range(1, len(res.hist.J) + 1), res.hist.J, "k-", linewidth=1.5)
    ax.set_xlabel("iteration")
    ax.set_ylabel("cost")
    ax.set_title("cost history")
    ax.grid(True)

    fig.tight_layout()
    outfile = OUT / "demo_gradient.png"
    fig.savefig(outfile, dpi=120)
    print(f"\n图片已保存: {outfile}")
    print(f"\ninitial cost = {res.hist.J[0]:.6e}, final cost = {res.J:.6e}")


if __name__ == "__main__":
    main()
