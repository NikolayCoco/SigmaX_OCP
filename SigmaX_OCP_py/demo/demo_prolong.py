# -*- coding: utf-8 -*-
"""demo_prolong.py —— 量子比特重置问题上的同伦延拓（prolongation）。

对应 MATLAB 的 ``demo/demo_prolong.m``。

终端成本 |p(tau)| 在时间窗口较长时会让"对初始控制的梯度"消失。
从**容易的子问题**一路走到**真正关心的问题**（每一步都热启动），正好治这个病。
模型暴露两个延拓族：

    homotopy_name = 'tau'    把时间长度从小长到大（本 demo 用的）
    homotopy_name = 'kappa'  把终端成本权重从 0 长到 1

见原理文档（descriptor/main.tex）里 "regularization & continuation" 一节。

规模说明
--------
MATLAB 版用的是 ``N=500, steps=14, max_iter=1000``；Python 的逐点循环比
MATLAB 慢，照搬会跑十几分钟。这里默认取适中规模以便快速看到效果，
需要复现 MATLAB 版规模时用环境变量覆盖：

    set SIGMAX_PROLONG_N=500
    set SIGMAX_PROLONG_STEPS=14
    set SIGMAX_PROLONG_MAXIT=1000

运行：
    python demo/demo_prolong.py
"""

from __future__ import annotations

import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib    # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt    # noqa: E402
import numpy as np    # noqa: E402

from sigmax_ocp import continuation, ocp    # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)

N = int(os.environ.get("SIGMAX_PROLONG_N", "100"))
STEPS = int(os.environ.get("SIGMAX_PROLONG_STEPS", "8"))
MAXIT = int(os.environ.get("SIGMAX_PROLONG_MAXIT", "250"))


def main() -> None:
    # ---- 1. 模型与延拓路径 ----
    model = ocp.QubitReset(tau=1.5, N=N, gamma=10, lambda_max=[2, 1.5])
    model.homotopy_name = "tau"          # 沿时间长度延拓

    # 从很短的时间窗开始，一路长到目标 tau（这就是延拓路径）
    tau_start, tau_end = 0.1, model.tau
    path = np.linspace(tau_start, tau_end, STEPS)

    # ---- 2. 包上模型，沿路径求解 ----
    c = continuation.HomotopyContinuation(model, path)
    c.solver.direction = "gradient"
    c.solver.line_search = "armijo"
    c.solver.max_iter = MAXIT
    c.solver.tol_grad = 1e-5
    c.solver.verbosity = 0

    print(f"\n===== homotopy on {model.homotopy_name}, {STEPS} steps "
          f"(N={N}, max_iter={MAXIT}) =====")
    t0 = time.time()
    res = c.run()
    print(f"elapsed {time.time() - t0:.1f} s")

    # ---- 3. 画成本随延拓参数的变化 + 最终控制 ----
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), facecolor="white")

    ax = axes[0]
    ax.plot(path, [st.J for st in res.results], "o-", linewidth=1.5, markersize=6)
    ax.set_xlabel(f"{model.homotopy_name} (continuation parameter)")
    ax.set_ylabel("cost")
    ax.set_title("cost along the continuation path")
    ax.grid(True)

    ax = axes[1]
    tg = model.time_grid(model.N)
    ax.plot(tg, res.U[0], "r-", linewidth=1.5, label=r"$\lambda_x$")
    ax.plot(tg, res.U[1], "b-", linewidth=1.5, label=r"$\lambda_z$")
    ax.set_xlabel("time")
    ax.set_ylabel("control")
    ax.set_title("final control")
    ax.grid(True)
    ax.legend(loc="best")

    fig.tight_layout()
    outfile = OUT / "demo_prolong.png"
    fig.savefig(outfile, dpi=120)
    print(f"图片已保存: {outfile}")

    # ---- 4. 逐步打印 ----
    for i, st in enumerate(res.results, 1):
        print(f"  step {i:2d}/{STEPS}  tau={st.params:.4g}  J={st.J:.6e}  "
              f"converged={st.converged}")
    print(f"\nfinal cost = {res.J:.6e}  (from {res.results[0].J:.6e})")


if __name__ == "__main__":
    main()
