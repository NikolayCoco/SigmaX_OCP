# -*- coding: utf-8 -*-
"""demo_animation.py —— 一边求解量子比特重置问题，一边把迭代过程录成动画。

对应 MATLAB 的 ``demo/demo_animation.m``。

    animate    -> True   把每一次迭代录到 'iteration.gif'（或 .mp4）
    show_live  -> False  不在屏幕上弹窗（无界面环境也能跑）
    format     -> 'gif'  （通用可看）或 'mp4'（需要 ffmpeg）

为让录制足够快，这里用的迭代上限比 MATLAB 版（1e5）小得多；两者都靠
``tol_grad`` 收敛自动停下来。

运行：
    python demo/demo_animation.py
输出：
    demo/output/iteration.gif
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from sigmax_ocp import ocp, solver, vis    # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)


def main() -> None:
    # ---- 模型与求解器 ----
    model = ocp.QubitReset(tau=1.5, N=60, gamma=10)
    s = solver.GradientSolver(model)
    s.direction = "gradient"
    s.line_search = "armijo"
    s.max_iter = 150        # MATLAB 版是 1e5，靠 tol_grad 收敛提前停止
    s.tol_grad = 1e-4
    s.verbosity = 0

    # ---- 可视化 / 动画器 ----
    v = vis.Visualizer(
        animate=True,
        format="gif",
        video_file=str(OUT / "iteration.gif"),
        fps=12,
        show_live=False,
    )
    tg = model.time_grid(model.N)
    v.init(model, tg)
    # 求解器每轮迭代回调一次；这里把 observer 接到可视化器上
    s.observer = lambda it, J, X, Mu, U, tgrid: v.snapshot(it, J, X, Mu, U, tgrid)

    # ---- 求解（observer 会录制每一轮） ----
    print("===== solving while animating =====")
    # 显式传 tg：否则 solve 内部会用自己的 N（默认 100），与这里的 60 点不一致
    res = s.solve(tg=tg)
    v.finish()

    print(f"converged={res.converged} ({res.reason})  iter={res.iterations}  J={res.J:.6e}")
    print(f"animation written to: {v.video_file}")


if __name__ == "__main__":
    main()
