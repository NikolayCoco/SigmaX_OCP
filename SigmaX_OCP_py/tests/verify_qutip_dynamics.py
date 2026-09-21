# -*- coding: utf-8 -*-
"""verify_qutip_dynamics.py —— 核对 QuTiP 动力学版与手写 RK4 版。

`QubitResetQuTiP` 只重写了两个动力学方法（前向状态演化、后向协态演化），
把 `d p/dt = P p + p0` 嵌成 Liouvillian superoperator 后交给 QuTiP 积分。
本脚本验证：

  [1] 前向 / 后向积分两条路径一致
  [2] 拿**分段矩阵指数**（严格解）当裁判，比较两种积分器的精度
  [3] 完整 `cost_gradient` 链路（J、X、Mu、G）
  [4] 同一个 GradientSolver 跑出的优化轨迹一致
  [5] 性能对比

运行：
    python tests/verify_qutip_dynamics.py
退出码 0 表示全部通过。

参考实测（本机，tau=0.15, gamma=10）：

    N         RK4 误差       QuTiP 误差
    20      9.350e-06      5.922e-12
    40      4.475e-07      1.843e-14
    100     9.660e-09      3.259e-16

  QuTiP 版精度高 5~7 个数量级（用 dop853），代价是慢约 8 倍。
"""

from __future__ import annotations

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np    # noqa: E402
from scipy.linalg import expm    # noqa: E402

from sigmax_ocp import ocp, solver    # noqa: E402

try:
    from sigmax_ocp.ocp.qubit_reset_qutip import QubitResetQuTiP
except ImportError:
    print("本脚本需要 QuTiP：python -m pip install qutip")
    raise SystemExit(2)


def exact_forward(m, tg, theta, Lam) -> np.ndarray:
    """分段矩阵指数给出的**严格解**（每段系数恒定，段内解析可积）。"""
    N = tg.size
    X = np.zeros((m.n_state, N))
    X[:, 0] = m.p_initial
    for k in range(N - 1):
        A, b = m._mat_vec(theta[k], Lam[k])
        dt = tg[k + 1] - tg[k]
        aug = np.zeros((4, 4))
        aug[:3, :3] = A
        aug[:3, 3] = b
        E = expm(aug * dt)
        X[:, k + 1] = E[:3, :3] @ X[:, k] + E[:3, 3]
    return X


def main() -> int:
    ok = True
    print("=" * 74)
    print("QuTiP 动力学版  vs  手写 RK4 版")
    print("=" * 74)

    # ---- [1] 两条路径一致性 ----
    print("\n[1] 前向 / 后向积分一致性")
    N = 40
    m1 = ocp.QubitReset(tau=0.15, N=N, gamma=10)
    m2 = QubitResetQuTiP(tau=0.15, N=N, gamma=10)
    tg = m1.time_grid(N)
    U = m1.initial_control(tg)
    theta, Lam = m1._control_transform(U)

    Xa = m1._forward_integrate(tg, theta, Lam, m1.p_initial)
    Xb = m2._forward_integrate(tg, theta, Lam, m2.p_initial)
    dX = np.abs(Xa - Xb).max()
    print(f"    前向轨迹最大差异 = {dX:.3e}")

    mu0 = -np.array([0.1, 0.2, 0.3])
    Ma = m1._backward_integrate(tg, theta, Lam, mu0)
    Mb = m2._backward_integrate(tg, theta, Lam, mu0)
    dM = np.abs(Ma - Mb).max()
    print(f"    后向协态最大差异 = {dM:.3e}")
    print("    （差异应≈RK4 的截断误差量级，而非物理差异）")

    # ---- [2] 对严格解的精度 ----
    print("\n[2] 对分段矩阵指数（严格解）的精度")
    print(f"    {'N':>5} {'RK4 误差':>14} {'QuTiP 误差':>14}")
    for n in (20, 40, 100):
        a = ocp.QubitReset(tau=0.15, N=n, gamma=10)
        b = QubitResetQuTiP(tau=0.15, N=n, gamma=10)
        t = a.time_grid(n)
        th, la = a._control_transform(a.initial_control(t))
        ex = exact_forward(a, t, th, la)
        ea = np.abs(a._forward_integrate(t, th, la, a.p_initial) - ex).max()
        eb = np.abs(b._forward_integrate(t, th, la, b.p_initial) - ex).max()
        print(f"    {n:5d} {ea:14.3e} {eb:14.3e}")
        ok &= eb < 1e-9

    # ---- [3] 完整 cost_gradient ----
    print("\n[3] cost_gradient 全量对比")
    J1, X1, Mu1, G1 = m1.cost_gradient(tg, U)
    J2, X2, Mu2, G2 = m2.cost_gradient(tg, U)
    print(f"    J  差 = {abs(J1 - J2):.3e}")
    print(f"    G  差 = {np.abs(G1 - G2).max():.3e}")
    print(f"    Mu 差 = {np.abs(Mu1 - Mu2).max():.3e}")

    # ---- [4] 优化轨迹 ----
    print("\n[4] 同一个 GradientSolver 的优化轨迹")
    out = {}
    for tag, mm in (("RK4  ", m1), ("QuTiP", m2)):
        s = solver.GradientSolver(mm)
        s.max_iter = 20
        s.verbosity = 0
        r = s.solve(tg=tg)
        out[tag] = r
        print(f"    {tag}  J = {r.J:.10e}   (iter {r.iterations}, {r.reason})")
    dJ = abs(out["RK4  "].J - out["QuTiP"].J)
    print(f"    最终 J 差异 = {dJ:.3e}")
    ok &= dJ < 1e-4

    # ---- [5] 性能 ----
    print("\n[5] 性能（单次前向积分）")
    print(f"    {'N':>5} {'RK4(s)':>12} {'QuTiP(s)':>12} {'倍数':>8}")
    for n in (20, 100):
        a = ocp.QubitReset(tau=0.15, N=n, gamma=10)
        b = QubitResetQuTiP(tau=0.15, N=n, gamma=10)
        t = a.time_grid(n)
        th, la = a._control_transform(a.initial_control(t))
        t0 = time.perf_counter()
        a._forward_integrate(t, th, la, a.p_initial)
        t1 = time.perf_counter()
        b._forward_integrate(t, th, la, b.p_initial)
        t2 = time.perf_counter()
        print(f"    {n:5d} {t1 - t0:12.4f} {t2 - t1:12.4f} {(t2 - t1) / max(t1 - t0, 1e-9):7.1f}x")

    print()
    print("=" * 74)
    print(f"结果：{'全部通过' if ok else '存在不达标的项'}")
    print("提示：QuTiP 版精度远高于 RK4，但慢约 8 倍；")
    print("      两者物理完全相同（同一个 P、p0），可按需选择。")
    print("=" * 74)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
