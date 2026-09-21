# -*- coding: utf-8 -*-
"""gradient_check.py —— 用**数值微分**逐点核对解析偏导与解析梯度。

和 ``validate_suite.py`` 的分工
------------------------------
``validate_suite.py`` 的"下降方向检验"是**整体性**检验：只要求"沿负梯度走
一小步，成本下降"。它能抓住大的符号/维度错误，但对"某个采样点算错了"
或"某个偏导有 1% 偏差"这类局部错误不敏感。

本脚本做**逐点精确**检验，三层：

  [1] 偏导核对：中心差分核对
      ∂P/∂θ、∂p₀/∂θ、∂P/∂Λ、∂p₀/∂Λ 的解析实现。
      （这四个偏导在 descriptor/main.tex 里没有展开，是代码里手推的
        —— 正是最需要数值验证的地方。本项曾抓出一处真实错误：
        ∂p₀/∂θ 的 Γ₊ 项几何因子被写成 (s²-c²-2c)，正确应为 (c²-2c-s²)。）

  [2] 雅可比核对：中心差分核对极坐标变换
      ∂(θ,Λ)/∂(λx,λz) = [[λz/Λ², -λx/Λ²], [λx/Λ, λz/Λ]]

  [3] 梯度检验：核对解析梯度 G 与 J 对控制采样点的有限差分导数。
      理论关系  ∂J/∂U[i,k] ≈ G[i,k]·Δt_k
      二者残差按 **O(Δt) 一阶收敛** —— 这是"连续伴随 vs 离散伴随"的
      固有差异（解析式用采样点 k 的协态，离散伴随严格应当用 k+1 处的
      协态），**不是实现错误**。因此本项判据是"残差随 N 一阶下降"，
      而不是某个固定阈值。

运行：
    python tests/gradient_check.py
退出码 0 表示全部通过。
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np    # noqa: E402

from sigmax_ocp import ocp    # noqa: E402

H = 1e-6            # 中心差分步长（偏导核对用）
H_FD = 1e-7         # 有限差分步长（梯度检验用）


def _max_rel_err(num: np.ndarray, ana: np.ndarray) -> tuple[float, float]:
    """返回 (最大绝对误差, 以解析值量级归一的最大相对误差)。"""
    abs_err = float(np.abs(num - ana).max())
    scale = max(float(np.abs(ana).max()), 1e-12)
    return abs_err, abs_err / scale


# ======================================================================
# [1] 解析偏导
# ======================================================================
def check_partials(m) -> bool:
    print("[1] 解析偏导 vs 中心差分")
    rng = np.random.default_rng(12345)
    npnt = 80
    th = rng.uniform(0.15, 1.45, npnt)
    la = rng.uniform(0.05, 3.0, npnt)      # 避开 Λ=0 的奇点

    Pth, p0th = m._mat_vec_theta(th, la)
    Pla, p0la = m._mat_vec_lambda(th, la)

    e = {"P_th": 0.0, "p0_th": 0.0, "P_la": 0.0, "p0_la": 0.0}
    s = {"P_th": 0.0, "p0_th": 0.0, "P_la": 0.0, "p0_la": 0.0}

    for k in range(npnt):
        t, l = th[k], la[k]

        Pp, p0p = m._mat_vec(t + H, l)
        Pm, p0m = m._mat_vec(t - H, l)
        a, b = _max_rel_err((Pp - Pm) / (2 * H), Pth[:, :, k])
        e["P_th"], s["P_th"] = max(e["P_th"], a), max(s["P_th"], b)
        a, b = _max_rel_err((p0p - p0m) / (2 * H), p0th[:, k])
        e["p0_th"], s["p0_th"] = max(e["p0_th"], a), max(s["p0_th"], b)

        Pp, p0p = m._mat_vec(t, l + H)
        Pm, p0m = m._mat_vec(t, l - H)
        a, b = _max_rel_err((Pp - Pm) / (2 * H), Pla[:, :, k])
        e["P_la"], s["P_la"] = max(e["P_la"], a), max(s["P_la"], b)
        a, b = _max_rel_err((p0p - p0m) / (2 * H), p0la[:, k])
        e["p0_la"], s["p0_la"] = max(e["p0_la"], a), max(s["p0_la"], b)

    ok = True
    for key in e:
        good = s[key] < 1e-6        # 中心差分本身精度约 1e-9 量级
        ok &= good
        print(f"    d{key:6s}  绝对误差={e[key]:.3e}  相对误差={s[key]:.3e}  "
              f"{'OK' if good else 'MISMATCH'}")
    return ok


# ======================================================================
# [2] 极坐标雅可比
# ======================================================================
def check_jacobian(m) -> bool:
    print("[2] 极坐标雅可比 vs 中心差分")
    rng = np.random.default_rng(999)
    N = 60
    th = rng.uniform(0.15, 1.45, N)
    la = rng.uniform(0.1, 3.0, N)
    U = np.vstack([la * np.sin(th), la * np.cos(th)])   # 反解 λx, λz

    tx, tz, lx, lz = m._transform_jacobian(th, la)

    errs = {"dth/dlx": 0.0, "dth/dlz": 0.0, "dL/dlx": 0.0, "dL/dlz": 0.0}
    rels = {"dth/dlx": 0.0, "dth/dlz": 0.0, "dL/dlx": 0.0, "dL/dlz": 0.0}

    for k in range(N):
        Up, Um = U.copy(), U.copy()
        Up[0, k] += H
        Um[0, k] -= H
        tp, Lp = m._control_transform(Up)
        tm, Lm = m._control_transform(Um)
        a, b = _max_rel_err((tp[k] - tm[k]) / (2 * H), tx[k])
        errs["dth/dlx"], rels["dth/dlx"] = max(errs["dth/dlx"], a), max(rels["dth/dlx"], b)
        a, b = _max_rel_err((Lp[k] - Lm[k]) / (2 * H), lx[k])
        errs["dL/dlx"], rels["dL/dlx"] = max(errs["dL/dlx"], a), max(rels["dL/dlx"], b)

        Up, Um = U.copy(), U.copy()
        Up[1, k] += H
        Um[1, k] -= H
        tp, Lp = m._control_transform(Up)
        tm, Lm = m._control_transform(Um)
        a, b = _max_rel_err((tp[k] - tm[k]) / (2 * H), tz[k])
        errs["dth/dlz"], rels["dth/dlz"] = max(errs["dth/dlz"], a), max(rels["dth/dlz"], b)
        a, b = _max_rel_err((Lp[k] - Lm[k]) / (2 * H), lz[k])
        errs["dL/dlz"], rels["dL/dlz"] = max(errs["dL/dlz"], a), max(rels["dL/dlz"], b)

    ok = True
    for key in errs:
        good = rels[key] < 1e-6
        ok &= good
        print(f"    {key:8s}  绝对误差={errs[key]:.3e}  相对误差={rels[key]:.3e}  "
              f"{'OK' if good else 'MISMATCH'}")
    return ok


# ======================================================================
# [3] 梯度检验（收敛性判据）
# ======================================================================
def _grad_residual(N: int, kappa: float = 1.0) -> float:
    """在给定 N 下，返回梯度检验的最大相对残差。"""
    m = ocp.QubitReset(tau=0.15, N=N, gamma=10)
    m.homotopy_value = kappa
    tg = m.time_grid(N)
    tau = m.tau
    # 与 N 无关的固定平滑控制形状
    U = m.project(np.vstack([
        1.5 + 0.5 * np.sin(2 * np.pi * tg / tau),
        1.0 + 0.3 * np.cos(2 * np.pi * tg / tau),
    ]))
    J, X, Mu, G = m.cost_gradient(tg, U)
    dt = tg[1] - tg[0]

    worst = 0.0
    for k in (N // 5, N // 3, N // 2, (2 * N) // 3):
        for i in (0, 1):
            Up = U.copy()
            Up[i, k] += H_FD
            Jp, _, _, _ = m.cost_gradient(tg, Up)
            num = (Jp - J) / (H_FD * dt)
            ana = G[i, k]
            worst = max(worst, abs(num - ana) / max(abs(ana), 1e-12))
    return worst


def check_gradient() -> bool:
    print("[3] 梯度检验：残差应收敛到 0（验证解析梯度方向与量级正确）")
    Ns = [40, 80, 160, 320]
    errs = [_grad_residual(N) for N in Ns]
    for N, e in zip(Ns, errs):
        print(f"    N={N:4d}  最大相对残差={e:.3e}")

    orders = []
    for i in range(1, len(Ns)):
        orders.append(np.log(errs[i - 1] / errs[i]) / np.log(2.0))
    print("    收敛阶：" + "  ".join(f"p≈{o:.2f}" for o in orders))

    # 残差单调下降 且 近似一阶（连续伴随 vs 离散伴随的固有差异）
    decreasing = all(errs[i] < errs[i - 1] for i in range(1, len(errs)))
    order_ok = all(0.6 < o < 1.5 for o in orders)
    ok = bool(decreasing and order_ok)
    print(f"    残差单调下降={decreasing}  阶数≈1（连续伴随固有精度）={order_ok}  "
          f"{'OK' if ok else 'MISMATCH'}")
    return ok


def main() -> int:
    print("=== 解析偏导 / 雅可比 / 梯度 的数值微分核对 ===\n")
    m = ocp.QubitReset(tau=0.15, N=6, gamma=10)

    results = [
        ("[1] 解析偏导", check_partials(m)),
        ("[2] 极坐标雅可比", check_jacobian(m)),
        ("[3] 梯度检验", check_gradient()),
    ]
    print()
    allok = True
    for name, ok in results:
        print(f"  {name:22s} {'PASS' if ok else 'FAIL'}")
        allok &= ok
    print(f"\n=== {'全部通过' if allok else '存在不一致，请检查上面标记 MISMATCH 的项'} ===")
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
