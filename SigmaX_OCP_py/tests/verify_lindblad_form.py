# -*- coding: utf-8 -*-
"""verify_lindblad_form.py —— 验证 QubitReset 的 P 矩阵能否写成 Lindblad 形式。

从 analysis_p_matrix.py 已确定：
  H = (h·σ)/2,  h = (-2λz, -2λx, 0)            （反对称部分，零误差）

耗散的横向 2x2 块可以严格写成
  S = -P·I + Q·R(2θ),  R(2α) = [[cos2α, sin2α],[sin2α, -cos2α]]
这正是"横向算符 L = cosθ·σx - sinθ·σy"产生的形状（D[L] 是 L 的二次型）。

本脚本用下面四个通道去拟合：
  L1 = cosθ·σx - sinθ·σy   率 a      （随驱动方向旋转的横向退相位）
  L2 = σz                  率 b      （固定退相位）
  L3 = σ-                  率 c      （弛豫）
  L4 = σ+                  率 d      （激发）

四个未知数 a,b,c,d 由四个条件定出：
  S = -P I + Q R(2θ) 给出 2 个方程；S22 给出 1 个；p0 的 z 分量给出 1 个。

然后**用 QuTiP 的算符代数重建生成元**，逐元素和你的 P、p0 对比。
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import qutip as qt

from sigmax_ocp import ocp

m = ocp.QubitReset(tau=0.15, N=10, gamma=10)

# --- 算符基：{I, σx, σy, σz}（用归一化版本，使 tr(Bi† Bj) = δij） ---
SQ2 = np.sqrt(2.0)
BASIS = [qt.qeye(2) / SQ2, qt.sigmax() / SQ2, qt.sigmay() / SQ2, qt.sigmaz() / SQ2]


def generator(H, c_ops):
    """把 Lindblad 生成元展开在 BASIS 下：L(B_j) = Σ_i M[i,j] B_i。"""
    n = len(BASIS)
    M = np.zeros((n, n), dtype=complex)
    for j, Bj in enumerate(BASIS):
        LB = -1j * (H * Bj - Bj * H)
        for Lk in c_ops:
            LB = LB + Lk * Bj * Lk.dag() - 0.5 * (Lk.dag() * Lk * Bj + Bj * Lk.dag() * Lk)
        for i, Bi in enumerate(BASIS):
            M[i, j] = complex((Bi.dag() * LB).tr())
    return M


def check(theta: float, Lam: float, verbose: bool = True) -> bool:
    A, p0_user = m._mat_vec(theta, Lam)
    lam_x, lam_z = Lam * np.sin(theta), Lam * np.cos(theta)

    A_asym = (A - A.T) / 2.0
    S = (A + A.T) / 2.0

    # ---- 从 A 读出 P, Q, S22 ----
    c2, s2 = np.cos(2 * theta), np.sin(2 * theta)
    Q = (S[0, 0] - S[1, 1]) / (2 * c2)
    R = S[0, 1] / s2
    P = -(S[0, 0] + S[1, 1]) / 2.0
    S22 = S[2, 2]

    # ---- 解 a, b, c, d ----
    # 各通道对 Bloch 生成元的贡献（率 γ 都以"c_op = sqrt(γ)·L"的形式传入）：
    #   旋转横向 L=sinα σx + cosα σy : 横向 -γI + γR(2α)   纵向 -2γ z
    #   固定退相位 L=σz              : 横向 -2γ I          纵向 0
    #   弛豫 L=σ-                    : 横向 -γ/2 I         纵向 -γ z  (+γ 非齐次)
    #   激发 L=σ+                    : 横向 -γ/2 I         纵向 -γ z  (-γ 非齐次)
    #
    # 横向:  -(a + 2b + c/2 + d/2) I + a·R(2α)  =  -P I + Q·R(2θ)
    #   取 α 使 a>0：Q<0 时取 α=θ+π/2（R(2α)=-R(2θ)）得 a=-Q；Q>0 时取 α=θ 得 a=Q
    a = abs(Q)
    # 纵向:  -(2a + c + d) = S22
    c_plus_d = -S22 - 2.0 * a
    # 非齐次(z):  c - d = p0_z
    c_minus_d = p0_user[2]
    c = (c_plus_d + c_minus_d) / 2.0
    d = (c_plus_d - c_minus_d) / 2.0
    # 各向同性:  a + 2b + (c+d)/2 = P
    b = (P - a - c_plus_d / 2.0) / 2.0

    # ---- 用 QuTiP 重建 ----
    sm = qt.destroy(2)
    sp = qt.destroy(2).dag()
    # 旋转算符的方向 β 必须选对，否则对角元的分配和交叉项符号都会反：
    #   Q>0 : L1 = cosθ σx + sinθ σy   （β = π/2 - θ）
    #   Q<0 : L1 = sinθ σx - cosθ σy   （β = -θ，整体符号无关）
    if Q > 0:
        L1 = np.cos(theta) * qt.sigmax() + np.sin(theta) * qt.sigmay()
    else:
        L1 = np.sin(theta) * qt.sigmax() - np.cos(theta) * qt.sigmay()

    H = -lam_z * qt.sigmax() - lam_x * qt.sigmay()      # 注意：H 是算符本身（无 1/2）
    c_ops = [np.sqrt(a) * L1, np.sqrt(b) * qt.sigmaz(),
             np.sqrt(c) * sm, np.sqrt(d) * sp]

    M = generator(H, c_ops)
    A_built = M[1:4, 1:4].real          # Bloch (x,y,z) 子块
    p0_built = M[1:4, 0].real           # 非齐次项（常数部分）

    err_A = np.abs(A_built - A).max()
    err_p0 = np.abs(p0_built - p0_user).max()

    if verbose:
        print(f"  θ={theta:<6} Λ={Lam:<5}")
        print(f"      解得率: a(旋转横向)={a:+.6f}  b(σz)={b:+.6f}  "
              f"c(σ-)={c:+.6f}  d(σ+)={d:+.6f}")
        print(f"      P={P:+.6f} Q={Q:+.6f} R={R:+.6f} (Q-R={Q-R:+.2e}) S22={S22:+.6f}")
        print(f"      齐次部分  |A_重建 - A_原|   = {err_A:.3e}   "
              f"{'OK' if err_A < 1e-9 else 'MISMATCH'}")
        print(f"      非齐次部分 |p0_重建 - p0_原| = {err_p0:.3e}   "
              f"{'OK' if err_p0 < 1e-9 else 'MISMATCH'}")
        print(f"      你的 p0 = [{p0_user[0]:+.6f}, {p0_user[1]:+.6f}, {p0_user[2]:+.6f}]")
        print(f"      重建 p0 = [{p0_built[0]:+.6f}, {p0_built[1]:+.6f}, {p0_built[2]:+.6f}]")
        if not (a > -1e-12 and b > -1e-12 and c > -1e-12 and d > -1e-12):
            print("      ⚠ 有负的率 —— 这套通道分解在该参数点不物理（需要换分解）")
    return (err_A < 1e-9) and (err_p0 < 1e-9)


print("=" * 78)
print("验证：用 H = -λzσx - λxσy 与四个通道，能否复现你的 P 矩阵")
print("=" * 78)
print()
ok = True
for th, la in [(0.3, 1.0), (0.9, 2.0), (1.2, 0.5), (0.75, 3.0), (0.15, 0.8), (1.4, 2.5)]:
    ok &= check(th, la)
    print()

print("=" * 78)
print(f"总体：{'全部复现成功' if ok else '存在不一致'}")
print("=" * 78)
