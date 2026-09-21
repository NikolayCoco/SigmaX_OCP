# -*- coding: utf-8 -*-
"""analysis_p_matrix.py —— 把 QubitReset 的 P 矩阵拆成"驱动 + 耗散"两部分。

思路：
  任何线性生成元都可以唯一拆成  A = A_sym + A_asym
    A_asym（反对称）-> 对应量子力学的幺正演化 -i[H,·]，即 Bloch 旋转
    A_sym （对称）  -> 对应耗散通道（Lindblad 的 D[L] 项）

  本脚本：
   [1] 从 A_asym 反解 h，验证 H = (h·σ)/2 的具体形式
   [2] 把 A_sym 分解成"各向同性衰减 + 绕 z 轴旋转 2θ 的各向异性"
      并检验交叉项是否真为旋转形式（这是判断能否用固定 c_ops 的关键）
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from sigmax_ocp import ocp

m = ocp.QubitReset(tau=0.15, N=10, gamma=10)

print("=" * 78)
print("[1] 从反对称部分反解量子哈密顿量 H")
print("=" * 78)
print("  记 A = P 的 3x3 部分；A_asym = (A - Aᵀ)/2 应当等于旋转矩阵 [h]_× :")
print("      [h]_× = [[0, -hz, hy], [hz, 0, -hx], [-hy, hx, 0]]")
print("  于是 h 可由 A_asym 直接读出： hx = A_asym[2,1], hy = A_asym[0,2], hz = A_asym[1,0]")
print()

for (th, la) in [(0.3, 1.0), (0.9, 2.0), (1.2, 0.5), (0.75, 3.0)]:
    A, b = m._mat_vec(th, la)
    A_asym = (A - A.T) / 2.0

    h = np.array([A_asym[2, 1], A_asym[0, 2], A_asym[1, 0]])
    lam_x = la * np.sin(th)
    lam_z = la * np.cos(th)

    expect = np.array([-2 * lam_z, -2 * lam_x, 0.0])
    print(f"  θ={th:<5} Λ={la:<4}  λx={lam_x:+.6f}  λz={lam_z:+.6f}")
    print(f"      反解 h        = [{h[0]:+.8f}, {h[1]:+.8f}, {h[2]:+.8f}]")
    print(f"      猜测 (-2λz,-2λx,0) = [{expect[0]:+.8f}, {expect[1]:+.8f}, {expect[2]:+.8f}]"
          f"   最大差 {np.abs(h - expect).max():.2e}")

print()
print("  → 若最大差 ≈ 1e-15，说明 H = (h·σ)/2 = -λz·σx - λx·σy 是**严格成立**的。")

print()
print("=" * 78)
print("[2] 对称（耗散）部分的结构分析")
print("=" * 78)
print("  横向 2x2 块写成  S = -P·I + [[Q·cos2θ, R·sin2θ], [R·sin2θ, -Q·cos2θ]]")
print("  的形式，其中 P 是各向同性衰减、Q/R 是各向异性幅度。")
print("  **关键判据：Q 是否等于 R** —— 只有 Q=R 时才是纯粹的'旋转算符'形式。")
print()

for (th, la) in [(0.3, 1.0), (0.9, 2.0), (1.2, 0.5), (0.75, 3.0)]:
    A, b = m._mat_vec(th, la)
    S = (A + A.T) / 2.0

    c2 = np.cos(2 * th)
    s2 = np.sin(2 * th)

    # S[0,0] = -P + Q*c2 ;  S[1,1] = -P - Q*c2 ;  S[0,1] = R*s2
    if abs(c2) > 1e-9:
        Q = (S[0, 0] - S[1, 1]) / (2 * c2)
    else:
        Q = float("nan")
    R = S[0, 1] / s2 if abs(s2) > 1e-9 else float("nan")
    P = -(S[0, 0] + S[1, 1]) / 2.0

    gamma = 10.0
    Gz = float(m._gz(th, la))
    Gp = float(m._gp(th, la))
    Gm = float(m._gm(th, la))

    print(f"  θ={th:<5} Λ={la:<4}")
    print(f"      P（各向同性） = {P:+.8f}      -γ(2Γz+1.5(Γ++Γ-)) = "
          f"{-gamma * (2 * Gz + 1.5 * (Gp + Gm)):+.8f}")
    print(f"      Q（各向异性） = {Q:+.8f}      -γ(2Γz-0.5(Γ++Γ-)) = "
          f"{-gamma * (2 * Gz - 0.5 * (Gp + Gm)):+.8f}")
    print(f"      R（交叉项）   = {R:+.8f}      -γ(2Γz-0.5(Γ++Γ-)) = "
          f"{-gamma * (2 * Gz - 0.5 * (Gp + Gm)):+.8f}")
    print(f"      Q - R = {Q - R:+.3e}   {'相等 -> 纯旋转形式' if abs(Q - R) < 1e-9 else '不等'}")
    print(f"      S[2,2] = {S[2, 2]:+.8f}    -γ(4Γz+Γ++Γ-) = "
          f"{-gamma * (4 * Gz + Gp + Gm):+.8f}")
    print()
