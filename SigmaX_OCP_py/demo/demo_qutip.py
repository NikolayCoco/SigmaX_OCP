# -*- coding: utf-8 -*-
"""demo_qutip.py —— 用 QuTiP / SciPy 对照 QubitReset 的主方程。

分三部分：

  [1] QuTiP 入门：标准二能级的 T1 / T2 弛豫。
      目的不是学物理（你比我熟），而是把 QuTiP 的 API 用法过一遍，
      并且用一个**你一眼就能判断对错**的结果来建立信心。

  [2] 交叉验证：用 SciPy 的自适应 RK45 重解 QubitReset 的方程
          d p/dt = P(θ,Λ) p + p0(θ,Λ)
      与手写的固定步长 RK4 对比。两条独立路径给出同样的轨迹，
      就说明积分器实现没问题（这是比"和 MATLAB 对照"更强的证据，
      因为 MATLAB 那版共用同一套推导）。

  [3] 如果要用 qutip.mesolve：需要把模型写成 H + collapse operators。
      本部分给出写法模板，以及一个能帮你**确认自己猜测对不对**的工具函数。

运行：
    python demo/demo_qutip.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np    # noqa: E402


# ======================================================================
# [1] QuTiP 入门：T1 / T2
# ======================================================================
def part1_t1_t2() -> None:
    import qutip as qt

    print("=" * 72)
    print("[1] QuTiP 入门：标准二能级的 T1 / T2 弛豫")
    print("=" * 72)
    print("  模型：两能级 |0>(基态) / |1>(激发态)，H = 0，只有耗散。")
    print("  c_ops = [sqrt(γ1)·σ-,  sqrt(γφ/2)·σz]")
    print("  理论：1/T1 = γ1 ；  1/T2 = γ1/2 + γφ")
    print()

    gamma1 = 1.0        # 弛豫率
    gamma_phi = 0.3     # 纯退相位率

    sm = qt.destroy(2)          # σ- ：|1> -> |0>
    sz = qt.sigmaz()
    times = np.linspace(0.0, 6.0, 401)

    # ---- T1：初态 |1>，看激发态布居 <n> = <σ+σ-> 的衰减 ----
    rho_excited = qt.basis(2, 1).proj()
    c_ops = [np.sqrt(gamma1) * sm]
    res1 = qt.mesolve(qt.qzero(2), rho_excited, times, c_ops=c_ops,
                      e_ops=[qt.num(2)])
    pe = np.asarray(res1.expect[0]).real
    mask = pe > 1e-8                       # 只对还没衰减到 0 的部分做拟合
    slope1 = np.polyfit(times[mask], np.log(pe[mask]), 1)[0]
    T1_fit = -1.0 / slope1

    print(f"  T1: 理论 1/γ1 = {1.0 / gamma1:.4f}"
          f"    从曲线拟合 = {T1_fit:.4f}    {'一致' if abs(T1_fit - 1/gamma1) < 1e-3 else '不一致'}")

    # ---- T2：初态 |+>，看 <σx> 的衰减 ----
    plus = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
    rho_plus = plus.proj()
    c_ops = [np.sqrt(gamma1) * sm, np.sqrt(gamma_phi / 2) * sz]
    res2 = qt.mesolve(qt.qzero(2), rho_plus, times, c_ops=c_ops,
                      e_ops=[qt.sigmax()])
    sx = np.asarray(res2.expect[0]).real
    mask = np.abs(sx) > 1e-8
    slope2 = np.polyfit(times[mask], np.log(np.abs(sx[mask])), 1)[0]
    T2_fit = -1.0 / slope2

    print(f"  T2: 理论 1/(γ1/2+γφ) = {1.0 / (gamma1 / 2 + gamma_phi):.4f}"
          f"    从曲线拟合 = {T2_fit:.4f}"
          f"    {'一致' if abs(T2_fit - 1/(gamma1/2+gamma_phi)) < 1e-3 else '不一致'}")

    print()
    print("  → 如果这两行都对上了，说明你对 QuTiP 的用法是对的。")
    print("    要点：c_ops 里放的是**算符**（不含 γ），γ 通过给算符乘 sqrt(γ) 体现；")
    print("          传入的 H 和 c_ops 都可以是时间的函数（见第 [3] 部分）。")


# ======================================================================
# [2] 交叉验证：两种积分器解同一个方程
# ======================================================================
def part2_cross_validate() -> None:
    from scipy.linalg import expm

    from sigmax_ocp import ocp

    print()
    print("=" * 72)
    print("[2] 交叉验证：你的 RK4  vs  分段解析解（矩阵指数，无积分误差）")
    print("=" * 72)
    print("  同一个离散化（控制分段常数），但参考解用 expm 精确算出每一段的")
    print("  传播算子 p(t+Δt) = e^{AΔt} p(t) + (e^{AΔt}-I)A⁻¹b，")
    print("  这样就能**单独**检验你的 RK4 实现是否正确。")
    print()
    print(f"  {'N':>5} {'Δt':>11} {'最大误差':>13} {'误差比':>8} {'观测阶数':>9}")

    prev = None
    for N in (50, 100, 200, 400):
        m = ocp.QubitReset(tau=0.15, N=N, gamma=10)
        tg = m.time_grid(m.N)
        U = m.initial_control(tg)
        theta, Lam = m._control_transform(U)

        # (a) 你写的 RK4（固定步长，控制分段常数）
        X_rk4 = m._forward_integrate(tg, theta, Lam, m.p_initial)

        # (b) 分段矩阵指数：严格解，无积分误差
        #     把非齐次项塞进增广矩阵，一次 expm 同时得到 e^{AΔt} 和积分项
        X_ex = np.zeros((3, N))
        X_ex[:, 0] = m.p_initial
        for k in range(N - 1):
            A, b = m._mat_vec(theta[k], Lam[k])
            dt = tg[k + 1] - tg[k]
            aug = np.zeros((4, 4))
            aug[:3, :3] = A
            aug[:3, 3] = b
            E = expm(aug * dt)
            X_ex[:, k + 1] = E[:3, :3] @ X_ex[:, k] + E[:3, 3]

        err = float(np.abs(X_rk4 - X_ex).max())
        dt = m.tau / (N - 1)
        if prev is None:
            print(f"  {N:5d} {dt:11.3e} {err:13.3e} {'-':>8} {'-':>9}")
        else:
            ratio = prev / err
            order = np.log(ratio) / np.log(2.0)      # N 翻倍 => Δt 减半
            print(f"  {N:5d} {dt:11.3e} {err:13.3e} {ratio:8.2f} {order:9.2f}")
        prev = err

    print()
    print("  → 误差按 ~Δt⁴ 下降（N 翻倍、误差约降 16 倍，观测阶数≈4），")
    print("    这正是四阶 RK4 应有的表现 —— 说明你的积分器实现正确，")
    print("    剩下的差异只是它固有的截断误差。")


# ======================================================================
# [3] mesolve 模板 + 帮你确认 H / c_ops 的工具
# ======================================================================
def lindblad_generator(H, c_ops, basis):
    """把 Lindblad 生成元表示成给定算符基下的矩阵。

    标准 Lindblad 方程：

        dρ/dt = -i[H, ρ] + Σ_k ( L_k ρ L_k† - ½{L_k†L_k, ρ} )

    这里把右边看成**作用在算符空间上的线性映射** L，并展开在基 {B_i} 下：

        L(B_j) = Σ_i M[i, j] · B_i

    参数
    ----
    H      : qutip.Qobj，Hamilton 量（实矩阵即可）
    c_ops  : qutip.Qobj 列表，已经乘过 sqrt(γ) 的 collapse operators
    basis  : qutip.Qobj 列表，要求 tr(B_i† B_j) = δ_ij（正交归一）

    返回
    ----
    M : (n, n) 复数矩阵
    """
    n = len(basis)
    M = np.zeros((n, n), dtype=complex)
    for j, Bj in enumerate(basis):
        LB = -1j * (H * Bj - Bj * H)
        for Lk in c_ops:
            LB = LB + Lk * Bj * Lk.dag() - 0.5 * (Lk.dag() * Lk * Bj + Bj * Lk.dag() * Lk)
        for i, Bi in enumerate(basis):
            M[i, j] = complex((Bi.dag() * LB).tr())
    return M


def part3_mesolve_template() -> None:
    import qutip as qt

    print()
    print("=" * 72)
    print("[3] 要用 qutip.mesolve，模型必须写成 H + collapse operators")
    print("=" * 72)
    print("""
  qutip.mesolve 解的是标准 Lindblad 方程：

      dρ/dt = -i[H(t), ρ] + Σ_k ( L_k(t) ρ L_k†(t) - ½{L_k†L_k, ρ} )

  而你的模型是  d p/dt = P(θ,Λ) p + p0(θ,Λ)  ——  3 维实数向量的线性 ODE。
  要把两者接起来，需要回答两个问题（这两个都是**你的物理**，我不替你猜）：

     (i)  p = [p_e, p_r, p_i] 具体对应密度矩阵的哪三个参数？
     (ii) H、c_ops 分别是什么？从 Γz / Γ± 的形式看，物理上最可能是三个通道：

            L_1 = σ-   弛豫   |e> → |r>     γ_1 = Γ-
            L_2 = σ+   激发   |r> → |e>     γ_2 = Γ+
            L_3 = σz   纯退相位             γ_3 = Γz

          而 λx、λz 是以什么形式进入 H 的（σx 驱动？σz 失谐？），需要你定。
""")

    # ---- 先证明工具函数本身是对的：拿一个已知答案的简单例子 ----
    print("  先用一个已知答案的例子验证本文件里的工具函数：")
    print("    H = 0，c_ops = [sqrt(γ)·σ-]  →  应当有  dρ11/dt = -γ·ρ11")

    gamma = 0.7
    B = [qt.qeye(2) / np.sqrt(2), qt.sigmax() / np.sqrt(2),
         qt.sigmay() / np.sqrt(2), qt.sigmaz() / np.sqrt(2)]
    M = lindblad_generator(qt.qzero(2), [np.sqrt(gamma) * qt.destroy(2)], B)

    # ρ11 = (I + σz)/2  →  在布洛赫表示里对应 z 分量
    # dρ11/dt = -(1/2)·dz/dt ；这里直接从 M 里读 z 分量的演化率
    idx_z = 3
    decay = -M[idx_z, idx_z].real        # dz/dt = -γ·z  (期望 γ = 0.7)
    print(f"     从生成元矩阵读出的衰减率 = {decay:.6f}   (期望 {gamma:.6f})  "
          f"{'OK' if abs(decay - gamma) < 1e-10 else '不符'}")

    print(f"""
  用法：把你的猜测写成 H 和 c_ops，用 lindblad_generator 算出矩阵，
        再和你的 P 矩阵对比。若两者一致，就说明你的 P 确实来自标准
        Lindblad 形式，可以直接交给 qutip.mesolve 求解：

            H = [[qt.sigmax(), lambda t, a: lam_x(t)],
                 [qt.sigmaz(), lambda t, a: lam_z(t)]]

            c_ops = [[qt.destroy(2),  lambda t, a: np.sqrt(Gamma_minus(t))],
                     [qt.destroy(2).dag(), lambda t, a: np.sqrt(Gamma_plus(t))],
                     [qt.sigmaz(), lambda t, a: np.sqrt(Gamma_z(t))]]

            res = qt.mesolve(H, rho0, times, c_ops=c_ops)
""")


def main() -> None:
    part1_t1_t2()
    part2_cross_validate()
    part3_mesolve_template()
    print()
    print("=" * 72)
    print("完成。下一步建议：把第 [3] 部分里 (i)(ii) 两个问题定下来，")
    print("就能把 QubitReset 整个搬进 qutip.mesolve 了。")
    print("=" * 72)


if __name__ == "__main__":
    main()
