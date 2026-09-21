# -*- coding: utf-8 -*-
"""Visualizer —— 梯度迭代过程的实况四级图 + 动画。

MATLAB 版 ``src/+vis/Visualizer.m`` 的 Python 重写。

求解过程中 GradientSolver 每轮都会调用配置好的 observer：

    v = Visualizer(animate=True, video_file='iteration.gif')
    v.init(model, tg)
    s.observer = lambda it, J, X, Mu, U, tg: v.snapshot(it, J, X, Mu, U, tg)
    res = s.solve()
    v.finish()

四个子图：(1) 状态 (2) 协态 (3) 控制 (4) 成本历史。
曲线数量会自动适配 model.n_state / model.n_control，所以同一个可视化器
既能服务量子比特问题，也能服务标量例子。

动画格式：
    'gif'   通用可看（默认）
    'mp4'   需要系统里有 ffmpeg 可执行文件

与 MATLAB 的差异
----------------
* MATLAB 默认 ``showLive = true``（弹出现场图窗）。Python 版默认改成
  ``show_live = False``：无图形界面的环境（服务器、自动化脚本）下更安全，
  需要现场窗口时显式传 ``show_live=True``。
* 帧缓冲用纯 Agg 后端（``FigureCanvasAgg``）渲染，因此**不需要 GUI**，
  也能在没有显示器的机器上导出 GIF。
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from ..ocp.abstract_ocp import AbstractOCP


class Visualizer:
    """四级图 + 逐迭代录制的可视化器。"""

    def __init__(
        self,
        animate: bool = False,
        format: str = "gif",
        video_file: str = "iteration.gif",
        fps: int = 12,
        show_live: bool = False,
        xlabel: str = "time",
    ) -> None:
        # 对应 MATLAB properties 里的默认值
        self.animate = animate
        self.format = format                      # 'gif' | 'mp4'
        self.video_file = video_file
        self.fps = fps
        self.show_live = show_live
        self.xlabel = xlabel

        # ---- 私有状态（对应 MATLAB properties (Access = private)） ----
        self._fig = None
        self._canvas = None
        self._ax = []
        self._jhist: list[float] = []
        self._frames: list[np.ndarray] = []
        self._name = ""
        self._plt = None      # 仅在 show_live 时才导入 pyplot

    # ------------------------------------------------------------------
    def set(self, **kwargs) -> "Visualizer":
        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise AttributeError(f"Visualizer 没有属性 '{key}'")
            setattr(self, key, value)
        return self

    # ==================================================================
    # 生命周期
    # ==================================================================
    def init(self, model: AbstractOCP, tg: np.ndarray) -> None:
        """建图窗、开动画接收端。对应 MATLAB 的 ``init``。"""
        self._name = model.name
        self._jhist = []
        self._frames = []

        # 纯 Agg：不需要 GUI 也能画图和抓帧
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        self._fig = Figure(figsize=(12.0, 8.0), dpi=100, facecolor="white")
        self._canvas = FigureCanvasAgg(self._fig)

        self._ax = [self._fig.add_subplot(2, 2, p + 1) for p in range(4)]
        for ax in self._ax:
            ax.grid(True)
        self._ax[0].set_xlabel(self.xlabel)
        self._ax[0].set_ylabel("state")
        self._ax[1].set_xlabel(self.xlabel)
        self._ax[1].set_ylabel("costate")
        self._ax[2].set_xlabel(self.xlabel)
        self._ax[2].set_ylabel("control")
        self._ax[3].set_xlabel("iteration")
        self._ax[3].set_ylabel("cost")

        if self.show_live:
            # 需要现场图窗时才导入 pyplot（可能依赖 GUI 后端）
            import matplotlib.pyplot as plt

            self._plt = plt
            plt.ion()
            plt.show(block=False)

    def snapshot(
        self,
        iter: int,
        J: float,
        X: np.ndarray,
        Mu: np.ndarray,
        U: np.ndarray,
        tg: np.ndarray,
    ) -> None:
        """刷新四个面板并记录当前迭代。对应 MATLAB 的 ``snapshot``。"""
        self._jhist.append(float(J))
        self._draw_panels(iter, J, X, Mu, U, tg)
        if self.animate:
            self._canvas.draw()
            buf = np.asarray(self._canvas.buffer_rgba())
            self._frames.append(buf[:, :, :3].copy())
        if self.show_live and self._plt is not None:
            self._plt.pause(0.001)

    def finish(self) -> None:
        """收尾：把录下的帧写成动画文件。对应 MATLAB 的 ``finish``。"""
        if self.animate:
            self._write_animation()

    # ==================================================================
    # 动画接收端
    # ==================================================================
    def _write_animation(self) -> None:
        if not self._frames:
            return
        fmt = (self.format or "gif").lower()
        if fmt == "mp4":
            self._write_mp4()
        else:
            self._write_gif()

    def _write_gif(self) -> None:
        """用 Pillow 写 GIF（对应 MATLAB 里 rgb2ind + imwrite(...,'gif') 的循环）。"""
        from PIL import Image

        images = [Image.fromarray(f) for f in self._frames]
        duration_ms = int(round(1000.0 / max(self.fps, 1)))
        images[0].save(
            self.video_file,
            save_all=True,
            append_images=images[1:],
            duration=duration_ms,
            loop=0,           # 无限循环，对应 MATLAB 的 'LoopCount', inf
        )
        print(f"GIF 已写出: {os.path.abspath(self.video_file)}  ({len(images)} 帧)")

    def _write_mp4(self) -> None:
        """写 MP4。Python 需要 ffmpeg —— 没有就给出明确提示。"""
        import shutil
        import tempfile

        if shutil.which("ffmpeg") is None:
            raise RuntimeError(
                "导出 MP4 需要系统里有 ffmpeg 可执行文件。\n"
                "  方案 A：安装 ffmpeg（例如 winget install Gyan.FFmpeg）\n"
                "  方案 B：把 format 改成 'gif'（无需外部依赖）"
            )
        import subprocess

        from PIL import Image

        with tempfile.TemporaryDirectory() as td:
            for i, f in enumerate(self._frames):
                Image.fromarray(f).save(os.path.join(td, f"f{i:05d}.png"))
            cmd = [
                "ffmpeg", "-y", "-framerate", str(self.fps),
                "-i", os.path.join(td, "f%05d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                self.video_file,
            ]
            subprocess.run(cmd, check=True, capture_output=True)
        print(f"MP4 已写出: {os.path.abspath(self.video_file)}")

    # ==================================================================
    # 绘图
    # ==================================================================
    def _draw_panels(
        self,
        iter: int,
        J: float,
        X: np.ndarray,
        Mu: np.ndarray,
        U: np.ndarray,
        tg: np.ndarray,
    ) -> None:
        """重画四个面板。对应 MATLAB 的 ``drawPanels``。"""
        ns = X.shape[0]
        nc = U.shape[0]

        # MATLAB: if numel(Jhist) > 400, idx = 1:400; else idx = 1:numel(Jhist)
        n_hist = len(self._jhist)
        idx = np.arange(1, min(n_hist, 400) + 1)
        jvals = np.asarray(self._jhist[: idx.size], dtype=float)

        # ---- (1) 状态 ----
        ax = self._ax[0]
        ax.clear()
        ax.grid(True)
        for i in range(ns):
            ax.plot(tg, X[i, :], linewidth=1.5, label=f"p_{i + 1}")
        ax.set_xlim(0.0, float(tg[-1]))
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel("state")
        ax.legend(loc="best", fontsize=8)

        # ---- (2) 协态 ----
        ax = self._ax[1]
        ax.clear()
        ax.grid(True)
        for i in range(ns):
            ax.plot(tg, Mu[i, :], linewidth=1.5, label=f"mu_{i + 1}")
        ax.set_xlim(0.0, float(tg[-1]))
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel("costate")
        ax.legend(loc="best", fontsize=8)

        # ---- (3) 控制 ----
        ax = self._ax[2]
        ax.clear()
        ax.grid(True)
        for i in range(nc):
            ax.plot(tg, U[i, :], linewidth=1.5, label=f"u_{i + 1}")
        ax.set_xlim(0.0, float(tg[-1]))
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel("control")
        ax.legend(loc="best", fontsize=8)

        # ---- (4) 成本历史（对数纵轴） ----
        ax = self._ax[3]
        ax.clear()
        ax.grid(True)
        # 代价可能取到 0 或负值，semilogy 会报警告；夹一个正的下限
        safe = np.maximum(jvals, np.finfo(float).tiny)
        ax.semilogy(idx, safe, "k-", linewidth=1.5)
        lo, hi = float(safe.min()), float(safe.max())
        if lo < hi:
            ax.set_ylim(lo * 0.9, hi * 1.1)
        ax.set_xlabel("iteration")
        ax.set_ylabel("cost")
        ax.set_title(f"iter {iter}   J = {J:.4e}", fontsize=10)
        self._fig.suptitle(f"{self._name} -- iter {iter}", fontsize=11)
