# -*- coding: utf-8 -*-
"""ocp 子包 —— 最优控制问题（对应 MATLAB 的 ``+ocp`` 包）。

    AbstractOCP        抽象接口（可替换问题的契约）
    QubitReset         σx 量子比特重置问题（核心，手写 RK4 动力学）
    QubitResetQuTiP    同一问题，但动力学改用 QuTiP（可选，需要装 QuTiP）
    ControlExample     另一个可替换问题（descriptor 例 1）
"""

from .abstract_ocp import AbstractOCP
from .control_example import ControlExample
from .qubit_reset import QubitReset

__all__ = ["AbstractOCP", "QubitReset", "ControlExample"]

# ----------------------------------------------------------------------
# QuTiP 动力学版是**可选**的：装了 QuTiP 才导出，没装也不影响其余功能。
# （qubit_reset_qutip 模块内部已经把 import qutip 包在 try/except 里了。）
# ----------------------------------------------------------------------
try:
    from .qubit_reset_qutip import QubitResetQuTiP

    __all__.append("QubitResetQuTiP")
except ImportError:      # pragma: no cover  —— 未安装 QuTiP
    pass
