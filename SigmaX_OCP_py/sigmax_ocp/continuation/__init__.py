# -*- coding: utf-8 -*-
"""continuation 子包 —— 同伦延拓（对应 MATLAB 的 ``+continuation`` 包）。"""

from .homotopy_continuation import (
    ContinuationResult,
    ContinuationStep,
    HomotopyContinuation,
)

__all__ = ["HomotopyContinuation", "ContinuationResult", "ContinuationStep"]
