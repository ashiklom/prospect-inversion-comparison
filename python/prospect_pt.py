"""PROSPECT in PyTensor (for PyMC).

E1 modes:
  "exact"  - custom Op: value from scipy.special.exp1, gradient supplied
             analytically as dE1/dx = -exp(-x)/x.
  "approx" - Swamee & Ohija (2003) closed form, differentiated by PyTensor.
"""

from __future__ import annotations

import numpy as np
import pytensor.tensor as pt
from pytensor.graph.basic import Apply
from pytensor.graph.op import Op
from scipy.special import exp1


class E1(Op):
    """Elementwise exponential integral E1(x) with analytic gradient."""

    __props__ = ()

    def make_node(self, x):
        x = pt.as_tensor_variable(x)
        return Apply(self, [x], [x.type()])

    def perform(self, node, inputs, output_storage):
        (x,) = inputs
        output_storage[0][0] = np.asarray(exp1(x), dtype=node.outputs[0].dtype)

    def infer_shape(self, fgraph, node, shapes):
        return shapes

    def grad(self, inputs, output_grads):
        (x,) = inputs
        (gz,) = output_grads
        return [gz * (-pt.exp(-x) / x)]


e1_exact = E1()


def _register_numba():
    """Give the E1 Op a native numba kernel (scipy's compiled Cephes exp1)
    so PyTensor's default numba backend doesn't fall back to object mode."""
    try:
        import ctypes

        import numba
        from numba.extending import get_cython_function_address
        from pytensor.link.numba.dispatch.basic import (
            register_funcify_default_op_cache_key,
        )
    except ImportError:  # numba backend unavailable; perform() is used
        return
    addr = get_cython_function_address("scipy.special.cython_special", "__pyx_fuse_1exp1")
    _exp1_c = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)(addr)

    @register_funcify_default_op_cache_key(E1)
    def _numba_funcify_E1(op, node=None, **kwargs):
        @numba.njit
        def e1_numba(x):
            out = np.empty_like(x)
            xf = x.ravel()
            of = out.ravel()
            for i in range(xf.size):
                of[i] = _exp1_c(xf[i])
            return out

        return e1_numba


_register_numba()


def e1_approx(x):
    a = pt.log((0.56146 / x + 0.65) * (1 + x))
    b = x**4 * pt.exp(7.7 * x) * (2 + x) ** 3.7
    return (a**-7.7 + b) ** -0.13


def gpm(k, N, talf, t12, t21, e1="exact", want_t=False):
    """Generalized plate model. Assumes k > 0 (true whenever Cw or Cm > 0)."""
    e1v = e1_exact(k) if e1 == "exact" else e1_approx(k)
    tau = pt.maximum((1 - k) * pt.exp(-k) + k**2 * e1v, 0.0)
    ralf = 1 - talf
    r12 = 1 - t12
    r21 = 1 - t21
    denom = 1 - r21**2 * tau**2
    Ta = talf * tau * t21 / denom
    Ra = ralf + r21 * tau * Ta
    t = t12 * tau * t21 / denom
    r = r12 + r21 * tau * t
    D = pt.sqrt((1 + r + t) * (1 + r - t) * (1 - r + t) * (1 - r - t))
    va = (1 + r**2 - t**2 + D) / (2 * r)
    vb = (1 - r**2 + t**2 + D) / (2 * t)
    vbNN = vb ** (N - 1)
    vbNN2 = vbNN**2
    denomx = va**2 * vbNN2 - 1
    Rsub = va * (vbNN2 - 1) / denomx
    denomy = 1 - Rsub * r
    R = Ra + Ta * Rsub * t / denomy
    if not want_t:
        return R
    Tsub = vbNN * (va**2 - 1) / denomx
    return R, Ta * Tsub / denomy


def prospect(N, C, K, talf, t12, t21, e1="exact", want_t=False):
    """PROSPECT for structure parameter N and constituent vector C (columns of K)."""
    k = pt.dot(K, C) / N
    return gpm(k, N, talf, t12, t21, e1=e1, want_t=want_t)
