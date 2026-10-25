"""PROSPECT in PyTensor (for PyMC, nutpie, and NumPyro via PyTensor's JAX backend).

E1 modes:
  "exact"  - custom Op: value from Cephes exp1 (scipy; jax.scipy.special.exp1
             under JAX), gradient supplied analytically, dE1/dx = -exp(-x)/x.
  "fast"   - custom Op: NAG S13AAF Chebyshev expansions (see
             prospect_np.e1_fast; rel. error ~2e-12), same analytic gradient.
  "approx" - Swamee & Ohija (2003) closed form, differentiated by PyTensor.

Both custom Ops ship native kernels for PyTensor's numba backend (default;
also used by nutpie) and its JAX backend (NumPyro).
"""

from __future__ import annotations

import numpy as np
import pytensor.tensor as pt
from pytensor.graph.basic import Apply
from pytensor.graph.op import Op
from scipy.special import exp1

from prospect_np import E1_FAST_HI, E1_FAST_LO, e1_fast as _e1_fast_np


class _E1Base(Op):
    """Elementwise E1(x) with the analytic gradient dE1/dx = -exp(-x)/x."""

    __props__ = ()

    def make_node(self, x):
        x = pt.as_tensor_variable(x)
        return Apply(self, [x], [x.type()])

    def infer_shape(self, fgraph, node, shapes):
        return shapes

    def pullback(self, inputs, outputs, cotangents):
        (x,) = inputs
        (gz,) = cotangents
        return [gz * (-pt.exp(-x) / x)]


class E1(_E1Base):
    """Exact E1 (Cephes)."""

    def perform(self, node, inputs, output_storage):
        output_storage[0][0] = np.asarray(exp1(inputs[0]), dtype=node.outputs[0].dtype)


class E1Fast(_E1Base):
    """Fast E1 (NAG S13AAF Chebyshev expansions)."""

    def perform(self, node, inputs, output_storage):
        output_storage[0][0] = np.asarray(_e1_fast_np(inputs[0]), dtype=node.outputs[0].dtype)


e1_exact = E1()
e1_fast = E1Fast()


def _register_numba():
    """Native numba kernels, so PyTensor's numba backend (and nutpie) never
    fall back to object mode."""
    try:
        import ctypes

        import numba
        from numba.extending import get_cython_function_address
        from pytensor.link.numba.dispatch.basic import register_funcify_default_op_cache_key
    except ImportError:  # numba backend unavailable; perform() is used
        return
    addr = get_cython_function_address("scipy.special.cython_special", "__pyx_fuse_1exp1")
    _exp1_c = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)(addr)

    @register_funcify_default_op_cache_key(E1)
    def _numba_funcify_E1(op, node=None, **kwargs):
        @numba.njit(error_model="numpy")
        def e1_numba(x):
            out = np.empty_like(x)
            xf = x.ravel()
            of = out.ravel()
            for i in range(xf.size):
                of[i] = _exp1_c(xf[i])
            return out

        # The integer salts PyTensor's on-disk numba cache key; bump it whenever
        # the kernel changes (the default key ignores the kernel's code).
        return e1_numba, 2

    lo, hi = E1_FAST_LO, E1_FAST_HI

    @register_funcify_default_op_cache_key(E1Fast)
    def _numba_funcify_E1Fast(op, node=None, **kwargs):
        # error_model="numpy": x = 0 or inf gives inf/nan (-> rejected NUTS
        # proposal) instead of raising ZeroDivisionError; no fastmath, since
        # it would let LLVM assume inputs are finite.
        @numba.njit(error_model="numpy")
        def e1_fast_numba(x):
            out = np.empty_like(x)
            xf = x.ravel()
            of = out.ravel()
            for i in range(xf.size):
                xi = xf[i]
                if xi <= 4.0:
                    t = 0.5 * xi - 1.0
                    y = 0.0
                    for c in lo:
                        y = y * t + c
                    of[i] = y - np.log(xi)
                else:
                    t = 14.5 / (xi + 3.25) - 1.0
                    y = 0.0
                    for c in hi:
                        y = y * t + c
                    of[i] = np.exp(-xi) * y / xi
            return out

        return e1_fast_numba, 2


def _register_jax():
    """JAX kernels with the same analytic derivative (custom_jvp)."""
    try:
        import jax
        import jax.numpy as jnp
        from jax.scipy.special import exp1 as jexp1
        from pytensor.link.jax.dispatch import jax_funcify
    except ImportError:
        return

    def with_analytic_jvp(f):
        f = jax.custom_jvp(f)

        @f.defjvp
        def _jvp(primals, tangents):
            (x,), (dx,) = primals, tangents
            return f(x), -jnp.exp(-x) / x * dx

        return f

    def _fast(x):
        tl = 0.5 * x - 1.0
        th = 14.5 / (x + 3.25) - 1.0
        yl = jnp.zeros_like(x)
        yh = jnp.zeros_like(x)
        for c in E1_FAST_LO:
            yl = yl * tl + c
        for c in E1_FAST_HI:
            yh = yh * th + c
        return jnp.where(x <= 4.0, yl - jnp.log(x), jnp.exp(-x) * yh / x)

    e1_exact_jax = with_analytic_jvp(jexp1)
    e1_fast_jax = with_analytic_jvp(_fast)

    @jax_funcify.register(E1)
    def _jax_funcify_E1(op, **kwargs):
        return e1_exact_jax

    @jax_funcify.register(E1Fast)
    def _jax_funcify_E1Fast(op, **kwargs):
        return e1_fast_jax


_register_numba()
_register_jax()


def e1_approx(x):
    a = pt.log((0.56146 / x + 0.65) * (1 + x))
    b = x**4 * pt.exp(7.7 * x) * (2 + x) ** 3.7
    return (a**-7.7 + b) ** -0.13


E1_MODES = {"exact": e1_exact, "fast": e1_fast, "approx": e1_approx}


def gpm(k, N, talf, t12, t21, e1="exact", want_t=False):
    """Generalized plate model. Assumes k > 0 (true whenever Cw or Cm > 0)."""
    e1v = E1_MODES[e1](k)
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
