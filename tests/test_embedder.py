import jax.numpy as jnp
import pytest
import flt
from warper import legendre_embed_batch


@pytest.mark.parametrize(
    "f, expect",
    [
        (lambda x: jnp.ones_like(x), lambda k: jnp.where(k == 0, jnp.sqrt(1 / 2), 0.0)),
        (lambda x: x, lambda k: jnp.where(k == 1, jnp.sqrt(3 / 2), 0.0)),
    ],
)
def test_analytic(f, expect):
    L = 1024
    x = jnp.linspace(-1, 1, L)
    vec = legendre_embed_batch(f(x)[None, :], n_coeffs=8, normalise=False)[0]
    k = jnp.arange(vec.size)
    assert jnp.allclose(vec, expect(k), atol=1e-6)


def test_roundtrip():
    L = 1024
    xu = jnp.linspace(-1, 1, L)
    f = jnp.exp(-(xu**2))

    coeffs = legendre_embed_batch(f[None, :], n_coeffs=L, normalise=False)[0]
    coeffs /= jnp.sqrt((2 * jnp.arange(L) + 1) / 2)

    # Values actually sent into the DLT
    theta = flt.theta(L)  # (0,π)   size‑L
    x_leg = jnp.cos(theta)  # same order flt expects
    f_leg = jnp.interp(x_leg, xu, f)

    f_hat = flt.idlt(coeffs)  # comes back *in the same order*

    assert jnp.allclose(f_hat, f_leg, atol=1e-6, rtol=0)


def test_parseval():
    from numpy.polynomial.legendre import leggauss

    L = 1024
    xu = jnp.linspace(-1, 1, L)
    f = jnp.exp(-(xu**2))
    a_flt = legendre_embed_batch(f[None, :], n_coeffs=L, normalise=False)[0]

    k = jnp.arange(L)
    wt = 4.0 / (2 * k + 1) ** 2  # Gauss–Legendre Parseval weights
    E_spec = jnp.sum(a_flt**2 * wt)

    # ---- energy in time domain (same GL grid & weights) --------------
    nodes, w = leggauss(L)  # Gauss‑Legendre nodes/weights
    f_leg = jnp.interp(nodes, xu, f)  # data actually transformed
    E_time = jnp.sum(w * f_leg**2)

    assert jnp.allclose(E_time, E_spec, rtol=1e-6)
