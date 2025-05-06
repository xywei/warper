from functools import partial

import jax
import jax.numpy as jnp
import flt


@partial(jax.jit, static_argnames=("n_coeffs", "closed", "normalise"))
def legendre_embed_batch(
    windows: jax.Array,
    *,
    n_coeffs: int,
    closed: bool = False,
    normalise: bool = True,
) -> jax.Array:
    """
    Fast Legendre‑polynomial embedding for a batch of time‑series windows.

    Parameters
    ----------
    windows   : jax.Array[..., L]  – uniform‑grid samples (batchable).
    n_coeffs  : int                – number of leading Legendre coeffs to keep.
    closed    : bool               – choose open vs closed DLT (flt docs).
    normalise : bool               – ℓ₂‑normalise each embedding.

    Returns
    -------
    coeffs : jax.Array[..., n_coeffs]  – Legendre embeddings.
    """
    L = windows.shape[-1]  # static in JIT

    # --- pre‑compute Legendre nodes (constants for given L, closed) ---------
    theta = jnp.asarray(
        flt.theta(L, closed=closed), dtype=jnp.float32
    )  # :contentReference[oaicite:1]{index=1}
    x_leg = jnp.cos(theta)
    order = jnp.argsort(x_leg)
    x_sorted = jax.lax.stop_gradient(x_leg[order])  # keep as const
    x_uniform = jnp.linspace(-1.0, 1.0, L, dtype=jnp.float32)

    # --- per‑window embedder -------------------------------------------------
    def _embed_one(w):
        # 1‑D linear interpolation (JAX‑compatible)                          # :contentReference[oaicite:2]{index=2}
        y_sorted = jnp.interp(x_sorted, x_uniform, w)
        y_leg = jnp.empty_like(w).at[order].set(y_sorted)  # unsort

        # fast discrete Legendre transform (O(N log N))                     # :contentReference[oaicite:3]{index=3}
        coeffs = flt.dlt(y_leg, closed=closed)
        coeffs *= jnp.sqrt(
            (2 * jnp.arange(L) + 1) / 2
        )  # convert to orthonormal Legendre basis weights
        coeffs = coeffs[:n_coeffs]  # n_coeffs is *static* → allowed

        if normalise:
            coeffs /= jnp.linalg.norm(coeffs) + 1e-12
        return coeffs

    # vmaps over all leading batch axes in `windows`
    return jax.vmap(_embed_one, in_axes=0)(windows.reshape(-1, L)).reshape(
        windows.shape[:-1] + (n_coeffs,)
    )


def main():
    import jax.random as jr

    batch = jr.normal(jr.PRNGKey(0), (2_048, 1_024))  # 2 048 windows of length 1 024
    vecs = legendre_embed_batch(batch, n_coeffs=256)  # (2048, 256) float32

    print(vecs)

    # send to Faiss (CPU side)
    import faiss

    index = faiss.IndexFlatL2(vecs.shape[-1])
    index.add(jax.device_get(vecs))
