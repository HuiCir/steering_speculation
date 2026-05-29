"""
SPREAD: SPherical intervention for REAsoning Diversity
======================================================
Riemannian Block Coordinate Descent on product of spheres.
Maximizes log-det(I + (H+V)(H+V)^T) — the volume spanned
by intervened activations across N branches.

Reference: AAAI 2026, "Test-time Diverse Reasoning by
           Riemannian Activation Steering"
"""
import numpy as np


def riemannian_block_update(h_last, T=10, calpha_k=0.3):
    """
    Compute steering vectors V that maximize the volume spanned
    by the intervened activations H+V.

    Args:
        h_last: (N, D) numpy array — last-token hidden states
        T: Riemannian BCD iterations
        calpha_k: steering strength coefficient

    Returns:
        V: (N, D) numpy array — per-branch steering vectors
    """
    N, D = h_last.shape
    alpha_k = [calpha_k * (np.linalg.norm(h_last[k]) / (D + 1e-8)) for k in range(N)]
    H_norm_fro = np.linalg.norm(h_last, ord='fro')
    H_bar = np.mean(h_last, axis=0)

    # Initialize: random perturbation away from centroid
    V = np.stack([
        h_last[k] - H_bar + np.random.uniform(0, 1, size=D)
        for k in range(N)
    ])
    for k in range(N):
        V[k] = V[k] / (np.linalg.norm(V[k]) + 1e-8) * np.sqrt(alpha_k[k])

    # Riemannian Block Coordinate Descent
    for _ in range(T):
        for k in range(N):
            HV = h_last + V
            M_inv = np.linalg.inv(np.eye(N) + HV @ HV.T)
            g_k = -2 * HV.T @ M_inv[:, k]
            vk_prev = V[k]

            # Project gradient to tangent space of sphere
            proj = g_k - (1.0 / alpha_k[k]) * np.dot(vk_prev, g_k) * vk_prev

            # Lipschitz step size
            L = (2 + 4 * (H_norm_fro + alpha_k[k]) ** 2 +
                 (2.0 / np.sqrt(alpha_k[k] + 1e-8)) * (H_norm_fro + alpha_k[k]))
            d_k = (1.0 / L) * proj
            dn = np.linalg.norm(d_k)

            if dn > 0:
                # Exponential map on sphere (retraction)
                V[k] = (
                    np.cos(dn / np.sqrt(alpha_k[k])) * vk_prev -
                    np.sin(dn / np.sqrt(alpha_k[k])) * d_k / dn * np.sqrt(alpha_k[k])
                )

    return V
