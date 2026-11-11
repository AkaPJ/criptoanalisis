import numpy as np


def gram_schmidt(B):
    """Ortogonalizacion de Gram-Schmidt por filas."""
    n = B.shape[0]
    B_star = np.zeros((n, n), dtype=float)
    mu = np.zeros((n, n), dtype=float)

    for i in range(n):
        B_star[i] = B[i].astype(float)
        for j in range(i):
            denom = np.dot(B_star[j], B_star[j])
            if denom > 1e-12:
                mu[i, j] = np.dot(B[i], B_star[j]) / denom
            B_star[i] -= mu[i, j] * B_star[j]

    return B_star, mu


def lll_reduction(B, delta=0.99):
    """
    Algoritmo LLL optimizado para Tier 2 con delta alta (0.99).
    """
    B = B.astype(float).copy()
    n = B.shape[0]
    B_star, mu = gram_schmidt(B)
    k = 1

    while k < n:
        # Reduccion por tamaño
        for j in range(k - 1, -1, -1):
            if abs(mu[k, j]) > 0.5:
                q = np.round(mu[k, j])
                B[k] -= q * B[j]
                B_star, mu = gram_schmidt(B)

        # Condicion de Lovasz
        norma_k = np.dot(B_star[k], B_star[k])
        norma_k_prev = np.dot(B_star[k - 1], B_star[k - 1])

        if norma_k >= (delta - mu[k, k - 1] ** 2) * norma_k_prev:
            k += 1
        else:
            B[[k, k - 1]] = B[[k - 1, k]]
            B_star, mu = gram_schmidt(B)
            k = max(k - 1, 1)

    return np.round(B).astype(int)


def babai_nearest_plane(target, B_red):
    """
    Algoritmo de Plano Mas Cercano de Babai (Nearest Plane)
    iterando secuencialmente sobre la base de Gram-Schmidt.
    """
    n = B_red.shape[0]
    b_curr = target.astype(float).copy()
    c = np.zeros(n, dtype=int)

    # Obtener Gram-Schmidt de la base reducida
    B_star, _ = gram_schmidt(B_red)

    # Iterar de la dimension n-1 hacia 0
    for i in range(n - 1, -1, -1):
        denom = np.dot(B_star[i], B_star[i])
        if denom > 1e-12:
            c_i = np.round(np.dot(b_curr, B_star[i]) / denom)
        else:
            c_i = 0
        c[i] = int(c_i)
        b_curr -= c_i * B_red[i]

    # Reconstruir el punto de la celosía
    punto_celosia = np.dot(c, B_red).astype(int)
    return punto_celosia, c


def resolver_ggh_tier2(target, base_publica, tipo_problema="CVP"):
    """
    Resolvedor de GGH Tier 2: LLL (delta=0.99) + Babai Nearest Plane.
    """
    # 1. Reducir base con delta alto para máxima ortogonalidad
    B_red = lll_reduction(base_publica, delta=0.99)

    # Caso SVP: Vector no nulo de menor norma en la base reducida
    if tipo_problema.upper() == "SVP" or np.all(target == 0):
        normas = np.linalg.norm(B_red, axis=1)
        idx_min = np.argmin([n if n > 0 else np.inf for n in normas])
        return B_red[idx_min].astype(int), B_red

    # Caso CVP: Aplicar Babai's Nearest Plane
    punto_celosia, _ = babai_nearest_plane(target, B_red)

    return punto_celosia, B_red