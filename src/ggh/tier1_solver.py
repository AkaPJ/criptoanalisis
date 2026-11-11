import numpy as np


def gram_schmidt(B):
    """
    Calcula la ortogonalización de Gram-Schmidt (B_star)
    y la matriz de coeficientes mu.
    """
    n = B.shape[0]
    B_star = np.zeros((n, n), dtype=float)
    mu = np.zeros((n, n), dtype=float)

    for i in range(n):
        B_star[i] = B[i].astype(float)
        for j in range(i):
            denom = np.dot(B_star[j], B_star[j])
            if denom != 0:
                mu[i, j] = np.dot(B[i], B_star[j]) / denom
            B_star[i] -= mu[i, j] * B_star[j]

    return B_star, mu


def lll_reduction(B, delta=0.75):
    """
    Algoritmo LLL (Lenstra-Lenstra-Lovász).
    Transforma la base B en una base reducida B_red con vectores cortos y casi ortogonales.
    """
    B = B.astype(float).copy()
    n = B.shape[0]
    B_star, mu = gram_schmidt(B)
    k = 1

    while k < n:
        # 1. Reducción por tamaño (Size Reduction)
        for j in range(k - 1, -1, -1):
            if abs(mu[k, j]) > 0.5:
                q = np.round(mu[k, j])
                B[k] -= q * B[j]
                B_star, mu = gram_schmidt(B)

        # 2. Condición de Lovász
        norma_k = np.dot(B_star[k], B_star[k])
        norma_k_prev = np.dot(B_star[k - 1], B_star[k - 1])

        if norma_k >= (delta - mu[k, k - 1] ** 2) * norma_k_prev:
            k += 1
        else:
            # Intercambiar vectores B[k] y B[k-1]
            B[[k, k - 1]] = B[[k - 1, k]]
            B_star, mu = gram_schmidt(B)
            k = max(k - 1, 1)

    return np.round(B).astype(int)


def resolver_ggh_tier1(target, base_publica, tipo_problema="CVP"):
    """
    Resuelve CVP/SVP en Tier 1 usando LLL + Babai.
    """
    # 1. Reducir la base pública con LLL
    B_red = lll_reduction(base_publica)

    # Caso SVP: Devolver el vector no nulo más corto de la base reducida
    if tipo_problema.upper() == "SVP" or np.all(target == 0):
        normas = np.linalg.norm(B_red, axis=1)
        idx_min = np.argmin([n if n > 0 else np.inf for n in normas])
        punto_celosia = B_red[idx_min]

        y_coords = np.zeros(len(B_red), dtype=int)
        y_coords[idx_min] = 1
        return punto_celosia.astype(int), B_red

    # Caso CVP: Aplicar redondeo de Babai usando la base reducida B_red
    B_red_inv = np.linalg.inv(B_red)
    coordenadas_reales = np.dot(target, B_red_inv)
    y_red = np.round(coordenadas_reales).astype(int)

    # Reconstruir el punto de la celosía en coordenadas reales
    punto_celosia = np.dot(y_red, B_red).astype(int)

    return punto_celosia, B_red