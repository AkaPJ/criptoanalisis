import numpy as np
try:
    # Funciona cuando se importa desde fuera (ej. comprobar_tier3_ggh.py en src/)
    from ggh.tier2_solver import babai_nearest_plane
except ImportError:
    # Funciona cuando se ejecuta tier3_solver.py directamente desde src/ggh/
    from tier2_solver import babai_nearest_plane

def gram_schmidt_rect(B):
    """
    Ortogonalización de Gram-Schmidt soportando matrices rectangulares (m filas x n columnas)
    y vectores nulos tras reducciones modulares.
    """
    m, n = B.shape
    B_star = np.zeros((m, n), dtype=float)
    mu = np.zeros((m, m), dtype=float)

    for i in range(m):
        B_star[i] = B[i].astype(float)
        for j in range(i):
            denom = np.dot(B_star[j], B_star[j])
            if denom > 1e-12:
                mu[i, j] = np.dot(B[i], B_star[j]) / denom
            B_star[i] -= mu[i, j] * B_star[j]

    return B_star, mu


def lll_reduction_modular(B, delta=0.99):
    """
    Algoritmo LLL compatible con matrices rectangulares y filas linealmente dependientes
    (necesario cuando se apila q * I_n para celosías modulares).
    """
    B = B.astype(float).copy()
    m, _ = B.shape
    B_star, mu = gram_schmidt_rect(B)
    k = 1

    while k < m:
        # 1. Reducción por tamaño
        for j in range(k - 1, -1, -1):
            if abs(mu[k, j]) > 0.5:
                q = np.round(mu[k, j])
                B[k] -= q * B[j]
                B_star, mu = gram_schmidt_rect(B)

        # 2. Condición de Lovász (moviendo vectores cero al principio para limpiar dependencias)
        norma_k = np.dot(B_star[k], B_star[k])
        norma_k_prev = np.dot(B_star[k - 1], B_star[k - 1])

        if norma_k >= (delta - mu[k, k - 1] ** 2) * norma_k_prev:
            k += 1
        else:
            B[[k, k - 1]] = B[[k - 1, k]]
            B_star, mu = gram_schmidt_rect(B)
            k = max(k - 1, 1)

    B_int = np.round(B).astype(int)
    # Filtrar y devolver solo los vectores no nulos (elimina las filas reducidas a 0)
    filas_no_nulas = [fila for fila in B_int if not np.all(fila == 0)]
    return np.array(filas_no_nulas)


def kannan_embedding_cvp(target, B_red, trace=None):
    """
    Resuelve CVP transformándolo en SVP de dimensión (n+1) mediante
    la Inmersión de Kannan (Kannan's Embedding).
    """
    n = B_red.shape[0]
    normas = np.linalg.norm(B_red, axis=1)
    M = int(np.round(np.min(normas)))
    if M == 0:
        M = 1

    # Matriz aumentada (n+1) x (n+1): [ B_red  0 ; -target  M ]
    B_kannan = np.zeros((n + 1, n + 1), dtype=float)
    B_kannan[:n, :n] = B_red
    B_kannan[n, :n] = -target
    B_kannan[n, n] = M

    # Reducir la celosía aumentada con LLL (delta=0.99)
    B_kannan_red = lll_reduction_modular(B_kannan, delta=0.99)

    if trace is not None:
        trace.update(embedding_shape=list(B_kannan.shape), M=M, short_rows=B_kannan_red.tolist(), strategy="Kannan")

    mejor_punto = None
    menor_error = np.inf

    for fila in B_kannan_red:
        ultima_coord = fila[n]
        if abs(ultima_coord) == M:
            signo = 1 if ultima_coord == M else -1
            error_vec = signo * fila[:n]
            candidato_v = np.round(target + error_vec).astype(int)

            # Verificar pertenencia a la celosía L(B_red)
            coords = np.dot(candidato_v, np.linalg.inv(B_red))
            if np.allclose(coords, np.round(coords), atol=1e-4):
                norma_e = np.linalg.norm(error_vec)
                if norma_e < menor_error:
                    menor_error = norma_e
                    mejor_punto = candidato_v

    return mejor_punto


def resolver_ggh_tier3(target, base_publica, tipo_problema="CVP", modulo=None, trace=None):
    """
    Resolvedor completo para Tier 3:
    Soporta celosías estándar y celosías modulares (q-ary lattices).
    """
    dim = base_publica.shape[1]

    # Si existe módulo q, construimos la base aumentada [ B ; q * I ]
    if modulo is not None:
        matriz_mod = np.eye(dim, dtype=int) * int(modulo)
        base_trabajo = np.vstack([base_publica, matriz_mod])
    else:
        base_trabajo = base_publica

    # Reducción LLL (elimina automáticamente las dependencias lineales de q * I)
    B_red = lll_reduction_modular(base_trabajo, delta=0.99)

    if trace is not None:
        trace.update(original_shape=list(base_publica.shape), generators_shape=list(base_trabajo.shape), reduced_shape=list(B_red.shape))

    # Caso SVP: Vector no nulo de menor norma en la base reducida
    if "SVP" in tipo_problema.upper() or np.all(target == 0):
        normas = np.linalg.norm(B_red, axis=1)
        idx_min = np.argmin(normas)
        return B_red[idx_min].astype(int), B_red

    # Caso CVP: Kannan's Embedding sobre la base ya reducida
    punto_kannan = kannan_embedding_cvp(target, B_red, trace=trace)
    if punto_kannan is not None:
        return punto_kannan, B_red

    # Fallback: Babai's Nearest Plane si Kannan no encuentra la fila exacta
    if trace is not None:
        trace["strategy"] = "Babai fallback"
    punto_babai, _ = babai_nearest_plane(target, B_red)
    return punto_babai, B_red