import numpy as np


def resolver_ggh_tier0(target, base, tipo_problema="CVP"):
    """
    Resuelve CVP o SVP para celosías con bases ortogonales (Tier 0).
    """
    # Caso SVP: Buscar el vector no nulo más corto de la base ortogonal
    if tipo_problema.upper() == "SVP" or (np.all(target == 0) and tipo_problema.upper() != "CVP"):
        normas = np.linalg.norm(base, axis=1)
        # Seleccionamos la fila de la base con menor norma euclídea
        idx_min = np.argmin([n if n > 0 else np.inf for n in normas])
        punto_celosia = base[idx_min]

        y_coordenadas = np.zeros(len(base), dtype=int)
        y_coordenadas[idx_min] = 1
        return punto_celosia.astype(int), y_coordenadas

    # Caso CVP: Redondeo de Babai
    base_inv = np.linalg.inv(base)
    coordenadas_reales = np.dot(target, base_inv)
    y_coordenadas = np.round(coordenadas_reales).astype(int)
    punto_celosia = np.dot(y_coordenadas, base).astype(int)

    return punto_celosia, y_coordenadas