"""Auditoría entera independiente de las reducciones NumPy en coma flotante."""
from sympy import Matrix, eye
from sympy.matrices.normalforms import hermite_normal_form


def matriz_entera(datos):
    rows = datos.tolist() if hasattr(datos, 'tolist') else datos
    if isinstance(rows, Matrix):
        rows = rows.tolist()
    if any(x != int(x) for row in rows for x in row):
        raise ValueError('La auditoría exige datos enteros')
    return Matrix([[int(x) for x in row] for row in rows])


def base_hnf(generadores):
    return hermite_normal_form(matriz_entera(generadores).T).T


def verificar_pertenencia_celosia(punto, base):
    H = base_hnf(base)
    v = matriz_entera([list(punto)])
    coords = v * H.inv()
    return all(x.q == 1 for x in coords), coords


def verificar_base_reducida(original, reducida, modulo=None):
    G = matriz_entera(original)
    if modulo is not None:
        G = G.col_join(int(modulo) * eye(G.cols))
    return base_hnf(G) == base_hnf(reducida)
