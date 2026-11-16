#!/usr/bin/env python3
"""Recover small-noise LWE candidates with a primal Kannan embedding + LLL.

Usage:
    pip install sympy
    python solver_lwe_lll.py Tier1T01_LWE.txt Tier1T03_LWE.txt
    python solver_lwe_lll.py *_LWE.txt

A result is a small-residual LWE *candidate*, not a proof of a unique secret.
Cryptogram decoding assumes the conventional binary encoding 0 -> 0,
1 -> round(q/2) and must be checked against the exercise's specification.
"""

import argparse
import re
from time import perf_counter
from pathlib import Path
from sympy import Matrix, Rational, eye, zeros
from sympy.matrices.normalforms import hermite_normal_form


def parse_lwe(path):
    raw = Path(path).read_bytes()
    try:
        txt = raw.decode('utf-8')
    except UnicodeDecodeError:
        txt = raw.decode('latin-1')

    n_match = re.search(r'Dimensi.n del sistema de ecuaciones\s*(\d+)', txt)
    q_match = re.search(r'reducci.n m.dulo\s*(\d+)', txt)
    if not (n_match and q_match):
        raise ValueError('No encuentro la dimension o el modulo en el archivo')
    n, q = int(n_match.group(1)), int(q_match.group(1))

    a_section = txt.split('coeficientes de las ecuaciones', 1)[1].split('independientes', 1)[0]
    A = [list(map(int, re.findall(r'-?\d+', s)))
         for s in re.findall(r'\{([^{}]+)\}', a_section)]
    rhs_section = txt.split('independientes', 1)[1].split('#', 1)[0]
    b_match = re.search(r'\{([^{}]+)\}', rhs_section)
    if b_match is None:
        raise ValueError('No encuentro los terminos independientes')
    b = list(map(int, re.findall(r'-?\d+', b_match.group(1))))
    if not A or len(A) != len(b) or any(len(row) != n for row in A):
        raise ValueError('Dimensiones inconsistentes en el archivo')

    crypto_section = txt.split('Criptograma (N/A en caso de rotura)', 1)[1]
    crypto_line = next((line.strip() for line in crypto_section.splitlines()
                        if line.strip() and not line.lstrip().startswith('#')), '')
    cryptogram = None
    if crypto_line.startswith('{'):
        vals = list(map(int, re.findall(r'-?\d+', crypto_line)))
        if len(vals) != n+1:
            raise ValueError('Criptograma de longitud incorrecta')
        cryptogram = (vals[:-1], vals[-1])
    return A, b, n, q, cryptogram


def solve_mod(A, y, q):
    """Solve A*s=y over Z/qZ via modular elimination (here q is prime)."""
    m, n = len(A), len(A[0])
    rows = [[int(z) % q for z in A[i]] + [int(y[i]) % q] for i in range(m)]
    pivots = []
    r = 0
    for j in range(n):
        pivot = next((i for i in range(r, m) if rows[i][j] % q != 0), None)
        if pivot is None:
            continue
        rows[r], rows[pivot] = rows[pivot], rows[r]
        inv = pow(rows[r][j], -1, q)
        rows[r] = [(x * inv) % q for x in rows[r]]
        for i in range(m):
            if i != r:
                coeff = rows[i][j]
                rows[i] = [(rows[i][k] - coeff * rows[r][k]) % q
                           for k in range(n+1)]
        pivots.append(j)
        r += 1
        if r == m:
            break
    if r < n or any(rows[i][n] for i in range(r, m)):
        return None
    s = [0] * n
    for i, j in enumerate(pivots):
        s[j] = rows[i][n]
    return s


def primal_lll(A, b, q, embedding_factors=(1, 2, 4, 8, 16)):
    """Find a low-error secret candidate, using Kannan embedding."""
    m = len(b)
    # The columns of [A | q*I] generate {A*s + q*z : s,z integer}.
    # HNF returns a square *column* basis; transpose it for row-wise LLL.
    basis = hermite_normal_form(Matrix(A).row_join(q * eye(m))).T
    best = None
    for factor in embedding_factors:
        embedded = basis.row_join(zeros(m, 1)).col_join(Matrix([b + [factor]]))
        reduced = embedded.lll(delta=Rational(99, 100))
        for row in reduced.tolist():
            if abs(row[-1]) != factor:
                continue
            orientation = 1 if row[-1] > 0 else -1
            error = [orientation * int(x) for x in row[:-1]]
            s = solve_mod(A, [bi-ei for bi, ei in zip(b, error)], q)
            if s is None:
                continue
            # Verify every original equation; don't trust reduction alone.
            if any((sum(aij*sj for aij, sj in zip(ai, s)) + ei - bi) % q
                   for ai, ei, bi in zip(A, error, b)):
                continue
            score = sum(ei*ei for ei in error)
            candidate = (score, s, error, factor)
            if best is None or score < best[0]:
                best = candidate
        # This exit is just an optimization for very small residuals.
        if best is not None and max(map(abs, best[2])) < q // 10:
            break
    return best


def bit_from_secret(cryptogram, secret, q):
    u, v = cryptogram
    d = (v - sum(ui*si for ui, si in zip(u, secret))) % q
    # Circular distance to 0 mod q and distance to q/2.
    dist0 = min(d, q-d)
    dist1 = abs(d - q/2)
    return d, 0 if dist0 <= dist1 else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('files', nargs='+', help='Instancias .txt')
    args = parser.parse_args()
    inicio_total = perf_counter()
    for file in args.files:
        inicio_archivo = perf_counter()
        try:
            A, b, n, q, cryptogram = parse_lwe(file)
            inicio_lll = perf_counter()
            found = primal_lll(A, b, q)
            tiempo_lll = perf_counter() - inicio_lll
            print(f'\n{Path(file).name}: n={n}, m={len(b)}, q={q}')
            print(f'  Tiempo ataque LLL: {tiempo_lll:.4f} s')
            if found is None:
                print('  LLL no encontro un candidato; podria hacer falta BKZ o ajustar la base.')
                continue
            score, secret, error, factor = found
            print(f'  Secreto candidato: {secret}')
            print(f'  Error e:           {error}')
            print(f'  |e| max:          {max(map(abs,error))}')
            print(f'  ||e||^2:          {score}')
            print(f'  Factor embedding: {factor}')
            if cryptogram is not None:
                d, bit = bit_from_secret(cryptogram, secret, q)
                print(f'  Residuo cifrado:  {d}')
                print(f'  Bit probable:     {bit} (asumiendo codificacion 0 / q/2)')
        except (ValueError, KeyError, IndexError) as exc:
            print(f'\n{Path(file).name}: error al procesar: {exc}')
        finally:
            print(f'  Tiempo del archivo: {perf_counter() - inicio_archivo:.4f} s')
    print(f'\nTiempo total de ejecucion: {perf_counter() - inicio_total:.4f} s')


if __name__ == '__main__':
    main()
