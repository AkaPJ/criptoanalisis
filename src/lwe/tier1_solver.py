import numpy as np
import itertools

def modular_inverse(a, q):
    a = a % q
    if a == 0: raise ValueError("División por cero modular.")
    def extended_gcd(a, b):
        if a == 0: return b, 0, 1
        gcd, x1, y1 = extended_gcd(b % a, a)
        return gcd, y1 - (b // a) * x1, x1
    gcd, x, _ = extended_gcd(a, q)
    return x % q

def gaussian_elimination_zq(A, b, q):
    m, n = len(A), len(A[0])
    M = [[A[i][j] % q for j in range(n)] + [b[i] % q] for i in range(m)]
    
    pivot_row = 0
    for col in range(n):
        if pivot_row >= m: break
        sel = pivot_row
        while sel < m and M[sel][col] == 0: sel += 1
        if sel == m: continue
        
        M[pivot_row], M[sel] = M[sel], M[pivot_row]
        inv = modular_inverse(M[pivot_row][col], q)
        
        for j in range(col, n + 1): M[pivot_row][j] = (M[pivot_row][j] * inv) % q
        for i in range(pivot_row + 1, m):
            factor = M[i][col]
            for j in range(col, n + 1):
                M[i][j] = (M[i][j] - factor * M[pivot_row][j]) % q
        pivot_row += 1

    s = [0] * n
    for i in range(pivot_row - 1, -1, -1):
        col = next((c for c in range(n) if M[i][c] != 0), None)
        if col is not None:
            val = M[i][n]
            for j in range(col + 1, n): val = (val - M[i][j] * s[j]) % q
            s[col] = val % q
    return s

def resolver_lwe_tier1(instancia):
    """
    Resuelve instancias LWE de Tier 1 mediante búsqueda exhaustiva acotada 
    sobre el espacio de errores ampliado [-5, 5].
    """
    n = instancia['dimension']
    q = instancia['modulo']
    
    A = instancia['A']  
    b = instancia['B']  
    m_eq = len(A)
    
    A_list = A.tolist()
    b_list = b.tolist()
    
    # Rango de error ampliado para cubrir ruidos de hasta +/-5 en Tier 1
    rango_ruido = range(-5, 6)
    
    mejor_s = None
    min_ruido_max = float('inf')
    
    for e_guess in itertools.product(rango_ruido, repeat=n):
        b_sub = [(b_list[i] - e_guess[i]) % q for i in range(n)]
        try:
            s_guess = gaussian_elimination_zq(A_list[:n], b_sub, q)
        except ValueError:
            continue
            
        residuos = []
        for i in range(m_eq):
            dot = sum(A_list[i][j] * s_guess[j] for j in range(n))
            err = (b_list[i] - dot) % q
            err_centrado = err if err <= q//2 else err - q
            residuos.append(abs(err_centrado))
            
        ruido_max = max(residuos) if residuos else 999
        
        if ruido_max < min_ruido_max:
            min_ruido_max = ruido_max
            mejor_s = np.array(s_guess)
            
        # Si el ruido máximo es <= 3, aceptamos un candidato de residuo pequeño
        if min_ruido_max <= 3:
            break
            
    if mejor_s is None:
        mejor_s = np.zeros(n, dtype=int)

    # =========================================================================
    # DETERMINAR SI ES CRIPTOANÁLISIS O ROTURA PURA
    # =========================================================================
    cripto_str = instancia.get('criptograma', '')
    es_criptograma = bool(cripto_str and "N/A" not in cripto_str.upper())
    
    if es_criptograma:
        u_str = cripto_str[cripto_str.find('{')+1 : cripto_str.find('}')]
        u = np.array([int(x.strip()) for x in u_str.split(',')])
        v = int(cripto_str.split('}')[1].replace(',', '').strip())
        
        dot_us = np.dot(u, mejor_s) % q
        m_prime = (v - dot_us) % q
        
        dist_0 = min(m_prime, q - m_prime)
        dist_mitad = abs(m_prime - (q // 2))
        bit = 1 if dist_mitad < dist_0 else 0
        
        return np.array([bit]), mejor_s
    else:
        return mejor_s, mejor_s