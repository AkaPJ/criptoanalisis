import os
import glob
import time
import numpy as np
from utils.loader import cargar_instancia_ggh
from ggh.tier2_solver import resolver_ggh_tier2


def verificar_pertenencia_celosia(punto, base_publica):
    """
    Verifica si un punto v pertenece a la celosía L(B').
    Un punto v in L si v * (B')^-1 produce coeficientes enteros exactos.
    """
    try:
        b_inv = np.linalg.inv(base_publica)
        coords = np.dot(punto, b_inv)
        es_entero = np.allclose(coords, np.round(coords), atol=1e-4)
        return es_entero, coords
    except np.linalg.LinAlgError:
        return False, None


def comprobar_tier2():
    # 1. Ruta absoluta apuntando a data/ggh/t2
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    ruta_datos = os.path.join(project_root, "data", "ggh", "t2")

    # 2. Buscar archivos .csv y .txt
    patron = os.path.join(ruta_datos, "*.*")
    archivos = sorted([f for f in glob.glob(patron) if f.endswith('.csv') or f.endswith('.txt')])

    if not archivos:
        print(f"⚠️ No se encontraron archivos en: {ruta_datos}")
        return

    print(f"🚀 Evaluando {len(archivos)} instancias de GGH Tier 2 (LLL delta=0.99 + Nearest Plane)...\n" + "=" * 70)

    verificados = 0

    for filepath in archivos:
        nombre = os.path.basename(filepath)
        instancia = cargar_instancia_ggh(filepath)

        base_pub = instancia['base']
        dim = base_pub.shape[1]
        target = instancia['target']
        problema = instancia.get('problema', 'CVP').upper()

        # Ajustar la dimensión del target si es SVP o viene recortado en el CSV (ej: "0 0")
        if len(target) != dim:
            if "SVP" in problema or np.all(target == 0):
                target = np.zeros(dim)
            else:
                target = np.pad(target, (0, max(0, dim - len(target))))[:dim]

        t_inicio = time.time()
        punto_calculado, B_red = resolver_ggh_tier2(
            target=target,
            base_publica=base_pub,
            tipo_problema=problema
        )
        t_total = (time.time() - t_inicio) * 1000  # milisegundos

        solucion_esperada = instancia['solucion']

        # Validación geométrica
        es_en_celosia, _ = verificar_pertenencia_celosia(punto_calculado, base_pub)
        norma_error = np.linalg.norm(target - punto_calculado)

        if solucion_esperada is not None:
            # Caso 1: Hay solución explícita en el CSV
            es_correcto = np.array_equal(punto_calculado, solucion_esperada)
            if es_correcto:
                verificados += 1
                estado = "✅ CORRECTO (Igual a la solución del CSV)"
            else:
                estado = "❌ ERROR (No coincide con la solución del CSV)"
        else:
            # Caso 2: Solución N/D -> Autoverificación geométrica
            if es_en_celosia:
                verificados += 1
                if "SVP" in problema:
                    norma_v = np.linalg.norm(punto_calculado)
                    estado = f"✅ VÁLIDO (Vector en celosía | ||v|| = {norma_v:.4f})"
                else:
                    estado = f"✅ VÁLIDO (Punto en celosía | ||c - v|| = {norma_error:.4f})"
            else:
                estado = "❌ ERROR (El punto devuelto no pertenece a la celosía)"

        print(f"📄 Fichero:   {nombre} ({problema}, Dim: {dim})")
        print(f"⏱️ Tiempo:    {t_total:.2f} ms")
        print(f"   Calculado: {punto_calculado}")
        print(f"   Esperado:  {solucion_esperada if solucion_esperada is not None else 'N/D'}")
        print(f"   Resultado: {estado}\n" + "-" * 70)

    print(f"\n📊 RESUMEN TIER 2: {verificados}/{len(archivos)} instancias resueltas y validadas con éxito.")


if __name__ == "__main__":
    comprobar_tier2()