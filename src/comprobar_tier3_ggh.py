import os
import glob
import time
import numpy as np
from utils.loader import cargar_instancia_ggh
from ggh.tier3_solver import resolver_ggh_tier3


def verificar_pertenencia_celosia(punto, base_valida):
    """
    Verifica si un punto v pertenece a la celosía L(B).
    Nota: En celosías modulares, 'base_valida' debe ser B_red (que ya incorpora
    el módulo q) en lugar de la base pública sin reducir.
    """
    try:
        b_inv = np.linalg.inv(base_valida)
        coords = np.dot(punto, b_inv)
        es_entero = np.allclose(coords, np.round(coords), atol=1e-4)
        return es_entero, coords
    except np.linalg.LinAlgError:
        return False, None


def comprobar_tier3():
    # 1. Ruta absoluta apuntando a data/ggh/t3
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    ruta_datos = os.path.join(project_root, "data", "ggh", "t3")

    # 2. Buscar tanto archivos .csv como .txt
    patron = os.path.join(ruta_datos, "*.*")
    archivos = sorted([f for f in glob.glob(patron) if f.endswith('.csv') or f.endswith('.txt')])

    if not archivos:
        print(f"⚠️ No se encontraron archivos en: {ruta_datos}")
        return

    print(f"🔥 Evaluando {len(archivos)} instancias de GGH Tier 3 (Kannan + Celosías Modulares)...\n" + "=" * 75)

    verificados = 0

    for filepath in archivos:
        nombre = os.path.basename(filepath)
        instancia = cargar_instancia_ggh(filepath)

        base_pub = instancia['base']
        dim = base_pub.shape[1]
        target = instancia['target']
        problema = instancia.get('problema', 'CVP').upper()
        modulo = instancia.get('modulo')  # None si es N/A, o el entero q (ej. 541)

        # Ajustar dimensión del target si es SVP o viene recortado (ej: "0 0")
        if len(target) != dim:
            if "SVP" in problema or np.all(target == 0):
                target = np.zeros(dim)
            else:
                target = np.pad(target, (0, max(0, dim - len(target))))[:dim]

        t_inicio = time.time()
        punto_calculado, B_red = resolver_ggh_tier3(
            target=target,
            base_publica=base_pub,
            tipo_problema=problema,
            modulo=modulo
        )
        t_total = (time.time() - t_inicio) * 1000  # milisegundos

        solucion_esperada = instancia['solucion']
        norma_ruido = np.linalg.norm(target - punto_calculado)

        # Validación geométrica contra B_red (soporta tanto bases estándar como modulares)
        es_en_celosia, _ = verificar_pertenencia_celosia(punto_calculado, B_red)

        if solucion_esperada is not None:
            es_correcto = np.array_equal(punto_calculado, solucion_esperada)
            if es_correcto:
                verificados += 1
                estado = f"✅ CORRECTO (||c - v|| = {norma_ruido:.4f})"
            else:
                norma_diff = np.linalg.norm(punto_calculado - solucion_esperada)
                estado = f"❌ ERROR (distancia a solución = {norma_diff:.4f})"
        else:
            # Autoverificación geométrica cuando la solución es N/D
            if es_en_celosia:
                verificados += 1
                if "SVP" in problema:
                    norma_v = np.linalg.norm(punto_calculado)
                    # En SVP verificamos también que no haya devuelto el vector nulo [0,0,...,0]
                    if norma_v > 0:
                        estado = f"✅ VÁLIDO (Vector en celosía | ||v|| = {norma_v:.4f})"
                    else:
                        verificados -= 1
                        estado = "❌ ERROR (Vector nulo devuelto en SVP)"
                else:
                    estado = f"✅ VÁLIDO (Punto en celosía | ||c - v|| = {norma_ruido:.4f})"
            else:
                estado = "❌ ERROR (El punto devuelto no pertenece a la celosía)"

        info_mod = f"Mod: {modulo}" if modulo is not None else "Mod: N/A"
        print(f"📄 Fichero:   {nombre} ({problema}, Dim: {dim}, {info_mod})")
        print(f"⏱️ Tiempo:    {t_total:.2f} ms")
        print(f"   Calculado: {punto_calculado}")
        print(f"   Esperado:  {solucion_esperada if solucion_esperada is not None else 'N/D'}")
        print(f"   Resultado: {estado}\n" + "-" * 75)

    print(f"\n📊 RESUMEN TIER 3: {verificados}/{len(archivos)} instancias resueltas y validadas con éxito.")


if __name__ == "__main__":
    comprobar_tier3()