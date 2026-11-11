import os
import glob
import time
import numpy as np
from utils.loader import cargar_instancia_ggh
from ggh.tier1_solver import resolver_ggh_tier1


def comprobar_tier1():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    ruta_datos = os.path.join(project_root, "data", "ggh", "t1")

    patron = os.path.join(ruta_datos, "*.*")
    archivos = sorted([f for f in glob.glob(patron) if f.endswith('.csv') or f.endswith('.txt')])

    if not archivos:
        print(f"⚠️ No se encontraron archivos en: {ruta_datos}")
        return

    print(f"🔍 Evaluando {len(archivos)} instancias de GGH Tier 1 con LLL...\n" + "=" * 60)

    correctos = 0

    for filepath in archivos:
        nombre = os.path.basename(filepath)
        instancia = cargar_instancia_ggh(filepath)

        t_inicio = time.time()
        punto_calculado, B_red = resolver_ggh_tier1(
            target=instancia['target'],
            base_publica=instancia['base'],
            tipo_problema=instancia.get('problema', 'CVP')
        )
        t_total = (time.time() - t_inicio) * 1000  # milisegundos

        solucion_esperada = instancia['solucion']

        # Manejo de instancias sin solución previa (N/D)
        if solucion_esperada is None:
            estado = "❓ SIN SOLUCIÓN EN CSV (N/D)"
            es_correcto = False
        else:
            es_correcto = np.array_equal(punto_calculado, solucion_esperada)
            if es_correcto:
                correctos += 1
                estado = "✅ CORRECTO"
            else:
                estado = "❌ ERROR"

        print(f"📄 Fichero:   {nombre} ({instancia.get('problema')})")
        print(f"⏱️ Tiempo:    {t_total:.2f} ms")
        print(f"   Calculado: {punto_calculado}")
        print(f"   Esperado:  {solucion_esperada if solucion_esperada is not None else 'N/D'}")
        print(f"   Resultado: {estado}\n" + "-" * 60)

    print(f"\n📊 RESUMEN: {correctos}/{len(archivos)} tareas verificadas contra solución oficial.")


if __name__ == "__main__":
    comprobar_tier1()