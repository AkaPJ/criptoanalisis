import os
import glob
import time
import numpy as np
from utils.loader import cargar_instancia_ggh
from ggh.tier1_solver import resolver_ggh_tier1

def comprobar_tier1():
    ruta_datos = os.path.join("data", "ggh", "tier1")
    if not os.path.exists(ruta_datos):
        ruta_datos = os.path.join("..", "data", "ggh", "tier1")

    archivos = sorted(glob.glob(os.path.join(ruta_datos, "*.csv")))
    if not archivos:
        print(f"⚠️ No se encontraron archivos en: {ruta_datos}")
        return

    print(f"🔍 Evaluando {len(archivos)} instancias de GGH Tier 1 con LLL...\n" + "="*60)

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
        es_correcto = np.array_equal(punto_calculado, solucion_esperada)

        print(f"📄 Fichero:   {nombre} ({instancia.get('problema')})")
        print(f"⏱️ Tiempo:    {t_total:.2f} ms")
        print(f"   Calculado: {punto_calculado}")
        print(f"   Esperado:  {solucion_esperada}")
        print(f"   Estado:    {'✅ OK' if es_correcto else '❌ FALLO'}\n" + "-"*60)

if __name__ == "__main__":
    comprobar_tier1()