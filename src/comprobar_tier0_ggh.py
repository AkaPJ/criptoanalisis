import os
import glob
import numpy as np
from utils.loader import cargar_instancia_ggh
from ggh.tier0_solver import resolver_ggh_tier0


def comprobar_todos_los_csv(directorio_data):
    # Buscar ficheros .csv o .txt en la carpeta
    patron = os.path.join(directorio_data, "*.*")
    archivos = [f for f in glob.glob(patron) if f.endswith('.csv') or f.endswith('.txt')]

    if not archivos:
        print(f"⚠️ No se encontraron archivos en: {directorio_data}")
        return

    print(f"🔍 Comprobando {len(archivos)} instancia(s) de GGH Tier 0...\n" + "=" * 50)

    correctos = 0

    for filepath in sorted(archivos):
        nombre = os.path.basename(filepath)

        # 1. Cargar archivo con loader.py
        instancia = cargar_instancia_ggh(filepath)

        # 2. Resolver con el algoritmo
        punto_calculado, coords = resolver_ggh_tier0(
            target=instancia['target'],
            base=instancia['base'],
            tipo_problema=instancia.get('problema', 'CVP')
        )

        # 3. Validar contra la solución oficial del CSV
        solucion_esperada = instancia['solucion']
        es_correcto = np.array_equal(punto_calculado, solucion_esperada)

        if es_correcto:
            correctos += 1
            estado = "✅ CORRECTO"
        else:
            estado = "❌ ERROR"

        # 4. Mostrar traza en consola
        print(f"📄 Fichero:   {nombre}")
        print(f"   Problema:  {instancia.get('problema')}")
        print(f"   Calculado: {punto_calculado}")
        print(f"   Esperado:  {solucion_esperada}")
        print(f"   Resultado: {estado}\n" + "-" * 50)

    print(f"\n📊 RESUMEN: {correctos}/{len(archivos)} tareas resueltas con éxito.")


if __name__ == "__main__":
    # Ajustar la ruta relativa a la carpeta de datos
    ruta_datos = os.path.join("data", "ggh", "t0")
    if not os.path.exists(ruta_datos):
        ruta_datos = os.path.join("..", "data", "ggh", "t0")

    comprobar_todos_los_csv(ruta_datos)