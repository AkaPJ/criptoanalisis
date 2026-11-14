import os
import glob
import numpy as np

# IMPORTANTE: Ajusta estas rutas a la estructura real de tu proyecto
from utils.loader import cargar_instancia_lwe
from lwe.tier1_solver import resolver_lwe_tier1

def comprobar_todos_los_txt(directorio_data):
    patron = os.path.join(directorio_data, "*.*")
    archivos = [f for f in glob.glob(patron) if f.endswith('.txt') or f.endswith('.csv')]

    if not archivos:
        print(f"⚠️ No se encontraron archivos en: {directorio_data}")
        return

    print(f"🔍 Comprobando {len(archivos)} instancia(s) de LWE Tier 1...\n" + "=" * 50)

    correctos = 0

    for filepath in sorted(archivos):
        nombre = os.path.basename(filepath)

        # 1. Cargar archivo con loader.py
        instancia = cargar_instancia_lwe(filepath)
        if not instancia:
            print(f"📄 Fichero:   {nombre}\n   ❌ ERROR: No se pudo parsear el archivo.\n" + "-" * 50)
            continue

        # 2. Resolver con el algoritmo
        solucion_calculada, secreto_s = resolver_lwe_tier1(instancia)

        # 3. Validar contra la solución oficial
        solucion_esperada = instancia.get('solucion')
        
        # Calcular el residuo máximo (ruido) del sistema para trazas
        q = instancia['modulo']
        n = instancia['dimension']
        errores = []
        for i in range(len(instancia['A'])):
            dot = np.dot(instancia['A'][i][:n], secreto_s)
            e = (instancia['B'][i] - dot) % q
            e_centrado = e if e <= q//2 else e - q
            errores.append(abs(e_centrado))
        ruido_max = max(errores) if errores else 0

        # Validación con NumPy
        if solucion_esperada is not None:
            es_correcto = np.array_equal(solucion_calculada, solucion_esperada)
            if es_correcto:
                correctos += 1
                estado = f"✅ CORRECTO (Ruido máx derivado = {ruido_max})"
            else:
                estado = f"❌ ERROR (Discrepancia detectada)"
        else:
            estado = f"❓ N/D (Ruido máx derivado = {ruido_max})"

        # 4. Mostrar traza en consola
        print(f"📄 Fichero:   {nombre}")
        print(f"   Problema:  {instancia.get('tipo', 'Desconocido').strip()}")
        print(f"   Calculado: {solucion_calculada}")
        print(f"   Esperado:  {solucion_esperada}")
        print(f"   Resultado: {estado}\n" + "-" * 50)

    print(f"\n📊 RESUMEN: {correctos}/{len(archivos)} tareas resueltas con éxito.")

if __name__ == "__main__":
    # Ajustar la ruta relativa a la carpeta de datos
    ruta_datos = os.path.join("data", "lwe", "t2")
    if not os.path.exists(ruta_datos):
        ruta_datos = os.path.join("..", "data", "lwe", "t2")

    comprobar_todos_los_txt(ruta_datos)