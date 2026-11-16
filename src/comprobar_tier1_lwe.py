import os
import glob
import time
import numpy as np

from utils.loader import cargar_instancia_lwe
from lwe.tier1_solver import resolver_lwe_tier1


def comprobar_todos_los_txt(directorio_data):
    patron = os.path.join(directorio_data, "*.*")
    archivos = [f for f in glob.glob(patron) if f.endswith('.txt') or f.endswith('.csv')]

    if not archivos:
        print(f"⚠️ No se encontraron archivos en: {directorio_data}")
        return

    print(f"🔍 Comprobando {len(archivos)} instancia(s) de LWE Tier 1...\n" + "=" * 60)

    correctos = 0

    for filepath in sorted(archivos):
        nombre = os.path.basename(filepath)

        # 1. Cargar archivo con loader.py
        instancia = cargar_instancia_lwe(filepath)
        if not instancia:
            print(f"📄 Fichero:   {nombre}\n   ❌ ERROR: No se pudo parsear el archivo.\n" + "-" * 60)
            continue

        # 2. Resolver con el algoritmo y medir tiempo
        t_inicio = time.time()
        solucion_calculada, secreto_s = resolver_lwe_tier1(instancia)
        t_ms = (time.time() - t_inicio) * 1000

        # 3. Calcular el residuo máximo (ruido) y norma euclídea del error
        q = instancia['modulo']
        n = instancia['dimension']
        errores_abs = []
        errores_vec = []
        for i in range(len(instancia['A'])):
            dot = np.dot(instancia['A'][i][:n], secreto_s)
            e = (instancia['B'][i] - dot) % q
            e_centrado = e if e <= q // 2 else e - q
            errores_vec.append(e_centrado)
            errores_abs.append(abs(e_centrado))

        ruido_max = max(errores_abs) if errores_abs else 0
        norma_e = float(np.linalg.norm(errores_vec))

        # 4. Validar contra solución oficial o mediante autoverificación de ruido
        solucion_esperada = instancia.get('solucion')
        if solucion_esperada is not None:
            es_correcto = np.array_equal(solucion_calculada, solucion_esperada)
            if es_correcto:
                correctos += 1
                estado = f"✅ CORRECTO (||e|| = {norma_e:.4f}, máx = {ruido_max})"
            else:
                estado = "❌ ERROR (Discrepancia detectada respecto a solución oficial)"
        else:
            # Si el ruido máximo derivado es pequeño (<= 5), el secreto s es válido
            if ruido_max <= 5:
                correctos += 1
                estado = f"✅ VÁLIDO N/D (||e|| = {norma_e:.4f}, máx = {ruido_max})"
            else:
                estado = f"❌ ERROR N/D (Ruido excesivo: máx = {ruido_max})"

        tipo_prob = (instancia.get('tipo') or instancia.get('problema', 'Desconocido')).strip()

        # 5. Mostrar traza en consola
        print(f"📄 Fichero:   {nombre} ({tipo_prob}, n={n}, m={len(instancia['A'])}, q={q})")
        print(f"⏱️ Tiempo:    {t_ms:.2f} ms")
        print(f"   Secreto s: {secreto_s}")
        print(f"   Calculado: {solucion_calculada}")
        print(f"   Esperado:  {solucion_esperada if solucion_esperada is not None else 'N/D'}")
        print(f"   Resultado: {estado}\n" + "-" * 60)

    print(f"\n📊 RESUMEN LWE TIER 1: {correctos}/{len(archivos)} tareas resueltas con éxito.")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    ruta_datos = os.path.join(project_root, "data", "lwe", "t1")

    comprobar_todos_los_txt(ruta_datos)