
import os
import glob
import time
import numpy as np

from utils.loader import cargar_instancia_ggh
from ggh.tier3_solver import resolver_ggh_tier3


from utils.validacion_exacta import (verificar_pertenencia_celosia,
                                      verificar_base_reducida)


def comprobar_tier3():

    # 1. Localizar las instancias
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    ruta_datos = os.path.join(
        project_root, "data", "ggh", "t3"
    )

    patron = os.path.join(ruta_datos, "*.*")

    archivos = sorted([
        f for f in glob.glob(patron)
        if f.endswith((".csv", ".txt"))
    ])

    if not archivos:
        print(f"⚠️ No se encontraron archivos en: {ruta_datos}")
        return

    print(
        f"🔥 Evaluando {len(archivos)} instancias "
        f"de GGH Tier 3 (Kannan + Celosías Modulares)..."
    )
    print("=" * 75)

    verificados = 0
    optimos_demostrados = 0

    for filepath in archivos:

        nombre = os.path.basename(filepath)
        instancia = cargar_instancia_ggh(filepath)

        base_pub = instancia["base"]
        dim = base_pub.shape[1]

        target = np.asarray(instancia["target"])
        problema = instancia.get("problema", "CVP").upper()
        modulo = instancia.get("modulo")

        # 2. Ajustar dimensión del objetivo
        if len(target) != dim:

            if "SVP" in problema or np.all(target == 0):
                target = np.zeros(dim)

            else:
                target = np.pad(
                    target,
                    (0, max(0, dim - len(target)))
                )[:dim]

        # 3. Ejecutar el algoritmo
        t_inicio = time.perf_counter()

        punto_calculado, B_red = resolver_ggh_tier3(
            target=target,
            base_publica=base_pub,
            tipo_problema=problema,
            modulo=modulo
        )

        t_total = (
            time.perf_counter() - t_inicio
        ) * 1000

        punto_calculado = np.asarray(punto_calculado)

        assert verificar_base_reducida(base_pub, B_red, instancia.get("modulo")), "LLL cambió el retículo"

        solucion_esperada = instancia["solucion"]

        # 4. Calcular métricas
        norma_ruido = np.linalg.norm(
            target - punto_calculado
        )

        norma_v = np.linalg.norm(punto_calculado)

        # 5. Verificar pertenencia al retículo
        es_en_celosia, coords = verificar_pertenencia_celosia(
            punto_calculado,
            B_red
        )

        es_svp = "SVP" in problema
        es_no_nulo = norma_v > 0

        # Comprobación de integridad del vector
        es_vector_entero = all(x == int(x) for x in punto_calculado)

        # 6. Validar resultado
        es_valido = es_en_celosia

        if es_svp:
            es_valido = es_valido and es_no_nulo

        # Comparación con solución oficial, si existe
        coincide_referencia = None

        if solucion_esperada is not None:
            coincide_referencia = np.array_equal(
                punto_calculado,
                solucion_esperada
            )

        # 7. Comprobar optimalidad demostrable
        optimo_demostrado = False

        if es_valido and es_vector_entero:

            # En Z^n, un vector no nulo de norma 1
            # tiene la mínima norma euclídea posible.
            if es_svp and sum(int(x)**2 for x in punto_calculado) == 1:
                optimo_demostrado = True

            # En CVP, distancia cero implica optimalidad.
            elif not es_svp and np.array_equal(target, punto_calculado):
                optimo_demostrado = True

        # 8. Construir estado
        if not es_valido:

            if not es_en_celosia:
                estado = (
                    "❌ ERROR "
                    "(El vector no pertenece a la celosía)"
                )

            else:
                estado = (
                    "❌ ERROR "
                    "(Vector nulo devuelto en SVP)"
                )

        else:

            verificados += 1

            if optimo_demostrado:
                optimos_demostrados += 1
                estado = "🏆 ÓPTIMO DEMOSTRADO"

            elif coincide_referencia is True:
                estado = "✅ COINCIDE CON REFERENCIA"

            else:
                estado = "✅ PERTENENCIA VERIFICADA"

        # 9. Mostrar resultados
        info_mod = (
            f"Mod: {modulo}"
            if modulo is not None
            else "Mod: N/A"
        )

        print(
            f"📄 Fichero:   {nombre} "
            f"({problema}, Dim: {dim}, {info_mod})"
        )

        print(f"⏱️ Tiempo:    {t_total:.2f} ms")
        print(f"   Calculado: {punto_calculado}")

        print(
            f"   Esperado:  "
            f"{solucion_esperada if solucion_esperada is not None else 'N/D'}"
        )

        print(f"   Pertenencia a B_red: {es_en_celosia}")

        if es_svp:
            print(f"   Norma ||v||: {norma_v:.4f}")
        else:
            print(f"   Distancia ||c-v||: {norma_ruido:.4f}")

        if coincide_referencia is not None:
            print(
                f"   Coincide con referencia: "
                f"{coincide_referencia}"
            )

        print(f"   Resultado: {estado}")
        print("-" * 75)

    # 10. Resumen
    print("\n📊 RESUMEN TIER 3")
    print(
        f"   Pertenencia y condiciones verificadas: "
        f"{verificados}/{len(archivos)}"
    )
    print(
        f"   Optimalidad demostrada por cota elemental: "
        f"{optimos_demostrados}/{len(archivos)}"
    )


if __name__ == "__main__":
    comprobar_tier3()
