# Revisión técnica de la memoria

Revisión realizada el 16 de noviembre de 2026 sobre el proyecto existente.
El PDF final tiene cinco páginas. `compilation.log` corresponde a dos pasadas
independientes de pdfLaTeX en un directorio temporal; no contiene errores,
citas sin resolver, `Overfull` ni `Underfull`. Se inspeccionaron visualmente
las cinco páginas renderizadas con Poppler.

## Archivos modificados

- `docs/main.tex`, `docs/main.pdf`: metodología, tabla reconstruida, tres trazas
  intermedias, conclusiones y declaración de IA separada de las siete referencias.
- `src/ggh/tier2_solver.py`, `src/ggh/tier3_solver.py`,
  `src/lwe/solver_lwe_lll.py`: parámetro opcional `trace`, sin cambios en la
  estrategia, parámetros, criterios de selección ni valores de retorno normales.
- `src/comprobar_tier1_ggh.py`, `src/comprobar_tier2_ggh.py`,
  `src/comprobar_tier3_ggh.py`: pertenencia racional exacta y comparación HNF
  de los generadores originales frente a la base reducida. En Tier 3 las
  condiciones elementales de optimalidad usan igualdad, sin tolerancias.
- `src/lwe/tier1_solver.py`, `src/lwe/tier2_solver.py`: únicamente comentarios
  corregidos (candidato de residuo pequeño no implica clave exacta; el solver
  antiguo de Tier 2 no usa Kannan).
- Nuevos: `src/utils/validacion_exacta.py`, `src/auditar_memoria.py` y estos
  registros reproducibles en `docs/audit/`.

No se han cambiado los algoritmos de resolución. Las inconsistencias
comprobadas eran afirmaciones de la memoria, comentarios y validaciones
insuficientes, además de la mezcla de rutas LWE distintas.

## Ruta LWE confirmada

Las dos filas LWE y la traza LWE actuales se obtienen exclusivamente de
`src/lwe/solver_lwe_lll_timer.py`. Este archivo NO importa
`solver_lwe_lll.py`: define sus propias `parse_lwe`, `primal_lll`,
`solve_mod` y `bit_from_secret`. `main` carga cada instancia, mide su
`primal_lll` y después decodifica el criptograma cuando existe.
Solo importa biblioteca estándar (`argparse`, `re`, `time`, `pathlib`)
y SymPy (`Matrix`, `Rational`, `eye`, `zeros`, `hermite_normal_form`).
La auditoría usa NumPy para GGH; el programa LWE temporizado no lo usa.

La auditoría anterior llamaba al archivo sin temporizador. Se corrigió su
importación y se repitieron tres mediciones por instancia del `primal_lll`
definido en el temporizador, con el mismo intervalo que `main` mide.
Se ejecutó además la CLI real de ese programa sobre las diez instancias
(`lwe_primal_timer.log`). Las soluciones, errores, normas, factores y bits
coinciden con los anteriores; solo se actualizan los tiempos observados.
Cada resultado LWE de `results.json` identifica su función de origen.

La traza se recapturó mediante `sys.setprofile`, observando al retornar de
la función del temporizador sus variables `basis`, `embedded` y `reduced`.
No se modifica ningún solver ni se mide con el perfilador activo.
El programa temporizado conserva íntegramente su contenido original.
`lwe_timer_route.log` registra las nuevas mediciones; `timer_route_check.log`
verifica la ruta, las salidas CLI y la correspondencia de la fila intermedia.
Los registros antiguos de otras rutas se conservan como históricos y no
intervienen en la tabla, las trazas ni la explicación actual de LWE.

## Resultados y límites

- GGH: 7/7 Tier 1, 5/5 Tier 2, 5/5 Tier 3 con pertenencia exacta. Las HNF
  por filas de la base reducida y los generadores originales coinciden en
  los 17 casos. Los generadores modulares incluyen `q I` (no se compara
  únicamente con la base pública sin módulo).
- Solo se demuestra optimalidad para `Tier03T01_Latt.txt` y
  `Tier03T02_Latt.txt` (CVP, distancia exactamente cero) y
  `Tier03T05_Latt.txt` (SVP, vector entero no nulo de norma al cuadrado uno).
  En las otras 14 instancias no se ha demostrado optimalidad. Tampoco se
  afirma que no sean óptimas.
- Motor primal LWE: 5/5 Tier 1 y 5/5 Tier 2 satisfacen todas las congruencias
  originales exactamente. Los errores extraídos coinciden con los residuos
  centrados y cumplen los umbrales experimentales 5 y 6, respectivamente.
  Máximos observados: 4 y 6. Esos umbrales no se presentan como parámetros
  de una distribución de error especificada por los ficheros.
- No hay prueba de unicidad del secreto ni referencia oficial para autenticar
  los bits candidatos. La decodificación usa la convención binaria 0 / q/2.
- El motor primal forma `HNF([A|qI]).T`, no `HNF([A|qI].T)`.
  Usa LLL exacto de SymPy con delta 99/100, factores 1,2,4,8,16 y salida
  anticipada por residuo; todas las instancias salen con M=1.
- La media de LWE Tier 2 es 133.11 ms; la quinta instancia
  `(n,m,q)=(10,30,541)` requiere 392.89 ms de media. Los tiempos son
  descriptivos de esta ejecución y entorno, no cotas de complejidad.

`results.json` conserva tiempos sin redondear, entorno, resultados individuales
(y cuadrados exactos de las métricas), estados y datos intermedios completos.
`run.log` contiene las salidas individuales del experimento. Las matrices
reducidas se conservan en JSON para auditar las trazas, sin imprimirlas en
la memoria. `instrumentation.log` registra la comparación con el código
original de HEAD: 20/20 resultados iguales. El propio script de auditoría
comprueba además todas las salidas GGH T2/T3 con y sin `trace`, y todas las
salidas LWE del temporizador con y sin perfilado externo.

## Reproducción

Desde la raíz del proyecto:

```sh
python3 -m venv .venv
.venv/bin/pip install -r docs/audit/requirements.txt
.venv/bin/python src/auditar_memoria.py > docs/audit/run.log
.venv/bin/python src/comprobar_tier1_ggh.py
.venv/bin/python src/comprobar_tier2_ggh.py
.venv/bin/python src/comprobar_tier3_ggh.py
.venv/bin/python src/lwe/solver_lwe_lll_timer.py data/lwe/t1/*.txt data/lwe/t2/*.txt
```

La auditoría hace tres llamadas no instrumentadas por instancia; excluye
lectura, validación e impresión. No ejecutar simultáneamente otros solvers
para reproducir los tiempos de la tabla. El script regenera el JSON;
los valores de la memoria corresponden al JSON entregado, no se actualizan
automáticamente al repetir el experimento.

Para compilar sin colisiones con compilaciones automáticas del editor:

```sh
mkdir -p /tmp/cripto-build
pdflatex -interaction=nonstopmode -halt-on-error -output-directory /tmp/cripto-build docs/main.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory /tmp/cripto-build docs/main.tex
cp /tmp/cripto-build/main.pdf docs/main.pdf
```

## Referencias conservadas

Se mantienen las siete referencias y sus citas en los apartados que justifican
GGH, LWE, LLL, ataques primal, Kannan, Babai y ML-KEM. No se conserva el umbral
universal de BKZ beta > 400: no estaba justificado por las referencias dadas.
Se contrastaron las páginas primarias de [FIPS 203](https://csrc.nist.gov/pubs/fips/203/final)
y [Albrecht, Player y Scott](https://eprint.iacr.org/2015/046).
FIPS 203 relaciona ML-KEM con Module-LWE; el artículo se conserva como referencia
metodológica, sin atribuirle estimaciones actuales universales sobre ML-KEM.
