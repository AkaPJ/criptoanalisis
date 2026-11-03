import numpy as np


def cargar_instancia_ggh(filepath):
    """
    Parsea ficheros de celosías (GGH / CVP / SVP).
    Ignora comentarios '#' y extrae la dimensión, base, punto objetivo y solución.
    """
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        # Filtramos líneas vacías
        lines = [line.strip() for line in f if line.strip()]

    instancia = {}
    i = 0
    while i < len(lines):
        line = lines[i]

        # Ignoramos las líneas de separadores puros (ej: # #######)
        if line.startswith('#') and "Problema a resolver" in line:
            i += 1
            instancia['problema'] = lines[i]

        elif line.startswith('#') and "Punto objetivo" in line:
            i += 1
            # Si es SVP suele ser "0 0", si es CVP son coordenadas reales
            instancia['target'] = np.array([float(x) for x in lines[i].split()])

        elif line.startswith('#') and "Dimensi" in line:
            i += 1
            dim = int(lines[i])
            instancia['dimension'] = dim

        elif line.startswith('#') and "Base de la celos" in line:
            base_vectors = []
            i += 1
            for _ in range(dim):
                # Leemos tantos vectores como indique la dimensión
                base_vectors.append([float(x) for x in lines[i].split()])
                i += 1
            instancia['base'] = np.array(base_vectors)
            continue  # Evitamos el incremento extra de 'i'

        elif line.startswith('#') and "dulo" in line.lower():
            i += 1
            val = lines[i]
            instancia['modulo'] = int(val) if val.isdigit() else None

        elif line.startswith('#') and "Soluci" in line:
            i += 1
            instancia['solucion'] = np.array([float(x) for x in lines[i].split()])

        i += 1

    return instancia


def cargar_instancia_lwe(filepath):
    """
    Parsea ficheros de instancias LWE (rotura y criptoanálisis).
    Soporta el formato de arrays con llaves {...} y extrae coeficientes y términos.
    """
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = [line.strip() for line in f if line.strip()]

    instancia = {'A_list': []}
    i = 0

    def parse_linea_llaves(text):
        """Limpia las llaves de la sintaxis del CSV y convierte a enteros."""
        clean = text.replace('{', '').replace('}', '').strip()
        if not clean:
            return []
        return [int(x.strip()) for x in clean.split(',') if x.strip()]

    while i < len(lines):
        line = lines[i]

        if line.startswith('#') and "Problema a resolver" in line:
            i += 1
            instancia['tipo'] = lines[i]

        elif line.startswith('#') and "Criptograma" in line:
            i += 1
            instancia['criptograma'] = lines[i]

        elif line.startswith('#') and "Dimensi" in line:
            i += 1
            instancia['dimension'] = int(lines[i])

        elif line.startswith('#') and "m" in line.lower() and "dulo" in line.lower():
            i += 1
            instancia['modulo'] = int(lines[i])

        elif line.startswith('#') and "coeficientes de las ecuaciones" in line:
            i += 1
            # Leemos todas las líneas consecutivas que empiecen por '{'
            while i < len(lines) and lines[i].startswith('{'):
                instancia['A_list'].append(parse_linea_llaves(lines[i]))
                i += 1
            continue

        elif line.startswith('#') and "t" in line.lower() and "rminos independientes" in line:
            i += 1
            instancia['B'] = np.array(parse_linea_llaves(lines[i]))

        elif line.startswith('#') and "Soluci" in line:
            i += 1
            # Saltamos sub-comentarios descriptivos bajo la cabecera de solución
            while i < len(lines) and lines[i].startswith('#'):
                i += 1
            if i < len(lines):
                instancia['solucion'] = np.array(parse_linea_llaves(lines[i]))

        i += 1

    # Convertimos la lista de listas en una matriz 2D de NumPy
    if instancia['A_list']:
        instancia['A'] = np.array(instancia['A_list'])
        del instancia['A_list']

    return instancia