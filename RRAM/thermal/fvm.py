"""Solver térmico por volúmenes finitos: materiales, ensamblado y resolución."""

from collections import Counter
from typing import Optional

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import splu


def crear_matriz_materiales(matriz_filamentos):
    """
    Versión corregida: Respeta los filamentos que toquen los bordes superior/inferior.
    Solo aplica aislante (ID 2) donde antes había aire (ID 0).
    """
    # 1. Copia de seguridad
    inner_map = matriz_filamentos.copy()
    Ny, Nx = inner_map.shape

    # 2. Aplicar Condición de Aislante (Neumann) INTELIGENTE
    # Solo sobrescribimos si NO hay filamento (es decir, si es 0)
    # Fila Superior (0)
    mask_aire_sup = inner_map[0, :] == 0
    inner_map[0, mask_aire_sup] = 2

    # Fila Inferior (-1)
    mask_aire_inf = inner_map[-1, :] == 0
    inner_map[-1, mask_aire_inf] = 2

    # Nota: Si en el borde hay un 1 (Filamento), se queda como 1.
    # El solver aplicará automáticamente la condición adiabática (Neumann)
    # en la frontera del dominio, independientemente del material.

    # 3. Crear Electrodos Laterales (Dirichlet)
    columna_electrodo = np.full((Ny, 1), 3, dtype=int)

    # 4. Ensamblar
    types_map = np.hstack([columna_electrodo, inner_map, columna_electrodo])

    return types_map


#: Material de las celdas de electrodo (Dirichlet con la temperatura del electrodo).
ID_ELECTRODO = 3


#: Material de las celdas de óxido, las únicas sobre las que puede ir un muro térmico.
ID_OXIDO = 0


def _k_efectiva(k_centro: np.ndarray, k_vecino: np.ndarray) -> np.ndarray:
    """Media armónica 2·kc·kv/(kc+kv), con 0 si kc+kv == 0. Mismo orden de operaciones que el bucle original."""
    denom = k_centro + k_vecino
    return np.where(denom == 0, 0.0, (2 * k_centro * k_vecino) / np.where(denom == 0, 1.0, denom))


def _ensamblar_matriz_termica(types_map: np.ndarray, k_por_material: np.ndarray, mascara_muro: np.ndarray):
    """
    Matriz del sistema FVM en CSR.

    Genera las entradas (fila, columna, valor) en EXACTAMENTE el orden y con las mismas operaciones de
    coma flotante que el doble bucle original: por celda, vecinos arriba, abajo, izquierda y derecha
    y después la diagonal; una sola entrada diagonal 1.0 en las celdas Dirichlet (electrodo o muro).
    Así la matriz es idéntica bit a bit a la de antes.
    """
    Ny, Nx = types_map.shape
    N = Ny * Nx
    n = np.arange(N, dtype=np.int64).reshape(Ny, Nx)
    k = k_por_material[types_map]
    dirichlet = (types_map == ID_ELECTRODO) | mascara_muro

    valores = np.zeros((Ny, Nx, 5))
    columnas = np.zeros((Ny, Nx, 5), dtype=np.int64)
    validas = np.zeros((Ny, Nx, 5), dtype=bool)

    # (ranura, filas destino, filas origen) para arriba, abajo, izquierda, derecha
    vecinos = (
        (0, (slice(1, None), slice(None)), (slice(None, -1), slice(None)), -Nx),
        (1, (slice(None, -1), slice(None)), (slice(1, None), slice(None)), Nx),
        (2, (slice(None), slice(1, None)), (slice(None), slice(None, -1)), -1),
        (3, (slice(None), slice(None, -1)), (slice(None), slice(1, None)), 1),
    )
    diagonal = np.zeros((Ny, Nx))
    for ranura, destino, origen, desplazamiento in vecinos:
        k_eff = np.zeros((Ny, Nx))
        k_eff[destino] = _k_efectiva(k[destino], k[origen])
        existe = np.zeros((Ny, Nx), dtype=bool)
        existe[destino] = True
        diagonal = diagonal + k_eff  # 0.0 en los bordes: sumar 0.0 no cambia el valor
        valores[..., ranura] = -k_eff
        columnas[..., ranura] = n + desplazamiento
        validas[..., ranura] = existe & ~dirichlet

    valores[..., 4] = np.where(dirichlet, 1.0, diagonal)
    columnas[..., 4] = n
    validas[..., 4] = True

    filas = np.broadcast_to(n[..., np.newaxis], (Ny, Nx, 5))
    m = validas.ravel()
    A = coo_matrix((valores.ravel()[m], (filas.ravel()[m], columnas.ravel()[m])), shape=(N, N)).tocsr()
    A.sum_duplicates()  # lo mismo que hace spsolve antes de factorizar (aquí no hay duplicados)
    return A


def _factorizar(A_csr):
    """
    Factorización LU reutilizable que reproduce `spsolve(A_csr, b)` bit a bit.

    `spsolve` con una matriz CSR factoriza internamente la traspuesta (los arrays CSR leídos como CSC)
    y resuelve el sistema traspuesto. Hacer lo mismo con `splu` + `trans="T"` da exactamente los
    mismos bits; factorizar `A.tocsc()` no (difiere en ~1e-11 K).
    """
    traspuesta_csc = csc_matrix((A_csr.data, A_csr.indices, A_csr.indptr), shape=A_csr.shape)
    return splu(traspuesta_csc)


class SolverTermico:
    """
    Resuelve el estado térmico estacionario (FVM) y guarda la última factorización.

    La matriz solo depende de la geometría (mapa de materiales), de las conductividades y de qué
    celdas de óxido son muro. Mientras eso no cambie, cada paso solo rehace el vector independiente
    (calor, temperatura del electrodo y del muro) y reutiliza la factorización.
    """

    def __init__(self):
        self._clave = None
        self._lu = None
        self.factorizaciones = 0
        self.reutilizaciones = 0

    def resolver(self, types_map, Q_map, thermal_props, atom_size, T_electrodo, matriz_muros=None) -> np.ndarray:
        types_map = np.asarray(types_map)
        Ny, Nx = types_map.shape
        T_electrodo_filas = np.broadcast_to(np.asarray(T_electrodo, dtype=float), (Ny,))
        k_por_material = np.array([thermal_props[i]["k"] for i in range(max(thermal_props) + 1)], dtype=float)

        # Muro térmico: solo sobre celdas de óxido, nunca sobre filamento (fuente) ni electrodo
        if matriz_muros is None:
            mascara_muro = np.zeros((Ny, Nx), dtype=bool)
        else:
            mascara_muro = (np.asarray(matriz_muros) > 0.0) & (types_map == ID_OXIDO)

        clave = (types_map.shape, types_map.tobytes(), mascara_muro.tobytes(), k_por_material.tobytes())
        if clave != self._clave:
            self._lu = _factorizar(_ensamblar_matriz_termica(types_map, k_por_material, mascara_muro))
            self._clave = clave
            self.factorizaciones += 1
        else:
            self.reutilizaciones += 1

        # Vector independiente: calor en el interior, Dirichlet en electrodos y muros
        cell_volume = atom_size * atom_size
        b = np.asarray(Q_map, dtype=float) * cell_volume
        b = np.where(types_map == ID_ELECTRODO, T_electrodo_filas[:, np.newaxis], b)
        if matriz_muros is not None:
            b = np.where(mascara_muro & (types_map != ID_ELECTRODO), matriz_muros, b)

        T_vec = self._lu.solve(np.ascontiguousarray(b.ravel()), trans="T")
        return np.asarray(T_vec.reshape((Ny, Nx)))


#: Solver por defecto de `solve_thermal_state`. Un proceso simula una sola simulación, y la caché
#: solo guarda la última geometría: el resultado depende únicamente de los argumentos.
_SOLVER_POR_DEFECTO = SolverTermico()


def solve_thermal_state(
    types_map: np.ndarray,
    Q_map: np.ndarray,
    thermal_props: dict,
    atom_size: float,
    T_electrodo,
    matriz_muros: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Ensambla y resuelve la ecuación del calor en estado estacionario (FVM).

    `T_electrodo` es la temperatura de las celdas de electrodo (Dirichlet): un escalar
    común o un perfil (Ny,) con una temperatura por fila.

    Mismo resultado, bit a bit, que el ensamblado celda a celda + `spsolve` original; reutiliza la
    factorización mientras la geometría y el muro no cambien (ver `SolverTermico`).
    """
    return _SOLVER_POR_DEFECTO.resolver(types_map, Q_map, thermal_props, atom_size, T_electrodo, matriz_muros)


def obtener_centro_CF(types_map: np.ndarray, cf_ranges: list) -> list:
    """
    Calcula los centros de los filamentos identificando bloques contiguos en cada columna,
    y obteniendo la coordenada (fila) central que más se repite a lo largo de cada rango definido.

    Argumentos:
    - types_map: Matriz 2D del sistema donde 1 indica filamento y 0 vacío.
    - cf_ranges: (OBLIGATORIO) Lista de tuplas indicando el límite físico de cada filamento en el eje Y.
                 Ejemplo para 1 filamento: [(0, 99)].
                 Ejemplo para 2 filamentos: [(0, 49), (50, 99)].

    Retorna:
    - Una lista con los centros (números enteros de fila). Devuelve un centro exacto por cada rango.
    """
    Ny, Nx = types_map.shape
    todos_los_centros = []

    # Recorremos columnas (evitando electrodos 0 y Nx-1)
    for j in range(1, Nx - 1):
        column_data = types_map[:, j]

        # Encontramos los índices (filas) donde hay filamento en esta columna
        fil_indices = np.where(column_data == 1)[0]

        if len(fil_indices) == 0:
            continue

        # --- 1. ALGORITMO DE AGRUPACIÓN (CLUSTERING) ---
        clusters = []
        current_group = [fil_indices[0]]

        for i in range(1, len(fil_indices)):
            # Si el índice actual es consecutivo al anterior, pertenece al mismo bloque
            if fil_indices[i] == fil_indices[i - 1] + 1:
                current_group.append(fil_indices[i])
            else:
                # Se rompió la continuidad: guardamos el bloque y empezamos uno nuevo
                clusters.append(current_group)
                current_group = [fil_indices[i]]
        clusters.append(current_group)  # Guardar el último bloque

        # --- 2. CÁLCULO DEL CENTRO LOCAL DE CADA CLUSTER ---
        for cluster in clusters:
            # Calculamos la media aritmética de los índices (Ej: [18,19,20] -> 19.0)
            media = np.mean(cluster)

            # Redondeamos al entero más cercano y forzamos a que sea tipo 'int'
            centro_entero = int(np.round(media))

            # Guardamos el centro en la lista global del sistema
            todos_los_centros.append(centro_entero)

    # Si la matriz estaba vacía o no había filamentos en el interior
    if not todos_los_centros:
        return [None] * len(cf_ranges)

    # --- 3. OBTENER EL/LOS CENTROS MÁS REPETIDOS SEGÚN LOS RANGOS ---
    centros_por_filamento = []

    for fila_min, fila_max in cf_ranges:
        # Filtramos solo los centros que caen estrictamente dentro de este filamento físico
        centros_en_rango = [c for c in todos_los_centros if fila_min <= c <= fila_max]

        if centros_en_rango:
            # Obtenemos el que MAS se repite en esta región
            centro_ganador = Counter(centros_en_rango).most_common(1)[0][0]
            centros_por_filamento.append(centro_ganador)
        else:
            # Si un rango no tiene filamento formado, devolvemos None
            centros_por_filamento.append(None)

    return centros_por_filamento
