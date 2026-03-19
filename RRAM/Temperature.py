from scipy.sparse.linalg import spsolve
from scipy.sparse import coo_matrix
from collections import Counter
import matplotlib.pyplot as plt
from typing import Optional
import numpy as np


def Temperature_Joule(potencial: float, intensidad: float, T_0: float, r_termica: float) -> float:
    """
    Calcula el incremento de temperatura debido al calentamiento por efecto Joule.

    Esta función determina el aumento de temperatura en un dispositivo causado por
    la disipación de potencia eléctrica (efecto Joule), considerando la resistencia
    térmica del sistema.

    Args:
        potencial (float): Diferencia de potencial aplicada en Voltios [V]
        intensidad (float): Corriente eléctrica que circula en Amperios [A]
        T_0 (float): Temperatura inicial o ambiente en Kelvin [K] (parámetro no utilizado)
        r_termica (float): Resistencia térmica del dispositivo en K/W [K/W]

    Returns:
        float: Incremento de temperatura por efecto Joule en Kelvin [K]

    Fórmula:
        ΔT = |V * I| * R_th
        donde P = V * I es la potencia disipada y R_th es la resistencia térmica del sistema.
    """
    temperatura_disipacion = T_0 + abs(potencial * intensidad) * r_termica

    return temperatura_disipacion


def legacy_matriz_materiales(matriz_filamentos: np.ndarray) -> np.ndarray:
    """
    Recibe la matriz de estado (0: Vacante/Aire, 1: Filamento).
    Devuelve la matriz ampliada incluyendo electrodos y bordes aislantes.
    Mapeo: 0=Oxido, 1=Filamento, 2=Aislante, 3=Electrodo.
    """
    rows, cols = matriz_filamentos.shape

    # 1. Crear matriz vacía con 2 filas extra (Electrodos Arriba y Abajo)
    # Dimensiones finales: (rows + 2, cols)
    # Inicializamos con 0 (Óxido)
    types_map = np.zeros((rows + 2, cols), dtype=int)

    # 2. Definir Electrodos (Fila 0 y Fila -1) -> ID 3
    types_map[0, :] = 3
    types_map[-1, :] = 3

    # 3. Copiar la matriz de filamentos en el centro
    # Desplazamos índice de fila en +1
    types_map[1:-1, :] = matriz_filamentos

    # En el notebook tenía bordes laterales aislantes (ID 2),
    # se mantiene aquí. Por defecto en FVM, si no hay vecino, es adiabático (aislante).
    # Si se quiere cambiar esto para forzar material '2' que es adibático en los lados, descomentamos esto:
    # types_map[1:-1, 0] = 2
    # types_map[1:-1, -1] = 2

    return types_map


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


def calculate_heat_source(
    types_map: np.ndarray, atom_size: float, I_total: float, R_cell: float, factor_generar_calor: float
) -> np.ndarray:
    """
    Calcula el mapa de calor Q [W/m^3] usando aproximación resistiva por columnas.
    Recibe la resistencia de celda y deduce la conductividad internamente para garantizar consistencia.

    Args:
        types_map (np.ndarray): Matriz extendida con electrodos.
        atom_size (float): Tamaño de celda 'h' [m].
        I_total (float): Corriente total [A].
        R_cell (float): Resistencia óhmica de un nodo [Ohm].
    """
    Ny, Nx = types_map.shape
    Q_map_global = np.zeros((Ny, Nx))

    # 1. Calculamos sigma localmente (Coste despreciable: 1 división)
    # Garantizamos que sigma y R son coherentes siempre.
    sigma_material = 1.0 / (R_cell * atom_size)

    # 2. Iteramos sobre columnas internas que es donde se encuentra el espacio de simulación real (quitando electrodos)
    for j in range(1, Nx - 1):
        column_data = types_map[:, j]
        fil_indices = np.where(column_data == 1)[0]
        N_total_columna = len(fil_indices)

        if N_total_columna == 0:
            continue

        # R equivalente de la columna (N resistencias en paralelo)
        R_col = R_cell / N_total_columna

        # Caída de tensión local (Aproximación: I_total pasa por esta columna)
        delta_V_local = I_total * R_col

        # Campo eléctrico local: E = V / h
        E_local = delta_V_local / atom_size

        # Calor Joule: Q = sigma * E^2
        Q_val_local = sigma_material * (E_local**2)

        # Factor de escala para convertir a W/m^3: Q_local [W/m^3] = sigma * (V/h)^2
        Q_map_global[fil_indices, j] = Q_val_local * factor_generar_calor

    return Q_map_global


# def solve_thermal_state(
#     types_map: np.ndarray, Q_map: np.ndarray, thermal_props: dict, atom_size: float, T_ambient: float
# ) -> np.ndarray:
#     """
#     Ensambla y resuelve la ecuación del calor en estado estacionario (FVM).

#     Matemática:
#         Sum(Flux_vecinos) + Source = 0
#         Flux_vecino = k_eff * (T_vecino - T_centro) * (Area / Distancia)

#     Args:
#         types_map (np.ndarray): Matriz de materiales extendida.
#         Q_map (np.ndarray): Matriz de calor Joule [W/m^3].
#         thermal_props (dict): Diccionario {ID: {'k': valor}}.
#         atom_size (float): Tamaño de la celda [m].
#         T_ambient (float): Temperatura fija para condiciones Dirichlet [K].

#     Returns:
#         np.ndarray: Mapa de temperaturas completo [K].
#     """
#     Ny, Nx = types_map.shape
#     N = Ny * Nx  # Número total de incógnitas

#     # Listas para formato COO (Sparse Matrix)
#     data = []
#     rows_idx = []
#     cols_idx = []
#     b = np.zeros(N)

#     # Volumen de la celda de control (asumiendo espesor unitario dz=1 o cancelándose)
#     # En 2D estacionario: Balance de Calor [W] -> Q [W/m^3] * Volumen [m^3]
#     # Si asumimos 2D puro, trabajamos por unidad de profundidad.
#     cell_volume = atom_size * atom_size

#     for i in range(Ny):
#         for j in range(Nx):
#             n = i * Nx + j  # Índice lineal actual
#             mat_id = types_map[i, j]

#             # --- CONDICIÓN DIRICHLET (Temperatura Fija) ---
#             # Si es Electrodo (ID 3), fijamos T = T_ambient
#             if mat_id == 3:
#                 data.append(1.0)
#                 rows_idx.append(n)
#                 cols_idx.append(n)
#                 b[n] = T_ambient
#                 continue

#             # --- CONDICIÓN NEUMANN / ECUACIÓN DEL CALOR ---
#             # Para materiales 0 (Oxido), 1 (Filamento) o 2 (Aislante)
#             # Nota: Si es ID 2 (Aislante), su k es muy baja (~0 PERO NO 0), por lo que
#             # naturalmente se comportará como adiabático con sus vecinos.

#             k_center = thermal_props[mat_id]["k"]
#             diag_sum = 0.0

#             # Lista de vecinos (Arriba, Abajo, Izquierda, Derecha)
#             vecinos = []
#             if i > 0:
#                 vecinos.append(((i - 1, j), (i - 1) * Nx + j))  # Arriba
#             if i < Ny - 1:
#                 vecinos.append(((i + 1, j), (i + 1) * Nx + j))  # Abajo
#             if j > 0:
#                 vecinos.append(((i, j - 1), i * Nx + (j - 1)))  # Izquierda
#             if j < Nx - 1:
#                 vecinos.append(((i, j + 1), i * Nx + (j + 1)))  # Derecha

#             for (vec_i, vec_j), vec_n in vecinos:
#                 mat_vec = types_map[vec_i, vec_j]
#                 k_vec = thermal_props[mat_vec]["k"]

#                 # Conductividad Efectiva en la interfaz (Media Armónica)
#                 # k_eff = 2 * k1 * k2 / (k1 + k2)
#                 # Evitamos división por cero si ambos son aislantes perfectos
#                 denom = k_center + k_vec
#                 if denom == 0:
#                     k_eff = 0.0
#                 else:
#                     k_eff = (2 * k_center * k_vec) / denom

#                 # Aporte a la matriz A (Término del vecino pasa restando al lado izq)
#                 # Flux = k_eff * (T_vec - T_cen). En eq: ... - k_eff * T_vec ...
#                 data.append(-k_eff)
#                 rows_idx.append(n)
#                 cols_idx.append(vec_n)

#                 # Aporte a la diagonal (Suma de coeficientes)
#                 diag_sum += k_eff

#             # Elemento diagonal principal (Coeficiente de T_center)
#             data.append(diag_sum)
#             rows_idx.append(n)
#             cols_idx.append(n)

#             # Término fuente (Calor Joule)
#             # Ecuación: Divergencia(Flux) = Q
#             # Discretizada: Sum(k_eff*(T_cen - T_vec)) = Q * Volumen
#             b[n] = Q_map[i, j] * cell_volume

#     # --- RESOLUCIÓN DEL SISTEMA ---
#     # Construir matriz dispersa
#     A_mat = coo_matrix((data, (rows_idx, cols_idx)), shape=(N, N))
#     A_csr = A_mat.tocsr()

#     # Resolver Ax = b
#     T_vec = spsolve(A_csr, b)

#     # Reconvertir vector 1D a matriz 2D
#     T_final = T_vec.reshape((Ny, Nx))

#     return np.asarray(T_final)


def solve_thermal_state(
    types_map: np.ndarray,
    Q_map: np.ndarray,
    thermal_props: dict,
    atom_size: float,
    T_ambient: float,
    matriz_muros: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Ensambla y resuelve la ecuación del calor en estado estacionario (FVM).
    """
    Ny, Nx = types_map.shape
    N = Ny * Nx  # Número total de incógnitas

    data = []
    rows_idx = []
    cols_idx = []
    b = np.zeros(N)

    cell_volume = atom_size * atom_size

    # print(
    #     f"La matriz con la temperatura de los muros térmicos es:\n{matriz_muros}\n"
    # )  # Depuración: Imprime la matriz de muros

    for i in range(Ny):
        for j in range(Nx):
            n = i * Nx + j  # Índice lineal actual

            # ==========================================================
            # --- NUEVO: CONDICIÓN DIRICHLET POR MURO TÉRMICO ---
            # Si nos han pasado la matriz de muros y esta celda tiene un valor > 0
            # ==========================================================
            if matriz_muros is not None and matriz_muros[i, j] > 0.0:
                data.append(1.0)  # Coeficiente diagonal = 1
                rows_idx.append(n)
                cols_idx.append(n)
                b[n] = matriz_muros[i, j]  # Temperatura fijada al perfil del muro
                continue  # Saltamos al siguiente píxel
            # ==========================================================

            mat_id = types_map[i, j]

            # --- CONDICIÓN DIRICHLET (Temperatura Fija - Electrodos) ---
            if mat_id == 3:
                data.append(1.0)
                rows_idx.append(n)
                cols_idx.append(n)
                b[n] = T_ambient
                continue

            # --- CONDICIÓN NEUMANN / ECUACIÓN DEL CALOR ---
            k_center = thermal_props[mat_id]["k"]
            diag_sum = 0.0

            vecinos = []
            if i > 0:
                vecinos.append(((i - 1, j), (i - 1) * Nx + j))  # Arriba
            if i < Ny - 1:
                vecinos.append(((i + 1, j), (i + 1) * Nx + j))  # Abajo
            if j > 0:
                vecinos.append(((i, j - 1), i * Nx + (j - 1)))  # Izquierda
            if j < Nx - 1:
                vecinos.append(((i, j + 1), i * Nx + (j + 1)))  # Derecha

            for (vec_i, vec_j), vec_n in vecinos:
                mat_vec = types_map[vec_i, vec_j]
                k_vec = thermal_props[mat_vec]["k"]

                denom = k_center + k_vec
                if denom == 0:
                    k_eff = 0.0
                else:
                    k_eff = (2 * k_center * k_vec) / denom

                data.append(-k_eff)
                rows_idx.append(n)
                cols_idx.append(vec_n)

                diag_sum += k_eff

            data.append(diag_sum)
            rows_idx.append(n)
            cols_idx.append(n)

            b[n] = Q_map[i, j] * cell_volume

    # --- RESOLUCIÓN DEL SISTEMA ---
    A_mat = coo_matrix((data, (rows_idx, cols_idx)), shape=(N, N))
    A_csr = A_mat.tocsr()

    T_vec = spsolve(A_csr, b)

    T_final = T_vec.reshape((Ny, Nx))

    # Aseguramos el tipo np.ndarray para satisfacer al linter (Pylance)
    return np.asarray(T_final)


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


def calcular_filas_intermedias(centros: list) -> tuple[list, list]:
    """
    Calcula la fila central (punto medio) entre los centros de filamentos consecutivos
    y las distancias físicas desde los centros hasta dicha fila media.

    Argumentos:
    - centros: Lista de enteros con los centros de cada filamento (Ej: [20, 50, 80]).
    - atom_size: Tamaño físico de cada celda del grid (Ej: 0.25).

    Retorna:
    - filas_medias: Lista de enteros con las filas intermedias (Ej: [35, 65]).
    - distancias: Lista de tuplas con las distancias desde cada filamento a la fila media (Ej: [(15, 15), (15, 15)]).
                  Formato: [(dist_centro1_a_medio, dist_centro2_a_medio), ...]
    """
    filas_medias = []
    distancias = []

    # Si hay menos de 2 filamentos, no hay distancia intermedia que calcular
    if not centros or len(centros) < 2:
        return filas_medias, distancias

    # Recorremos la lista emparejando el elemento actual con el siguiente
    for i in range(len(centros) - 1):
        centro_actual = centros[i]
        centro_siguiente = centros[i + 1]

        # En caso de que un filamento no se haya formado (None), no podemos calcular el medio
        if centro_actual is None or centro_siguiente is None:
            continue

        # Calculamos la media aritmética entre ambos centros
        medio = (centro_actual + centro_siguiente) / 2.0

        # Redondeamos al entero más cercano (fila de la matriz)
        fila_media = int(np.round(medio))
        filas_medias.append(fila_media)

        # Calculamos cuántas casillas hay de diferencia usando valor absoluto (abs)
        # y lo multiplicamos por el tamaño físico de la celda (atom_size)
        distancia_actual = abs(fila_media - centro_actual)
        distancia_siguiente = abs(centro_siguiente - fila_media)

        # Guardamos ambas distancias en la lista
        distancias.append((distancia_actual, distancia_siguiente))

    return filas_medias, distancias


def calcular_perfiles_muro(
    perfiles_filamentos: list,
    distancias_casillas: list,
    pendiente_temperatura: float,
    atom_size: float,
    T_ambient: float,
) -> list:
    """
    Calcula el perfil de temperatura 1D para cada muro térmico basándose en
    la temperatura de su filamento correspondiente columna a columna.
    """

    # BLOQUE 1: Preparación del contenedor de resultados
    # -----------------------------------------------------------------
    # Crearemos una lista que almacenará tuplas. Cada tupla contendrá
    # (perfil_muro_arriba, perfil_muro_abajo) en formato de array 1D.
    perfiles_muros = []

    # Verificación de seguridad: debe haber un perfil de filamento más que distancias
    if len(perfiles_filamentos) < 2 or not distancias_casillas:
        return perfiles_muros

    # BLOQUE 2: Iteración sobre las interfaces (muros)
    # -----------------------------------------------------------------
    # Recorremos la lista de distancias. El índice 'i' representa el espacio
    # INTERMEDIO entre el filamento 'i' (arriba) y el filamento 'i-1' (abajo).
    for i in range(len(distancias_casillas)):
        casillas_arriba, casillas_abajo = distancias_casillas[i]

        # BLOQUE 3: Extracción de los perfiles de los filamentos origen
        # -----------------------------------------------------------------
        # El muro superior está influenciado por el filamento que tiene encima (i)
        T_filamento_arriba = perfiles_filamentos[i]

        # El muro inferior está influenciado por el filamento que tiene debajo (i+1)
        T_filamento_abajo = perfiles_filamentos[i - 1]

        # print("El perfil de temperatura del filamento superior en la funcion del muro es:", T_filamento_arriba[10:21])
        # print("El perfil de temperatura del filamento inferior en la funcion del muro es:", T_filamento_abajo[10:21])

        # BLOQUE 4: Cálculo vectorizado de la temperatura
        # -----------------------------------------------------------------
        # Como T_filamento es un array de numpy (ej: 100 valores, uno por columna),
        # al sumar el término matemático se realiza la operación celda por celda
        # instantáneamente sin necesidad de hacer un bucle for anidado.

        # Ecuación: T_muro = T_filamento + pendiente_temperatura * distancia
        # (Nota: 'pendiente_temperatura' debe ser negativo ya que la temperatura cae al alejarse)
        T_muro_arriba = T_filamento_arriba + pendiente_temperatura * (casillas_arriba * atom_size * 1e9)
        T_muro_abajo = T_filamento_abajo + pendiente_temperatura * (casillas_abajo * atom_size * 1e9)

        # Imprimir los primeros 10 números del perfil para depuración
        # print("Temperatura calculada para el muro superior antes de limitar:", T_muro_arriba[10:21])
        # print("Temperatura calculada para el muro inferior antes de limitar:", T_muro_abajo[10:21])

        # Si la temperatura calculada para el muro es mayor que la del filamento, la limitamos a la temperatura del filamento
        T_muro_arriba = np.minimum(T_muro_arriba, T_filamento_arriba)
        T_muro_abajo = np.minimum(T_muro_abajo, T_filamento_abajo)

        # Si la temperatura calculada para el muro es menor que la temperatura ambiente, la limitamos a la temperatura ambiente
        T_muro_arriba = np.maximum(T_muro_arriba, T_ambient)
        T_muro_abajo = np.maximum(T_muro_abajo, T_ambient)

        # print("Temperatura calculada para el muro superior después de limitar:", T_muro_arriba[10:21])
        # print("Temperatura calculada para el muro inferior después de limitar:", T_muro_abajo[10:21])

        # Guardamos el par de perfiles calculados para esta interfaz
        perfiles_muros.append((T_muro_arriba, T_muro_abajo))

    return perfiles_muros


def colocar_muro_termico(
    matriz_molde: np.ndarray, filas_intermedias: list, perfiles_muros_calculados: list
) -> np.ndarray:
    """
    Coloca los perfiles 1D de temperatura en las filas correspondientes de la matriz 2D.
    """
    matriz_muros = np.zeros_like(matriz_molde, dtype=float)
    Ny, Nx = matriz_muros.shape

    for i, fila_mid in enumerate(filas_intermedias):
        if fila_mid is None:
            continue

        # Extraemos los perfiles 1D ya calculados por tu nueva función
        perfil_muro_arriba, perfil_muro_abajo = perfiles_muros_calculados[i]

        # 1er Muro: Fila Intermedia
        mask_vacio_arriba = matriz_molde[fila_mid, :] == 0
        # Numpy asigna el valor del perfil SOLO en las columnas donde la máscara es True
        matriz_muros[fila_mid, mask_vacio_arriba] = perfil_muro_arriba[mask_vacio_arriba]

        # 2do Muro: Una fila por debajo
        fila_encima = fila_mid + 1
        if 0 <= fila_encima < Ny:
            mask_vacio_abajo = matriz_molde[fila_encima, :] == 0
            matriz_muros[fila_encima, mask_vacio_abajo] = perfil_muro_abajo[mask_vacio_abajo]

    return matriz_muros


def extraer_perfiles_filamentos(matriz_temperaturas: np.ndarray, filas_centros: list) -> list:
    """
    Extrae los perfiles de temperatura 1D (fila completa) de una matriz térmica 2D,
    basándose en una lista de filas específicas (centros de los filamentos).

    Argumentos:
    - matriz_temperaturas: Matriz 2D con los resultados del solver térmico.
    - filas_centros: Lista de enteros indicando las filas a extraer (Ej: [20, 80]).

    Retorna:
    - perfiles: Lista de arrays 1D correspondientes a la temperatura de cada filamento.
    """

    # BLOQUE 1: Preparación
    perfiles = []
    Ny, Nx = matriz_temperaturas.shape

    # BLOQUE 2: Iteración y Extracción
    for fila in filas_centros:
        # Caso A: El filamento no se formó (su centro es None)
        if fila is None:
            # Añadimos None a la lista para mantener la correspondencia de índices
            perfiles.append(None)
            print("Advertencia: Se detectó un filamento no formado (None). Perfil omitido.")

        # Caso B: La fila es un índice válido dentro de la matriz
        elif 0 <= fila < Ny:
            # Extraemos la fila completa (todas las columnas de esa fila)
            # Usamos .copy() para evitar modificar la matriz original accidentalmente
            perfil_1d = matriz_temperaturas[fila, :].copy()
            perfiles.append(perfil_1d)

        # Caso C: Índice fuera de los límites de la matriz (Error de entrada)
        else:
            raise IndexError(
                f"La fila solicitada ({fila}) está fuera de los límites de la matriz térmica (0 a {Ny - 1})."
            )

    return perfiles


def extraer_perfiles_temperatura(lista_matrices: list, etiquetas: list, columna_x: int, atom_size: float):
    """
    Extrae el perfil vertical de temperatura para múltiples matrices.

    Argumentos:
    - lista_matrices: Lista de arrays 2D de temperatura.
    - etiquetas: Nombres para la leyenda (ej: ["1.0 V", "1.5 V", "2.0 V"]).
    - columna_x: Índice de la columna a evaluar.
    - atom_size: Tamaño físico de cada celda en nm.

    Retorna:
    - distancias: Array 1D con el eje X físico.
    - perfiles: Diccionario con formato { "etiqueta": array_temperaturas_1D }
    """
    if len(lista_matrices) != len(etiquetas):
        raise ValueError("Debe haber el mismo número de matrices que de etiquetas.")

    # Usamos la forma de la primera matriz para calcular las distancias
    distancias = np.arange(0, lista_matrices[0].shape[0]) * atom_size

    perfiles = {}
    for matriz, etiqueta in zip(lista_matrices, etiquetas):
        # Extraemos solo la columna deseada
        perfiles[etiqueta] = matriz[:, columna_x]

    return distancias, perfiles
