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
    Construye la matriz extendida de materiales con electrodos y bordes aislantes.

    Convención de ejes:
        shape[0] = eje X (filas) = distancia entre electrodos.
        shape[1] = eje Y (columnas) = ancho transversal.

    Los electrodos (ID 3) se añaden como FILAS en los extremos del eje X (Dirichlet).
    El aislante (ID 2) se aplica en las COLUMNAS laterales del eje Y donde no hay filamento (Neumann).
    Resultado: shape (x_size + 2, y_size).
    """
    inner_map = matriz_filamentos.copy()
    x_size, y_size = inner_map.shape  # filas = X, columnas = Y

    # Aplicar aislante (Neumann) en columnas laterales Y solo donde no hay filamento
    mask_aire_izq = inner_map[:, 0] == 0
    inner_map[mask_aire_izq, 0] = 2

    mask_aire_der = inner_map[:, -1] == 0
    inner_map[mask_aire_der, -1] = 2

    # Añadir electrodos como filas en los extremos del eje X (Dirichlet)
    fila_electrodo = np.full((1, y_size), 3, dtype=int)
    types_map = np.vstack([fila_electrodo, inner_map, fila_electrodo])  # (x_size+2, y_size)

    return types_map


def calculate_heat_source(
    types_map: np.ndarray, atom_size: float, I_total: float, R_cell: float, factor_generar_calor: float
) -> np.ndarray:
    """
    Calcula el mapa de calor Q [W/m^3] usando aproximación resistiva por filas X.

    Convención de ejes:
        shape[0] = eje X (filas) = dirección entre electrodos.
        shape[1] = eje Y (columnas) = ancho transversal.

    Para cada fila X interior, las celdas de filamento en dirección Y actúan como
    resistencias en paralelo. La corriente circula a lo largo de X.

    Args:
        types_map (np.ndarray): Matriz extendida con electrodos (shape: x_size+2, y_size).
        atom_size (float): Tamaño de celda 'h' [m].
        I_total (float): Corriente total [A].
        R_cell (float): Resistencia óhmica de un nodo [Ohm].
        factor_generar_calor (float): Factor de escala para la generación de calor.
    """
    x_size_ext, y_size = types_map.shape  # x_size_ext = x_size+2 (con electrodos), y_size = y_size
    Q_map_global = np.zeros((x_size_ext, y_size))

    sigma_material = 1.0 / (R_cell * atom_size)

    # Iteramos sobre filas X internas (excluyendo electrodos en fila 0 y fila -1)
    for i in range(1, x_size_ext - 1):
        row_data = types_map[i, :]
        fil_indices = np.where(row_data == 1)[0]  # índices Y (columnas) con filamento
        N_total_fila = len(fil_indices)

        if N_total_fila == 0:
            continue

        # R equivalente de la fila (N celdas Y en paralelo)
        R_fila = R_cell / N_total_fila

        # Caída de tensión local (I_total pasa por esta fila X)
        delta_V_local = I_total * R_fila

        # Campo eléctrico local: E = V / h
        E_local = delta_V_local / atom_size

        # Calor Joule: Q = sigma * E^2, asignado a las celdas Y del filamento en esta fila X
        Q_val_local = sigma_material * (E_local**2)
        Q_map_global[i, fil_indices] = Q_val_local * factor_generar_calor

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

    Convención de ejes:
        shape[0] = eje X (filas) = dirección entre electrodos. Electrodos en filas 0 y -1.
        shape[1] = eje Y (columnas) = ancho transversal. Aislante adiabático en columnas 0 y -1.

    Args:
        types_map (np.ndarray): Matriz de materiales con electrodos (shape: x_size+2, y_size).
        Q_map (np.ndarray): Mapa de calor Joule [W/m^3] (misma shape que types_map).
        thermal_props (dict): {ID_material: {'k': conductividad}}.
        atom_size (float): Tamaño de la celda [m].
        T_ambient (float): Temperatura fija para condiciones Dirichlet [K].
        matriz_muros (np.ndarray, optional): Mapa de muros térmicos (misma shape).
    """
    x_size_ext, y_size = types_map.shape  # x_size_ext = x_size+2, y_size = y_size
    N = x_size_ext * y_size

    data = []
    rows_idx = []
    cols_idx = []
    b = np.zeros(N)

    cell_volume = atom_size * atom_size

    for i in range(x_size_ext):      # i recorre eje X (filas)
        for j in range(y_size):      # j recorre eje Y (columnas)
            n = i * y_size + j       # índice lineal

            # --- CONDICIÓN DIRICHLET POR MURO TÉRMICO ---
            if matriz_muros is not None and matriz_muros[i, j] > 0.0:
                data.append(1.0)
                rows_idx.append(n)
                cols_idx.append(n)
                b[n] = matriz_muros[i, j]
                continue

            mat_id = types_map[i, j]

            # --- CONDICIÓN DIRICHLET (Electrodos, ID 3) ---
            if mat_id == 3:
                data.append(1.0)
                rows_idx.append(n)
                cols_idx.append(n)
                b[n] = T_ambient
                continue

            # --- ECUACIÓN DEL CALOR (Neumann en bordes Y, interior) ---
            k_center = thermal_props[mat_id]["k"]
            diag_sum = 0.0

            vecinos = []
            if i > 0:
                vecinos.append(((i - 1, j), (i - 1) * y_size + j))        # X anterior
            if i < x_size_ext - 1:
                vecinos.append(((i + 1, j), (i + 1) * y_size + j))        # X siguiente
            if j > 0:
                vecinos.append(((i, j - 1), i * y_size + (j - 1)))        # Y anterior
            if j < y_size - 1:
                vecinos.append(((i, j + 1), i * y_size + (j + 1)))        # Y siguiente

            for (vec_i, vec_j), vec_n in vecinos:
                mat_vec = types_map[vec_i, vec_j]
                k_vec = thermal_props[mat_vec]["k"]

                denom = k_center + k_vec
                k_eff = 0.0 if denom == 0 else (2 * k_center * k_vec) / denom

                data.append(-k_eff)
                rows_idx.append(n)
                cols_idx.append(vec_n)
                diag_sum += k_eff

            data.append(diag_sum)
            rows_idx.append(n)
            cols_idx.append(n)

            b[n] = Q_map[i, j] * cell_volume

    A_mat = coo_matrix((data, (rows_idx, cols_idx)), shape=(N, N))
    T_vec = spsolve(A_mat.tocsr(), b)
    return np.asarray(T_vec.reshape((x_size_ext, y_size)))


def obtener_centro_CF(types_map: np.ndarray, cf_ranges: list) -> list:
    """
    Calcula los centros en el eje Y (columnas) de cada filamento.

    Convención de ejes:
        shape[0] = eje X (filas) = dirección entre electrodos.
        shape[1] = eje Y (columnas) = ancho transversal, donde se separan los filamentos.

    Para cada fila X interior, identifica bloques contiguos de filamento en el eje Y
    y calcula su centro. Devuelve el centro Y más repetido para cada rango definido en cf_ranges.

    Argumentos:
    - types_map: Matriz 2D con electrodos (shape: x_size+2, y_size). 1=filamento, 0=vacío, 3=electrodo.
    - cf_ranges: Lista de tuplas con los rangos Y de cada filamento, ej: [(0, 49), (50, 99)].

    Retorna:
    - Lista de centros Y (enteros) por filamento. None si el rango no tiene filamento.
    """
    x_size_ext, y_size = types_map.shape  # x_size_ext = x_size+2, y_size = y_size
    todos_los_centros = []

    # Recorremos filas X internas (excluyendo electrodos en fila 0 y fila -1)
    for i in range(1, x_size_ext - 1):
        row_data = types_map[i, :]

        # Encontramos los índices Y (columnas) donde hay filamento en esta fila X
        fil_indices = np.where(row_data == 1)[0]

        if len(fil_indices) == 0:
            continue

        # --- 1. CLUSTERING de índices Y contiguos ---
        clusters = []
        current_group = [fil_indices[0]]

        for k in range(1, len(fil_indices)):
            if fil_indices[k] == fil_indices[k - 1] + 1:
                current_group.append(fil_indices[k])
            else:
                clusters.append(current_group)
                current_group = [fil_indices[k]]
        clusters.append(current_group)

        # --- 2. CENTRO Y de cada cluster ---
        for cluster in clusters:
            media = np.mean(cluster)
            centro_entero = int(np.round(media))
            todos_los_centros.append(centro_entero)

    if not todos_los_centros:
        return [None] * len(cf_ranges)

    # --- 3. CENTRO MÁS REPETIDO por rango Y ---
    centros_por_filamento = []

    for col_min, col_max in cf_ranges:
        centros_en_rango = [c for c in todos_los_centros if col_min <= c <= col_max]

        if centros_en_rango:
            centro_ganador = Counter(centros_en_rango).most_common(1)[0][0]
            centros_por_filamento.append(centro_ganador)
        else:
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
    Coloca los perfiles 1D de temperatura en las columnas Y intermedias de la matriz 2D.

    Convención de ejes:
        shape[0] = eje X (filas) = dirección entre electrodos.
        shape[1] = eje Y (columnas) = ancho transversal.

    Los muros térmicos se colocan en columnas Y (entre filamentos, en la dirección transversal).
    Los perfiles asociados son vectores de temperatura a lo largo del eje X (longitud x_size).

    Argumentos:
    - matriz_molde: Matriz de referencia para la máscara (shape: x_size, y_size).
    - filas_intermedias: Lista de índices Y (columnas) de las posiciones intermedias entre filamentos.
    - perfiles_muros_calculados: Lista de tuplas (perfil_arriba, perfil_abajo), arrays 1D de longitud x_size.
    """
    matriz_muros = np.zeros_like(matriz_molde, dtype=float)
    x_size, y_size = matriz_muros.shape

    for i, col_mid in enumerate(filas_intermedias):
        if col_mid is None:
            continue

        perfil_muro_arriba, perfil_muro_abajo = perfiles_muros_calculados[i]

        # 1er Muro: columna Y intermedia
        if 0 <= col_mid < y_size:
            mask_vacio = matriz_molde[:, col_mid] == 0
            matriz_muros[mask_vacio, col_mid] = perfil_muro_arriba[mask_vacio]

        # 2do Muro: columna Y siguiente
        col_siguiente = col_mid + 1
        if 0 <= col_siguiente < y_size:
            mask_vacio_sig = matriz_molde[:, col_siguiente] == 0
            matriz_muros[mask_vacio_sig, col_siguiente] = perfil_muro_abajo[mask_vacio_sig]

    return matriz_muros


def extraer_perfiles_filamentos(matriz_temperaturas: np.ndarray, filas_centros: list) -> list:
    """
    Extrae los perfiles de temperatura 1D a lo largo del eje X para cada filamento.

    Convención de ejes:
        shape[0] = eje X (filas) = dirección entre electrodos.
        shape[1] = eje Y (columnas) = ancho transversal.

    Para cada filamento, su 'centro' es un índice Y (columna). El perfil extraído
    es el vector de temperatura a lo largo del eje X en esa columna Y fija:
        perfil = matriz_temperaturas[:, centro_y]

    Argumentos:
    - matriz_temperaturas: Matriz 2D con los resultados del solver térmico (shape: x_size+2, y_size).
    - filas_centros: Lista de enteros Y (índices de columna) de los centros de cada filamento.

    Retorna:
    - perfiles: Lista de arrays 1D (longitud x_size+2) de temperatura por filamento.
    """
    perfiles = []
    x_size_ext, y_size = matriz_temperaturas.shape

    for columna_y in filas_centros:
        if columna_y is None:
            perfiles.append(None)
            print("Advertencia: Se detectó un filamento no formado (None). Perfil omitido.")

        elif 0 <= columna_y < y_size:
            # Perfil X completo en la columna Y del centro del filamento
            perfil_1d = matriz_temperaturas[:, columna_y].copy()
            perfiles.append(perfil_1d)

        else:
            raise IndexError(
                f"La columna Y solicitada ({columna_y}) está fuera de los límites "
                f"de la matriz térmica (0 a {y_size - 1})."
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
