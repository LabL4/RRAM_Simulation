from scipy.sparse.linalg import spsolve
from scipy.sparse import coo_matrix
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


# =============================================================================
# SOLVER TÉRMICO (ECUACIÓN DEL CALOR FVM)
# =============================================================================

from RRAM.Simulation import SimulationParameters, SimulationConstants
from scipy.sparse.linalg import spsolve
from scipy.sparse import coo_matrix
import numpy as np

# =============================================================================
# SOLVER TÉRMICO FVM (Basado en Temperature_Final.ipynb)
# =============================================================================


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


def calculate_heat_source(types_map: np.ndarray, atom_size: float, I_total: float, R_cell: float) -> np.ndarray:
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

        # Factor de escala (0.01)
        Q_map_global[fil_indices, j] = Q_val_local

    return Q_map_global


def solve_thermal_state(
    types_map: np.ndarray, Q_map: np.ndarray, thermal_props: dict, atom_size: float, T_ambient: float
) -> np.ndarray:
    """
    Ensambla y resuelve la ecuación del calor en estado estacionario (FVM).

    Matemática:
        Sum(Flux_vecinos) + Source = 0
        Flux_vecino = k_eff * (T_vecino - T_centro) * (Area / Distancia)

    Args:
        types_map (np.ndarray): Matriz de materiales extendida.
        Q_map (np.ndarray): Matriz de calor Joule [W/m^3].
        thermal_props (dict): Diccionario {ID: {'k': valor}}.
        atom_size (float): Tamaño de la celda [m].
        T_ambient (float): Temperatura fija para condiciones Dirichlet [K].

    Returns:
        np.ndarray: Mapa de temperaturas completo [K].
    """
    Ny, Nx = types_map.shape
    N = Ny * Nx  # Número total de incógnitas

    # Listas para formato COO (Sparse Matrix)
    data = []
    rows_idx = []
    cols_idx = []
    b = np.zeros(N)

    # Volumen de la celda de control (asumiendo espesor unitario dz=1 o cancelándose)
    # En 2D estacionario: Balance de Calor [W] -> Q [W/m^3] * Volumen [m^3]
    # Si asumimos 2D puro, trabajamos por unidad de profundidad.
    cell_volume = atom_size * atom_size

    for i in range(Ny):
        for j in range(Nx):
            n = i * Nx + j  # Índice lineal actual
            mat_id = types_map[i, j]

            # --- CONDICIÓN DIRICHLET (Temperatura Fija) ---
            # Si es Electrodo (ID 3), fijamos T = T_ambient
            if mat_id == 3:
                data.append(1.0)
                rows_idx.append(n)
                cols_idx.append(n)
                b[n] = T_ambient
                continue

            # --- CONDICIÓN NEUMANN / ECUACIÓN DEL CALOR ---
            # Para materiales 0 (Oxido), 1 (Filamento) o 2 (Aislante)
            # Nota: Si es ID 2 (Aislante), su k es muy baja (~0 PERO NO 0), por lo que
            # naturalmente se comportará como adiabático con sus vecinos.

            k_center = thermal_props[mat_id]["k"]
            diag_sum = 0.0

            # Lista de vecinos (Arriba, Abajo, Izquierda, Derecha)
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

                # Conductividad Efectiva en la interfaz (Media Armónica)
                # k_eff = 2 * k1 * k2 / (k1 + k2)
                # Evitamos división por cero si ambos son aislantes perfectos
                denom = k_center + k_vec
                if denom == 0:
                    k_eff = 0.0
                else:
                    k_eff = (2 * k_center * k_vec) / denom

                # Aporte a la matriz A (Término del vecino pasa restando al lado izq)
                # Flux = k_eff * (T_vec - T_cen). En eq: ... - k_eff * T_vec ...
                data.append(-k_eff)
                rows_idx.append(n)
                cols_idx.append(vec_n)

                # Aporte a la diagonal (Suma de coeficientes)
                diag_sum += k_eff

            # Elemento diagonal principal (Coeficiente de T_center)
            data.append(diag_sum)
            rows_idx.append(n)
            cols_idx.append(n)

            # Término fuente (Calor Joule)
            # Ecuación: Divergencia(Flux) = Q
            # Discretizada: Sum(k_eff*(T_cen - T_vec)) = Q * Volumen
            b[n] = Q_map[i, j] * cell_volume

    # --- RESOLUCIÓN DEL SISTEMA ---
    # Construir matriz dispersa
    A_mat = coo_matrix((data, (rows_idx, cols_idx)), shape=(N, N))
    A_csr = A_mat.tocsr()

    # Resolver Ax = b
    T_vec = spsolve(A_csr, b)

    # Reconvertir vector 1D a matriz 2D
    T_final = T_vec.reshape((Ny, Nx))

    return np.asarray(T_final).reshape(Ny, Nx)
