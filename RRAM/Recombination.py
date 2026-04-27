import numpy as np
import math
import sys

k_b_ev = 8.617333262145e-5  # Boltzmann constant in eV/K


def Init_OxygenState(device_size_x: float, device_size_y: float, atom_size: float):
    """
    Inicializa la matriz de estado de iones de oxígeno con las dimensiones del dieléctrico.

    Parámetros:
    - device_size_x (float): Distancia entre electrodos [m] → shape[0].
    - device_size_y (float): Ancho del dispositivo [m] → shape[1].
    - atom_size (float): Tamaño de celda de la malla [m].

    Retorna:
    - InitialOxygenState (np.ndarray): Matriz de ceros con shape (eje_x, eje_y).
    """
    eje_x = int(math.ceil(device_size_x / atom_size))
    eje_y = int(math.ceil(device_size_y / atom_size))

    InitialOxygenState = np.zeros((eje_x, eje_y), dtype=int)

    return InitialOxygenState


def generate_oxygen(oxygen_state: np.ndarray, max_num_oxygen: int) -> np.ndarray:
    """
    Genera una cantidad aleatoria de oxígenos (hasta un máximo de 'max_num_oxygen')
    en posiciones libres de la interfaz (columna x=0).
    """
    if max_num_oxygen <= 0:
        return oxygen_state

    # Genera un número aleatorio entre 0 y el máximo que se indica en max_num_oxygen.
    # Ej: Si max_num_oxygen es 3, puede generar 0, 1, 2 o 3 en este paso.
    cantidad_real = np.random.randint(0, max_num_oxygen + 1)

    # (Nota: Si prefieres que simule un "lanzamiento de moneda" independiente
    # por cada oxígeno con un 60% de probabilidad de éxito, podrías usar:
    # cantidad_real = np.random.binomial(n=max_num_oxygen, p=0.6) )

    if cantidad_real == 0:
        return oxygen_state

    # Buscamos qué posiciones (índices de fila) están libres en la columna 0
    posiciones_libres = np.where(oxygen_state[:, 0] == 0)[0]

    # Limitamos por si la interfaz estuviera casi llena y no cupieran todos
    cantidad_final = min(cantidad_real, len(posiciones_libres))

    if cantidad_final > 0:
        # Elegimos aleatoriamente las posiciones SIN repetición (replace=False)
        posiciones_elegidas = np.random.choice(posiciones_libres, size=cantidad_final, replace=False)

        # Asignamos los oxígenos elegidos
        oxygen_state[posiciones_elegidas, 0] = 1

    return oxygen_state


def Prob_Recombination(
    paso_temporal: float,
    velocidad: float | np.ndarray,
    temp: float | np.ndarray,
    vibration_frequency: float,
    recom_enchancement_factor: float,
    recombination_energy: float,
    long_decaimiento_concentracion: float,
) -> float | np.ndarray:

    # IMPORTANTE: Usamos np.exp en lugar de math.exp para que pueda manejar tanto floats como arrays (mapas de calor) sin problemas de tipo de variable.
    prob_in_equilibrio = (paso_temporal * vibration_frequency) * np.exp(-recombination_energy / (k_b_ev * temp))

    exp_beta = np.exp(-(paso_temporal * velocidad) / long_decaimiento_concentracion) * recom_enchancement_factor
    prob_recom = prob_in_equilibrio * exp_beta

    return prob_recom


def Recombine_opt(
    vacancy_state: np.ndarray,
    oxygen_state: np.ndarray,
    paso_temp: float,
    velocidad: np.ndarray | float,
    temperatura: np.ndarray | float,
    vibration_frequency: float,
    recom_enchancement_factor: float,
    recombination_energy: float,
    long_decaimiento_concentracion: float,
) -> tuple[np.ndarray, np.ndarray]:

    state_updated = np.copy(vacancy_state)
    oxygen_updated = np.copy(oxygen_state)

    # 1. Buscar las coordenadas exactas donde coinciden un ion y una vacante
    active_indices = np.where((oxygen_state == 1) & (vacancy_state == 1))

    # Guardamos cuántas coincidencias hubo
    num_coincidencias = len(active_indices[0])

    # Si no hay coincidencias, salimos
    if num_coincidencias == 0:
        return state_updated, oxygen_updated

    # =========================================================================
    # 2. GESTIÓN INTELIGENTE DE MATRICES (Extraer solo los datos útiles)
    # =========================================================================

    # Filtrar TEMPERATURAS
    if isinstance(temperatura, np.ndarray):
        active_temps = temperatura[active_indices]
    else:
        active_temps = temperatura

    # Filtrar VELOCIDADES (¡La solución al problema!)
    if isinstance(velocidad, np.ndarray):
        active_vels = velocidad[active_indices]
    else:
        active_vels = velocidad

    # =========================================================================
    # 3. Calcular probabilidades
    # =========================================================================
    active_probs = Prob_Recombination(
        paso_temporal=paso_temp,
        velocidad=active_vels,
        temp=active_temps,
        vibration_frequency=vibration_frequency,
        recom_enchancement_factor=recom_enchancement_factor,
        recombination_energy=recombination_energy,
        long_decaimiento_concentracion=long_decaimiento_concentracion,
    )

    # Saturamos a 1.0 por seguridad usando la versión de NumPy
    active_probs = np.minimum(active_probs, 1.0)

    # 4. Tirar los dados
    random_values = np.random.rand(num_coincidencias)

    # 5. Ver cuáles han tenido éxito
    success_mask_1d = random_values < active_probs

    # 6. Reconstruir las coordenadas 2D de los que tuvieron éxito
    success_indices = (active_indices[0][success_mask_1d], active_indices[1][success_mask_1d])

    # 7. Actualizar el estado físico rompiendo el filamento
    state_updated[success_indices] = 0
    oxygen_updated[success_indices] = 0

    return state_updated, oxygen_updated


def move_oxygen_ions(
    paso_temp: float,
    oxygen_state: np.ndarray,
    temperature: np.ndarray | float,
    E_field: float,
    grid_size: float,
    vibration_frequency: float,
    gamma_drift: float,
    migration_energy: float,
    cte_red: float,
    voltage: float,
    velocity_thresholds: dict,  # Diccionario pasado como argumento
) -> tuple[np.ndarray, np.ndarray | float]:
    """
    Mueve los iones de oxígeno de forma estocástica. La velocidad se determina
    comparando el parámetro 'voltage' contra los umbrales en 'velocity_thresholds'.
    """

    # =========================================================================
    # 1. CÁLCULO FÍSICO DE LA VELOCIDAD (Referencia que luego no se usa)
    # =========================================================================
    try:
        senoh = np.sinh((cte_red * E_field * gamma_drift) / (2 * k_b_ev * temperature))
        exp_velocity = np.exp(-migration_energy / (k_b_ev * temperature))
        # Nota: Esta variable se calcula pero será sobrescrita por la lógica de umbrales
        oxygen_velocity_fisica = 2 * cte_red * vibration_frequency * (senoh * exp_velocity)

    except OverflowError as Overflow_exception:
        print(f"\n Error en el cálculo de la velocidad: {Overflow_exception}")
        sys.exit(1)

    # =========================================================================
    # 2. LÓGICA DINÁMICA DE UMBRALES
    # =========================================================================
    # Inicializamos con el valor por defecto (si no supera ningún umbral)
    oxygen_velocity = 0

    # Ordenamos las llaves de mayor a menor para asegurar la asignación correcta
    for threshold in sorted(velocity_thresholds.keys(), reverse=True):
        if abs(voltage) > abs(threshold):
            oxygen_velocity = velocity_thresholds[threshold]
            # print(f"Voltage {voltage} supera el umbral {threshold}, asignando velocidad {oxygen_velocity}")
            break  # Salimos al encontrar el primer umbral que se cumple

    # =========================================================================
    # 3. CÁLCULO DEL DESPLAZAMIENTO MÁXIMO
    # =========================================================================
    max_displacement = np.array(np.round((oxygen_velocity * paso_temp) / grid_size), dtype=int)
    oxygen_state_new = np.zeros_like(oxygen_state)

    # =========================================================================
    # 4. MOVIMIENTO ALEATORIO INDEPENDIENTE
    # =========================================================================
    rows, cols = np.where(oxygen_state == 1)
    num_ions = len(rows)

    if num_ions > 0:
        actual_shifts = np.zeros(num_ions, dtype=int)

        if max_displacement.ndim > 0:
            max_allowed = max_displacement[rows, cols]
            mask = max_allowed > 0
            if np.any(mask):
                actual_shifts[mask] = np.random.randint(1, max_allowed[mask] + 1)
        else:
            if max_displacement > 0:
                actual_shifts = np.random.randint(1, max_displacement + 1, size=num_ions)

        new_cols = cols + actual_shifts

        # Verificamos límites del array
        valid_mask = (new_cols >= 0) & (new_cols < oxygen_state.shape[1])
        valid_rows = rows[valid_mask]
        valid_new_cols = new_cols[valid_mask]

        oxygen_state_new[valid_rows, valid_new_cols] = 1

    return oxygen_state_new, oxygen_velocity
