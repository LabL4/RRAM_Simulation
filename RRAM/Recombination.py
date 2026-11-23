from numba import njit
import numpy as np
import logging
import math
import sys

from . import exceptions

logger = logging.getLogger(__name__)

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


def filas_interfaz_filamento(Ny: int, centros, grosor) -> np.ndarray:
    """
    Máscara 1D (Ny,) de las filas de la interfaz donde se pueden generar oxígenos:
    centro ± grosor de cada filamento.

    `grosor` es un entero común o una lista con un valor por filamento.
    """
    grosores = [int(grosor)] * len(centros) if isinstance(grosor, (int, np.integer)) else [int(g) for g in grosor]
    if len(grosores) != len(centros):
        raise ValueError("Debe haber un grosor asignado para cada centro de filamento.")
    filas = np.zeros(Ny, dtype=bool)
    for centro, g in zip(centros, grosores):
        filas[max(0, int(centro) - g) : min(Ny, int(centro) + g + 1)] = True
    return filas


def filas_libres_interfaz(oxygen_state: np.ndarray, filas_permitidas: np.ndarray) -> np.ndarray:
    """
    Índices de las filas de la interfaz (columna x=0) sin ion y dentro de `filas_permitidas`.
    """
    return np.where((oxygen_state[:, 0] == 0) & filas_permitidas)[0]


def resolver_limite_oxigenos(vacantes_generadas_set, max_oxigenos_reset) -> int:
    """
    Máximo de iones de oxígeno simultáneos en el reset.

    - Valor manual (`max_oxigenos_reset`, de simulation_config.py): manda siempre; si además
      existe el del set, se avisa en el log de que se usa el de la configuración.
    - Solo el del set (`vacantes_generadas_set`): se usa ese.
    - Ninguno: error en el log y LimiteOxigenosNoDefinidoException (se para la simulación).
    """
    if max_oxigenos_reset is not None:
        if vacantes_generadas_set is not None:
            logger.warning(
                f"Límite de oxígenos del reset: se usa el valor del archivo de configuración "
                f"max_oxigenos_reset={int(max_oxigenos_reset)} en lugar del heredado de la fase anterior "
                f"({int(vacantes_generadas_set)})."
            )
        else:
            logger.info(f"Límite de oxígenos del reset: max_oxigenos_reset={int(max_oxigenos_reset)} (archivo de configuración).")
        return int(max_oxigenos_reset)
    if vacantes_generadas_set is not None:
        logger.info(f"Límite de oxígenos del reset: vacantes_generadas_set={int(vacantes_generadas_set)} (obtenido en el set).")
        return int(vacantes_generadas_set)
    raise exceptions.LimiteOxigenosNoDefinidoException()


def cupo_oxigenos(oxygen_state: np.ndarray, max_oxigenos: int) -> int:
    """
    Iones de oxígeno que aún caben: max_oxigenos - iones presentes (nunca negativo).
    """
    return max(0, int(max_oxigenos) - int(np.sum(oxygen_state)))


def generate_oxygen(oxygen_state: np.ndarray, max_num_oxygen: int, filas_candidatas: np.ndarray, cupo: int) -> np.ndarray:
    """
    Genera una cantidad aleatoria de oxígenos (hasta un máximo de 'max_num_oxygen')
    en la interfaz (columna x=0), solo en las filas de `filas_candidatas`
    (ver filas_libres_interfaz) y sin pasar de `cupo` (ver cupo_oxigenos).
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

    # Limitamos por si la interfaz estuviera casi llena y no cupieran todos
    # y por el cupo de iones simultáneos
    cantidad_final = min(cantidad_real, len(filas_candidatas), cupo)

    if cantidad_final > 0:
        # Elegimos aleatoriamente las posiciones SIN repetición (replace=False)
        posiciones_elegidas = np.random.choice(filas_candidatas, size=cantidad_final, replace=False)

        # Asignamos los oxígenos elegidos
        oxygen_state[posiciones_elegidas, 0] = 1

    return oxygen_state


def Prob_Recombination(
    tiempo_tasa: float | np.ndarray,
    temp: float | np.ndarray,
    vibration_frequency: float,
    recom_enchancement_factor: float,
    recombination_energy: float,
    realce_max: float,
    tau_c: float,
) -> float | np.ndarray:
    """
    Probabilidad de recombinación en un paso (sin saturar):

        P = ν · t · exp(-E_r / (k_B·T)) · β · g(t),   g(t) = 1 + (G - 1) · t / (t + τ_c)

    - `tiempo_tasa` (t): reloj de igual voltaje del ion · dt (RelojesIones.reloj). El mismo
      tiempo entra en la tasa y en el realce g.
    - `temp`: temperatura de la celda.
    Ambos admiten escalar o matriz (celda a celda). g(0) = 1 y g → G para t >> τ_c.
    """
    # np.exp en lugar de math.exp: admite escalares y matrices.
    prob_base = (tiempo_tasa * vibration_frequency) * np.exp(-recombination_energy / (k_b_ev * temp))
    realce = 1.0 + (realce_max - 1.0) * tiempo_tasa / (tiempo_tasa + tau_c)
    return prob_base * recom_enchancement_factor * realce


def Recombine_opt(
    vacancy_state: np.ndarray,
    oxygen_state: np.ndarray,
    prob_recombinacion: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sortea la recombinación en las celdas donde coinciden ion y vacante, con la matriz de
    probabilidad ya calculada (`prob_recombinacion`, misma forma que vacancy_state y ya
    saturada a 1). La celda que recombina pierde la vacante y el ion.
    """
    state_updated = np.copy(vacancy_state)
    oxygen_updated = np.copy(oxygen_state)

    # 1. Coordenadas donde coinciden un ion y una vacante
    active_indices = np.where((oxygen_state == 1) & (vacancy_state == 1))
    num_coincidencias = len(active_indices[0])
    if num_coincidencias == 0:
        return state_updated, oxygen_updated

    # 2. Probabilidad de esas celdas y sorteo
    active_probs = prob_recombinacion[active_indices]
    random_values = np.random.rand(num_coincidencias)
    success_mask_1d = random_values < active_probs

    # 3. Coordenadas 2D de los que tuvieron éxito
    success_indices = (active_indices[0][success_mask_1d], active_indices[1][success_mask_1d])

    # 4. Actualizar el estado físico rompiendo el filamento
    state_updated[success_indices] = 0
    oxygen_updated[success_indices] = 0

    return state_updated, oxygen_updated

@njit(cache=True)
def _mover_con_bloqueo(
    oxygen_state: np.ndarray,
    reloj: np.ndarray,
    velocidad: float,
    paso_temporal: float,
    grid_size: float,
) -> None:
    """
    Mueve los iones IN PLACE (oxygen_state y reloj a la vez) según su propio reloj.

    Cada ion intenta avanzar d = floor(v · reloj·dt / celda) celdas hacia la derecha
    (solo avanza cuando ha recorrido una celda completa).
    Cada fila se recorre de derecha a izquierda, así el ion de detrás ve ya resuelto al de
    delante (puede ocupar su hueco en el mismo paso):
      - d <= 0: no se mueve y conserva su reloj.
      - sin ion delante y destino fuera de la malla: sale y desaparece (ion y reloj a 0).
      - ion delante: avanza como mucho hasta la celda anterior a él.
      - si avanza >= 1 celda, su reloj pasa a 0; si está bloqueado, lo conserva.
    """
    filas, cols = oxygen_state.shape
    factor = velocidad * paso_temporal / grid_size
    for f in range(filas):
        delante = cols  # columna del ion de delante ya resuelto; cols = no hay (borde libre)
        for c in range(cols - 1, -1, -1):
            if oxygen_state[f, c] == 0:
                continue
            d = int(math.floor(factor * reloj[f, c]))
            if d <= 0:
                delante = c
                continue
            destino = c + d
            if delante == cols and destino >= cols:
                oxygen_state[f, c] = 0
                reloj[f, c] = 0
                continue
            nueva = min(destino, delante - 1)
            if nueva > c:
                oxygen_state[f, c] = 0
                reloj[f, c] = 0
                oxygen_state[f, nueva] = 1
                reloj[f, nueva] = 0
                delante = nueva
            else:
                delante = c  # bloqueado: sigue acumulando


def move_oxygen_ions(
    reloj_iones: np.ndarray,
    oxygen_state: np.ndarray,
    temperature: np.ndarray | float,
    E_field_filas: np.ndarray,
    grid_size: float,
    paso_temporal: float,
    vibration_frequency: float,
    gamma_drift: float,
    migration_energy: float,
    cte_red: float,
    voltage: float,
    velocity_thresholds: dict,  # Diccionario pasado como argumento
) -> tuple[np.ndarray, float]:
    """
    Mueve los iones de oxígeno según el reloj propio de cada uno. La velocidad se determina
    comparando el parámetro 'voltage' contra los umbrales en 'velocity_thresholds'.

    `reloj_iones` (int, misma forma que oxygen_state) son los pasos acumulados por cada ion
    (reloj de igual voltaje); cada uno avanza floor(v·reloj·dt/celda) celdas, con bloqueo
    por el ion de delante (ver `_mover_con_bloqueo`).
    Modifica IN PLACE `oxygen_state` y `reloj_iones`.
    Devuelve (oxygen_state, velocidad).

    `E_field_filas` es el campo local de cada fila (GapElectricField, un valor por
    fila de oxygen_state). Hoy solo alimenta la velocidad física de referencia; el
    objetivo es volver a ese modelo físico, con una velocidad por celda.
    """

    # =========================================================================
    # 1. CÁLCULO FÍSICO DE LA VELOCIDAD (Referencia que luego no se usa)
    # =========================================================================
    # Campo de cada fila extendido a toda la fila: (filas, 1) se combina con una
    # temperatura escalar o con el mapa (filas, columnas) y da una velocidad por celda.
    E_field = np.asarray(E_field_filas, dtype=float)[:, np.newaxis]
    try:
        senoh = np.sinh((cte_red * E_field * gamma_drift) / (2 * k_b_ev * temperature))
        exp_velocity = np.exp(-migration_energy / (k_b_ev * temperature))
        # Nota: Esta variable se calcula pero será sobrescrita por la lógica de umbrales
        oxygen_velocity_fisica = 2 * cte_red * vibration_frequency * (senoh * exp_velocity)

    except OverflowError as Overflow_exception:
        logger.info(f"\n Error en el cálculo de la velocidad: {Overflow_exception}")
        sys.exit(1)

    # =========================================================================
    # 2. LÓGICA DINÁMICA DE UMBRALES
    # =========================================================================
    # Inicializamos con el valor por defecto (si no supera ningún umbral)
    oxygen_velocity = 0.0

    # Ordenamos las llaves de mayor a menor para asegurar la asignación correcta
    for threshold in sorted(velocity_thresholds.keys(), reverse=True):
        if abs(voltage) > abs(threshold):
            oxygen_velocity = float(velocity_thresholds[threshold])
            break  # Salimos al encontrar el primer umbral que se cumple

    # =========================================================================
    # 3. MOVIMIENTO CON RELOJ POR ION Y BLOQUEO
    # =========================================================================
    _mover_con_bloqueo(oxygen_state, reloj_iones, oxygen_velocity, paso_temporal, grid_size)

    return oxygen_state, oxygen_velocity
