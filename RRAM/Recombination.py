import logging

import numpy as np
from numba import njit

from . import exceptions
from .topologia_filamentos import filas_banda_filamentos

logger = logging.getLogger(__name__)

k_b_ev = 8.617333262145e-5  # Boltzmann constant in eV/K


#: Filas de la interfaz donde se pueden generar oxígenos: la banda (centro ± grosor) de cada filamento.
filas_interfaz_filamento = filas_banda_filamentos


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


def generate_oxygen(oxygen_state: np.ndarray, max_num_oxygen: int, filas_candidatas: np.ndarray, cupo: int, rng=None) -> np.ndarray:
    """
    Genera una cantidad aleatoria de oxígenos (hasta un máximo de 'max_num_oxygen')
    en la interfaz (columna x=0), solo en las filas de `filas_candidatas`
    (ver filas_libres_interfaz) y sin pasar de `cupo` (ver cupo_oxigenos).
    """
    if max_num_oxygen <= 0:
        return oxygen_state
    rng = np.random if rng is None else rng

    # Genera un número aleatorio entre 0 y el máximo que se indica en max_num_oxygen.
    # Ej: Si max_num_oxygen es 3, puede generar 0, 1, 2 o 3 en este paso.
    cantidad_real = rng.randint(0, max_num_oxygen + 1)

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
        posiciones_elegidas = rng.choice(filas_candidatas, size=cantidad_final, replace=False)

        # Asignamos los oxígenos elegidos
        oxygen_state[posiciones_elegidas, 0] = 1

    return oxygen_state


def Prob_Recombination(
    tiempo_tasa: float,
    paso_temporal: float,
    velocidad: float | np.ndarray,
    temp: float | np.ndarray,
    vibration_frequency: float,
    recom_enchancement_factor: float,
    recombination_energy: float,
    long_decaimiento_concentracion: float,
) -> float | np.ndarray:

    """
    Probabilidad de recombinación en un paso.

    El tiempo entra con dos papeles distintos:
      - `tiempo_tasa`: ventana de tiempo de la tasa (ν·t). Con tiempo acumulativo es el
        tiempo efectivo t_ef; si no, paso_temporal.
      - `paso_temporal`: dt del paso, en exp(-dt·v/L) (desplazamiento del ion en un paso).
        No se acumula: con t_ef el exponente haría caer la probabilidad decenas de órdenes.
    """
    # IMPORTANTE: Usamos np.exp en lugar de math.exp para que pueda manejar tanto floats como arrays (mapas de calor) sin problemas de tipo de variable.
    prob_in_equilibrio = (tiempo_tasa * vibration_frequency) * np.exp(-recombination_energy / (k_b_ev * temp))

    exp_beta = np.exp(-(paso_temporal * velocidad) / long_decaimiento_concentracion) * recom_enchancement_factor
    prob_recom = prob_in_equilibrio * exp_beta

    return prob_recom


def Recombine_opt(
    vacancy_state: np.ndarray,
    oxygen_state: np.ndarray,
    tiempo_tasa: float,
    paso_temporal: float,
    velocidad: np.ndarray | float,
    temperatura: np.ndarray | float,
    vibration_frequency: float,
    recom_enchancement_factor: float,
    recombination_energy: float,
    long_decaimiento_concentracion: float,
    rng=None,
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
        tiempo_tasa=tiempo_tasa,
        paso_temporal=paso_temporal,
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
    random_values = (np.random if rng is None else rng).rand(num_coincidencias)

    # 5. Ver cuáles han tenido éxito
    success_mask_1d = random_values < active_probs

    # 6. Reconstruir las coordenadas 2D de los que tuvieron éxito
    success_indices = (active_indices[0][success_mask_1d], active_indices[1][success_mask_1d])

    # 7. Actualizar el estado físico rompiendo el filamento
    state_updated[success_indices] = 0
    oxygen_updated[success_indices] = 0

    return state_updated, oxygen_updated

@njit(cache=True)
def _mover_con_bloqueo(
    oxygen_state: np.ndarray, reloj: np.ndarray, velocidad: float, paso_temporal: float, grid_size: float
) -> None:
    """
    Mueve los iones IN PLACE (oxygen_state y reloj a la vez) según su propio reloj.

    Cada ion intenta avanzar d = round(v · reloj·dt / celda) celdas hacia la derecha.
    Cada fila se recorre de derecha a izquierda, así el ion de detrás ve ya resuelto al de
    delante (puede ocupar su hueco en el mismo paso):
      - d <= 0: no se mueve y conserva su reloj.
      - sin ion delante y destino fuera de la malla: sale y desaparece (ion y reloj a 0).
      - ion delante: avanza como mucho hasta la celda anterior a él.
      - si avanza >= 1 celda, su reloj pasa a 0; si está bloqueado, conserva su reloj.
    """
    filas, cols = oxygen_state.shape
    factor = velocidad * paso_temporal / grid_size
    for f in range(filas):
        delante = cols  # columna del ion de delante ya resuelto; cols = no hay (borde libre)
        for c in range(cols - 1, -1, -1):
            if oxygen_state[f, c] == 0:
                continue
            d = int(round(factor * reloj[f, c]))
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


def velocidad_arrhenius(
    temperature: np.ndarray | float,
    E_field_filas: np.ndarray,
    cte_red: float,
    vibration_frequency: float,
    gamma_drift: float,
    migration_energy: float,
) -> np.ndarray:
    """
    Velocidad física de deriva de los iones (modelo de referencia, NO se usa en la simulación).

        v = 2 · a · ν · sinh(a · E · γ / (2 k_B T)) · exp(-E_m / (k_B T))

    `E_field_filas` es el campo local de cada fila (un valor por fila); se extiende a toda la fila,
    de modo que con una temperatura escalar o un mapa (filas, columnas) devuelve una velocidad por
    celda. Hasta ahora la simulación usa los umbrales de voltaje de `move_oxygen_ions`; esta función
    queda como punto de partida para recuperar el modelo físico con una velocidad por celda.
    """
    E_field = np.asarray(E_field_filas, dtype=float)[:, np.newaxis]
    senoh = np.sinh((cte_red * E_field * gamma_drift) / (2 * k_b_ev * temperature))
    exp_velocity = np.exp(-migration_energy / (k_b_ev * temperature))
    return 2 * cte_red * vibration_frequency * (senoh * exp_velocity)


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

    `reloj_iones` (int, misma forma que oxygen_state) son los pasos acumulados por cada ion;
    cada uno avanza round(v·reloj·dt/celda) celdas, con bloqueo por el ion de delante
    (ver `_mover_con_bloqueo`). Modifica IN PLACE `oxygen_state` y `reloj_iones`.
    Devuelve (oxygen_state, velocidad).

    La velocidad es la de los umbrales de voltaje. `temperature`, `E_field_filas`, `vibration_frequency`,
    `gamma_drift`, `migration_energy` y `cte_red` NO se usan con este modelo: se conservan en la firma
    hasta limpiar la interfaz y solo sirven al modelo físico de referencia `velocidad_arrhenius`.
    """

    # =========================================================================
    # 1. VELOCIDAD POR UMBRALES DE VOLTAJE (modelo vigente)
    # =========================================================================
    # Inicializamos con el valor por defecto (si no supera ningún umbral)
    oxygen_velocity = 0.0

    # Umbrales de mayor a menor |V|: la comparación es en valor absoluto, así que el orden también
    for threshold in sorted(velocity_thresholds.keys(), key=abs, reverse=True):
        if abs(voltage) > abs(threshold):
            oxygen_velocity = float(velocity_thresholds[threshold])
            break  # Salimos al encontrar el primer umbral que se cumple

    # =========================================================================
    # 2. MOVIMIENTO CON RELOJ POR ION Y BLOQUEO
    # =========================================================================
    _mover_con_bloqueo(oxygen_state, reloj_iones, oxygen_velocity, paso_temporal, grid_size)

    return oxygen_state, oxygen_velocity
