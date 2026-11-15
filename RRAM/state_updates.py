"""Actualizaciones del estado del sistema (generación y recombinación)."""

import logging
from typing import Callable

import numpy as np

from . import Recombination
from . import Generation
from .acumulador_tiempo import RelojesIones
from .constants_simulation import SimulationConstants
from .parameters import SimulationParameters

logger = logging.getLogger(__name__)


def update_state_generation(
    state: np.ndarray,
    params: SimulationParameters,
    sim_ctes: SimulationConstants,
    E_field: np.ndarray | float,
    temperatura: np.ndarray | float,
    factor_vecinos: float,
    factor_sin_vecinos: float,
    max_vacantes_permitidas: int,
    neighbor_mode: str = "both",
    custom_mask: np.ndarray | None = None,
    num_iteracion=None,
    tiempo_efectivo: float | None = None,
    rng=None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Orquesta el proceso de generación de vacantes asegurando que no se supere
    el límite máximo establecido. Si las candidatas superan el cupo, se eligen
    al azar entre ellas (todas con la misma oportunidad).

    Devuelve (estado, probabilidades).
    """

    act_state = state.copy()
    n_filas, n_cols = state.shape

    # 1. Ajuste del Campo Eléctrico
    if isinstance(E_field, np.ndarray) and E_field.ndim == 1:
        E_field_matrix = np.repeat(E_field[:, np.newaxis], n_cols, axis=1)
    else:
        E_field_matrix = E_field

    # 2. Ajuste de la matriz de Temperatura
    if isinstance(temperatura, np.ndarray) and temperatura.shape != state.shape:
        temp_matrix = temperatura[:, 1:-1]
    else:
        temp_matrix = temperatura

    # 3. Cálculo de la matriz de probabilidades
    prob_final = Generation.get_generation_probabilities_matrix(
        state=state,
        paso_temporal=params.paso_temporal if tiempo_efectivo is None else tiempo_efectivo,
        Electric_field=E_field_matrix,
        temperatura=temp_matrix,
        factor_vecinos=factor_vecinos,
        factor_sin_vecinos=factor_sin_vecinos,
        vibration_frequency=sim_ctes.vibration_frequency,
        generation_energy=sim_ctes.generation_energy,
        cte_red=sim_ctes.cte_red,
        gamma=sim_ctes.gamma,
        num_iteracion=num_iteracion,
        neighbor_mode=neighbor_mode,
        custom_mask=custom_mask,
    )

    # 4. Evaluación de la estocástica (cuáles "intentan" generarse)
    # TODO para que todas las simulaciones sean iguales, eliminar dps
    aleatorios = (np.random if rng is None else rng).rand(n_filas, n_cols)
    nueva_vacante = aleatorios < prob_final

    # 5. LIMITACIÓN INTELIGENTE DE VACANTES
    vacantes_actuales = np.sum(act_state)
    num_nuevas = np.sum(nueva_vacante)
    vacantes_disponibles = int(max_vacantes_permitidas - vacantes_actuales)

    if vacantes_disponibles <= 0:
        # Ya estamos en el límite, no se permite generar ninguna más
        return act_state, prob_final

    if num_nuevas > vacantes_disponibles:
        # Se ha superado el límite: se eligen al azar `vacantes_disponibles` candidatas.
        # Todas pasaron ya el sorteo `aleatorios < prob_final`, así que la probabilidad
        # ya está tenida en cuenta; ordenar por ella la aplicaría dos veces y, con
        # empates (p = 1), la elección la decidiría el orden interno de argsort.
        logger.warning(
            f"Paso {num_iteracion}: {num_nuevas} vacantes candidatas superan el cupo "
            f"({vacantes_disponibles} disponibles de {max_vacantes_permitidas}); "
            f"se eligen {vacantes_disponibles} al azar entre las candidatas."
        )

        # Obtenemos las coordenadas (i,j) de las candidatas
        coords = np.argwhere(nueva_vacante)

        # Elección al azar sin repetición (con el `rng` de la fase)
        elegidas = coords[(np.random if rng is None else rng).choice(len(coords), vacantes_disponibles, replace=False)]

        # Actualizamos la matriz solo con las elegidas
        act_state[elegidas[:, 0], elegidas[:, 1]] = 1
    else:
        # Si no se ha superado el límite, se aprueban todas
        act_state[nueva_vacante] = 1

    return act_state, prob_final


def update_state_recombinate(
    voltage: float,
    E_field_filas: np.ndarray,
    oxygen_config: dict,
    sim_ctes: SimulationConstants,
    params: SimulationParameters,
    actual_state: np.ndarray,
    oxygen_state: np.ndarray,
    temperatura: float | np.ndarray,
    velocity_thresholds: dict,  # Diccionario pasado como argumento
    relojes: RelojesIones,
    filas_permitidas: np.ndarray,
    max_oxigenos: int,
    tiempo_efectivo: float | None = None,
    rng=None,
) -> tuple[np.ndarray, np.ndarray, Callable[[], np.ndarray]]:
    """
    Orquesta el proceso completo de RESET en un paso de tiempo.

    `filas_permitidas` (Ny,) son las filas de la interfaz donde se pueden generar
    oxígenos (banda de los filamentos, ver Recombination.filas_interfaz_filamento).
    `max_oxigenos` es el máximo de iones simultáneos (ver Recombination.resolver_limite_oxigenos).
    `tiempo_efectivo` multiplica las tasas de recombinación (si no se pasa, vale `paso_temporal`);
    el exponente exp(-dt·v/L) usa siempre `paso_temporal`.
    `relojes` lleva el reloj propio de cada ion: el llamador ya lo ha avanzado este paso
    (`relojes.actualizar`, antes de generar); aquí se mueve con él y, tras recombinar, se
    borra el reloj de los iones desaparecidos (`relojes.limpiar`).
    Devuelve (vacantes, oxígenos, campo_probabilidad):
      - campo_probabilidad es una FUNCIÓN sin argumentos que devuelve el CAMPO de probabilidad de
        recombinación de este paso en toda la matriz (misma forma que actual_state), con el mismo
        tiempo, velocidad y temperatura que el sorteo real; solo se usa de verdad en las celdas con
        ion y vacante. Se devuelve sin evaluar porque solo se necesita al guardar el estado (cada
        `pasos_guardar_estado`) y evaluarlo en todos los pasos cuesta exponenciales sobre toda la
        matriz. Se evalúa con `evaluar_campo_probabilidad`.

    1. Generación de iones oxígenos por voltaje (solo en `filas_permitidas`).
    2. Movimiento de iones.
    3. Recombinación.
    """

    # 1. Se generan oxígenos según el voltaje
    for threshold_voltage, max_oxigenos_paso in oxygen_config.items():
        if abs(voltage) > abs(threshold_voltage):
            filas_candidatas = Recombination.filas_libres_interfaz(oxygen_state, filas_permitidas)
            cupo = Recombination.cupo_oxigenos(oxygen_state, max_oxigenos)
            oxygen_state = Recombination.generate_oxygen(oxygen_state, max_oxigenos_paso, filas_candidatas, cupo, rng)
            break  # Solo usar el umbral más alto superado

    # 2. Movemos los ionesx
    oxygen_state, velocidad = Recombination.move_oxygen_ions(
        reloj_iones=relojes.reloj,
        oxygen_state=oxygen_state,
        temperature=temperatura,
        E_field_filas=E_field_filas,
        grid_size=params.atom_size,
        paso_temporal=params.paso_temporal,
        vibration_frequency=sim_ctes.vibration_frequency,
        gamma_drift=sim_ctes.gamma_drift,
        migration_energy=sim_ctes.E_m,
        cte_red=sim_ctes.cte_red,
        voltage=voltage,
        velocity_thresholds=velocity_thresholds,
    )

    # print(f"Velocidad de los iones oxígeno en este paso: {velocidad:.4e} m/s")

    # 3.Recombinación de iones con vacantes. t_ef solo multiplica la tasa (ν·t); el exponente
    # exp(-dt·v/L) usa siempre el dt del paso (ver Recombination.Prob_Recombination).
    tiempo_tasa = params.paso_temporal if tiempo_efectivo is None else tiempo_efectivo
    actual_state_update, oxygen_state_update = Recombination.Recombine_opt(
        vacancy_state=actual_state,
        oxygen_state=oxygen_state,
        tiempo_tasa=tiempo_tasa,
        paso_temporal=params.paso_temporal,
        velocidad=velocidad,
        temperatura=temperatura,
        vibration_frequency=sim_ctes.vibration_frequency,
        recom_enchancement_factor=sim_ctes.recom_enchancement_factor,
        recombination_energy=sim_ctes.recombination_energy,
        long_decaimiento_concentracion=sim_ctes.long_decaimiento_concentracion,
        rng=rng,
    )
    relojes.limpiar(oxygen_state_update)  # iones recombinados: su reloj a 0

    # 4. Campo de probabilidad de recombinación (para guardarlo, como la matriz de generación).
    # Se devuelve diferido: captura los valores de ESTE paso y solo se calcula si alguien lo pide.
    forma = actual_state.shape

    def campo_probabilidad() -> np.ndarray:
        prob = np.minimum(
            Recombination.Prob_Recombination(
                tiempo_tasa=tiempo_tasa,
                paso_temporal=params.paso_temporal,
                velocidad=velocidad,
                temp=temperatura,
                vibration_frequency=sim_ctes.vibration_frequency,
                recom_enchancement_factor=sim_ctes.recom_enchancement_factor,
                recombination_energy=sim_ctes.recombination_energy,
                long_decaimiento_concentracion=sim_ctes.long_decaimiento_concentracion,
            ),
            1.0,
        )
        return np.broadcast_to(prob, forma).copy()

    return actual_state_update, oxygen_state_update, campo_probabilidad


def evaluar_campo_probabilidad(campo: Callable[[], np.ndarray] | None) -> np.ndarray | None:
    """Evalúa el campo diferido de `update_state_recombinate`; None si aún no hubo ningún paso."""
    return None if campo is None else campo()
