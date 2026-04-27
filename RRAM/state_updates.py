"""Actualizaciones del estado del sistema (generación y recombinación)."""

import numpy as np

from . import Generation, Recombination
from .constants_simulation import SimulationConstants
from .parameters import SimulationParameters


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
) -> tuple[np.ndarray, np.ndarray]:
    """
    Orquesta el proceso de generación de vacantes asegurando que no se supere
    el límite máximo establecido. Prioriza aquellas con mayor probabilidad.
    """

    # En NumPy el primer valor devuelto por shape son las filas y el segundo las columnas, por lo que asumo que x_size son tus filas en el eje Y físico, y y_size son tus columnas en el eje X físico. Mientras mantengas esa lógica, es perfecto).
    act_state = state.copy()
    x_size, y_size = state.shape

    # 1. Ajuste del Campo Eléctrico
    if isinstance(E_field, np.ndarray) and E_field.ndim == 1:
        E_field_matrix = np.repeat(E_field[:, np.newaxis], y_size, axis=1)
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
        paso_temporal=params.paso_temporal,
        Electric_field=E_field_matrix,
        temperatura=temp_matrix,
        factor_vecinos=factor_vecinos,
        factor_sin_vecinos=factor_sin_vecinos,
        vibration_frequency=sim_ctes.vibration_frequency,
        generation_energy=sim_ctes.generation_energy,
        cte_red=sim_ctes.cte_red,
        gamma=sim_ctes.gamma,
        neighbor_mode=neighbor_mode,
        custom_mask=custom_mask,
    )

    # 4. Evaluación de la estocástica (cuáles "intentan" generarse)
    aleatorios = np.random.rand(x_size, y_size)
    nueva_vacante = aleatorios < prob_final

    # 5. LIMITACIÓN INTELIGENTE DE VACANTES
    vacantes_actuales = np.sum(act_state)
    num_nuevas = np.sum(nueva_vacante)
    vacantes_disponibles = int(max_vacantes_permitidas - vacantes_actuales)

    if vacantes_disponibles <= 0:
        # Ya estamos en el límite, no se permite generar ninguna más
        return act_state, prob_final

    if num_nuevas > vacantes_disponibles:
        # Se ha superado el límite. Toca priorizar.

        # Obtenemos las coordenadas (i,j) de las candidatas
        coords = np.argwhere(nueva_vacante)

        # Extraemos las probabilidades exactas que tenían estas candidatas
        probs_candidatas = prob_final[nueva_vacante]

        # Ordenamos los índices de las candidatas de MAYOR a MENOR probabilidad
        indices_ordenados = np.argsort(probs_candidatas)[::-1]

        # Nos quedamos estrictamente con las mejores según el cupo disponible
        mejores_indices = indices_ordenados[:vacantes_disponibles]
        mejores_coords = coords[mejores_indices]

        # Actualizamos la matriz solo con las ganadoras
        act_state[mejores_coords[:, 0], mejores_coords[:, 1]] = 1
    else:
        # Si no se ha superado el límite, se aprueban todas
        act_state[nueva_vacante] = 1

    return act_state, prob_final


def update_state_recombinate(
    voltage: float,
    E_field: float,
    oxygen_config: dict,
    sim_ctes: SimulationConstants,
    params: SimulationParameters,
    actual_state: np.ndarray,
    oxygen_state: np.ndarray,
    temperatura: float | np.ndarray,
    velocity_thresholds: dict,  # Diccionario pasado como argumento
) -> tuple[np.ndarray, np.ndarray]:
    """
    Orquesta el proceso completo de RESET en un paso de tiempo:
    1. Generación de iones oxígenos por voltaje.
    2. Movimiento de iones.
    3. Recombinación.
    """

    # 1. Se generan oxígenos según el voltaje
    for threshold_voltage, max_oxigenos in oxygen_config.items():
        if abs(voltage) > abs(threshold_voltage):
            oxygen_state = Recombination.generate_oxygen(oxygen_state, max_oxigenos)
            break  # Solo usar el umbral más alto superado

    # 2. Movemos los ionesx
    oxygen_state, velocidad = Recombination.move_oxygen_ions(
        paso_temp=params.paso_temporal,
        oxygen_state=oxygen_state,
        temperature=temperatura,
        E_field=E_field,
        grid_size=params.atom_size,
        vibration_frequency=sim_ctes.vibration_frequency,
        gamma_drift=sim_ctes.gamma_drift,
        migration_energy=sim_ctes.E_m,
        cte_red=sim_ctes.cte_red,
        voltage=voltage,
        velocity_thresholds=velocity_thresholds,
    )

    # print(f"Velocidad de los iones oxígeno en este paso: {velocidad:.4e} m/s")

    # 3.Recombinación de iones con vacantes
    actual_state_update, oxygen_state_update = Recombination.Recombine_opt(
        vacancy_state=actual_state,
        oxygen_state=oxygen_state,
        paso_temp=params.paso_temporal,
        velocidad=velocidad,
        temperatura=temperatura,
        vibration_frequency=sim_ctes.vibration_frequency,
        recom_enchancement_factor=sim_ctes.recom_enchancement_factor,
        recombination_energy=sim_ctes.recombination_energy,
        long_decaimiento_concentracion=sim_ctes.long_decaimiento_concentracion,
    )

    return actual_state_update, oxygen_state_update
