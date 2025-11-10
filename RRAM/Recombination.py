from RRAM import Constants as cte
from .Constants import k_b_ev

import numpy as np
import math
import sys


def Init_OxygenState(espesor_dispositivo: float, atom_size: float):
    """
    Initializes the state of oxygen atoms in a device.

    Parameters:
    - espesor_dispositivo (float): The thickness of the device.
    - atom_size (float): The size of each atom, is equal to the mesh size of the simulation.

    Returns:
    - InitialOxigenState (numpy.ndarray): The initial state of oxygen atoms in the device.
    """

    eje_x = round(espesor_dispositivo / atom_size)
    eje_y = round(espesor_dispositivo / atom_size)

    InitialOxygenState = np.zeros((eje_x, eje_y), dtype=int)

    return InitialOxygenState


def Generate_Oxygen(oxygen_state: np.ndarray, num_oxygen: int):
    """
    Generates random oxygen positions in the given oxygen state matrix.

    Args:
        oxigen_state (np.ndarray): The current oxygen state matrix.
        num_oxigen (int): The number of oxygen to generate.

    Returns:
        np.ndarray: The updated oxygen state matrix with the generated oxygen positions.
    """

    eje_y = oxygen_state.shape[1]
    y = np.zeros(num_oxygen, dtype=int)

    for i in range(num_oxygen):
        # Genero las coordenadas aleatorias para el eje y donde habrá un oxígeno
        y[i] = np.random.randint(0, eje_y)

    # Itero sobre cada par coordenada para asignar el valor de 1 que representa que se generó un oxígeno en esa posición
    for i in range(num_oxygen):
        random_number = np.random.rand()
        if oxygen_state[y[i], 0] == 0 and random_number < 0.35:
            oxygen_state[y[i], 0] = 1

    # Devuelvo la matriz con los oxígenos generados
    return oxygen_state


def Move_OxygenIons(
    paso_temp: float,
    oxygen_state: np.ndarray,
    temperature: float,
    E_field: float,
    grid_size: float,
    **kwargs,
):  # type: ignore
    """
    Move the oxygen ions in the simulation based on the given parameters.

    Parameters:
        - simu_time (float): The simulation time.
        - oxygen_state (np.ndarray): The matrix representing the state of oxygen ions.
        - temperature (float): The temperature of the system.
        - E_field (float): The electric field strength.
        - atom_size (float): The size of each atom.

    Returns:
    np.ndarray: The updated matrix representing the state of oxygen ions after the movement.
    """

    # Obtengo los valores de las constantes si las estoy pasando como argumentos
    if kwargs:
        # Obtengo el valor de las constantes que necesita la función
        t_0 = float(kwargs.get("vibration_frequency"))  # type: ignore
        gamma_drift = float(kwargs.get("drift_coefficient"))  # type: ignore
        E_m = float(kwargs.get("migration_energy"))  # type: ignore
        cte_red = float(kwargs.get("cte_red"))  # type: ignore
    else:
        t_0 = cte.t_0
        gamma_drift = cte.gamma_drift
        E_m = cte.E_m
        cte_red = cte.cte_red

    # Duplico la matriz de oxígeno para no modificar la original
    oxygen_state_before = np.copy(oxygen_state)

    # Obtengo la velocidad de los iones de oxígeno v = ((2 * a)/t0)*exp(−Em/kT) sinh((d * γ_drift * F)/2kT)
    try:
        # Obtengo la velocidad de los iones de oxígeno v = ((2 * a)/t0)*exp(−Em/kT) sinh((d * γ_drift * F)/2kT)
        senoh = np.sinh((cte_red * E_field * gamma_drift) / (2 * k_b_ev * temperature))
        exp_velocity = np.exp(-E_m / (k_b_ev * temperature))
    except OverflowError as Overflow_exception:
        print(
            "\n Error en el cálculo de la velocidad de los iones de oxígeno, los valores empleados son:"
        )
        print(f"OverflowError: {Overflow_exception}")
        print(f"cte_red: {cte_red}")
        print(f"E_field: {E_field}")
        print(f"gamma_drift: {gamma_drift}")
        print(f"k_b_ev: {k_b_ev}")
        print(f"temperature: {temperature}")
        print(f"E_m: {E_m}")
        sys.exit(1)  # Termina la ejecución del programa con un código de salida 1

    # Esto es un arreglo temporal para dar cuenta que hay un tiempo hasta que los iones de oxígeno se muevan
    # En la expresión original se multiplica por 2 lo he quitado para ver si sale algo mejor
    abs_field = abs(E_field * 10e-9)
    thresholds = [
        (0.7, 5.2e-07),
        (0.5, 3e-07),
    ]
    oxigen_velocity = 0
    for limit, vel in thresholds:
        if abs_field > limit:
            oxigen_velocity = vel
            break

    # Calculo la cantidad de "casillas" que se moverá el ion de oxígeno
    displacement = int(round((oxigen_velocity * paso_temp) / grid_size))
    if displacement > 3:
        displacement = 3

    if displacement == 0:
        pass  # print("No se mueve")
    else:
        # Recorro la matriz de oxígeno para mover los iones
        for i in range(oxygen_state_before.shape[0]):  # Recorro las filas
            for j in range(oxygen_state_before.shape[1]):
                if oxygen_state_before[j, i] == 1:
                    # Muevo el oxígeno si queda dentro de la matriz
                    if i + displacement < oxygen_state_before.shape[1]:
                        oxygen_state[j, i + displacement] = 1
                        oxygen_state[j, i] = 0
                    else:  # Si se sale de la matriz, lo elimino
                        oxygen_state[j, i] = 0

    return oxygen_state, oxigen_velocity, displacement


def Recombine(
    actual_state: np.ndarray,
    oxygen_state: np.ndarray,
    paso_temp: float,
    velocidad: float,
    temp: float,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Recombines oxygen and actual states based on certain conditions.

    Parameters:
    oxygen_state (np.ndarray): The matrix representing the oxygen state.
    actual_state (np.ndarray): The matrix representing the actual state.

    Returns:
    np.ndarray: The updated actual state after recombination.
    """

    # Duplico la matriz de oxígeno y configuraciones para no modificar la original
    oxygen_state_before = np.copy(oxygen_state)
    actual_state_before = np.copy(actual_state)

    # Calculo la probabilidad de recombinación.
    prob_recom = min(
        Prob_Recombination(paso_temp, velocidad, temp, **kwargs), 1.0
    )  # Para que no se pase de 1.0

    # Recorro la matriz de oxígeno para saber en qué posiciones hay oxígeno
    for i in range(oxygen_state_before.shape[0]):
        for j in range(oxygen_state_before.shape[1]):
            # Si hay oxígeno en la posición de la matriz de oxígeno y hay una vacante en la matriz de estado actual
            if oxygen_state_before[i, j] == 1 and actual_state_before[i, j] == 1:
                # Si hay un hueco, calculo la probabilidad de recombinación
                random_number = np.random.rand()  # Genero un número aleatorio
                if random_number < prob_recom:
                    actual_state[i, j] = 0
                    oxygen_state[i, j] = 0

    return (actual_state, oxygen_state, prob_recom)


def Prob_Recombination(
    paso_temporal: float, velocidad: float, temp: float, **kwargs
) -> float:
    # Obtengo las constantes necesarias para el cálculo
    if kwargs:
        # Obtengo el valor de las constantes que necesita la función
        t_0 = float(kwargs.get("vibration_frequency"))  # type: ignore
        beta_0 = float(kwargs.get("recom_enchancement_factor"))  # type: ignore
        E_r = float(kwargs.get("recombination_energy"))  # type: ignore
        L_p = float(kwargs.get("decaimiento_concentracion"))  # type: ignore
    else:
        t_0 = cte.t_0
        beta_0 = cte.beta_0
        E_r = cte.E_r
        L_p = cte.L_p

    # Calculo la probabilidad de recombinación en equilibrio
    prob_in_equilibrio = (paso_temporal * t_0) * (math.exp(-E_r / (k_b_ev * temp)))
    exp_beta = math.exp(-(paso_temporal * velocidad) / L_p) * beta_0
    prob_recom = prob_in_equilibrio * exp_beta

    return prob_recom


def Recombine_opt(
    vacancy_state: np.ndarray,
    oxygen_state: np.ndarray,
    paso_temp: float,
    velocidad: float,
    temp: float,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Función optimizada para recombinar oxígeno con vacantes en la simulación,
    mediante operaciones vectorizadas que eliminan bucles explícitos.

    Args:
        actual_state (np.ndarray): Matriz de estado actual con vacantes (1).
        oxygen_state (np.ndarray): Matriz de iones oxígeno (1 donde hay oxígeno).
        paso_temp (float): Paso temporal de simulación.
        velocidad (float): Velocidad promedio de movimiento de iones de oxígeno.
        temp (float): Temperatura actual.
        **kwargs: Constantes físicas necesarias para cálculo.

    Returns:
        Tuple conteniendo:
        - actual_state actualizado (np.ndarray).
        - oxygen_state actualizado (np.ndarray).
        - probabilidad de recombinación calculada (float).
    """

    state_updated = np.copy(vacancy_state)
    oxygen_updated = np.copy(oxygen_state)

    # Posiciones donde hay oxígeno y una vacante
    mask_recomb = (oxygen_state == 1) & (vacancy_state == 1)

    # Matriz de números aleatorios del mismo tamaño que la máscara
    random_values = np.random.rand(*oxygen_state.shape)

    # Calculo la probabilidad de recombinación.
    prob_recom = min(Prob_Recombination(paso_temp, velocidad, temp, **kwargs), 1.0)

    recombination_mask = mask_recomb & (random_values < prob_recom)

    state_updated[recombination_mask] = 0
    oxygen_updated[recombination_mask] = 0

    return (state_updated, oxygen_updated)


def update_oxygen_state(
    paso_temp: float,
    oxygen_state: np.ndarray,
    temperature: float,
    E_field: float,
    grid_size: float,
    **kwargs,
):  # type: ignore
    """
    Move the oxygen ions in the simulation based on the given parameters.
    Parameters:
        - simu_time (float): The simulation time.
        - oxygen_state (np.ndarray): The matrix representing the state of oxygen ions.
        - temperature (float): The temperature of the system.
        - E_field (float): The electric field strength.
        - atom_size (float): The size of each atom.
    Returns:
    np.ndarray: The updated matrix representing the state of oxygen ions after the movement.
    """

    # Obtengo los valores de las constantes si las estoy pasando como argumentos
    if kwargs:
        # Obtengo el valor de las constantes que necesita la función
        t_0 = float(kwargs.get("vibration_frequency"))  # type: ignore
        gamma_drift = float(kwargs.get("drift_coefficient"))  # type: ignore
        E_m = float(kwargs.get("migration_energy"))  # type: ignore
        cte_red = float(kwargs.get("cte_red"))  # type: ignore
    else:
        t_0 = cte.t_0
        gamma_drift = cte.gamma_drift
        E_m = cte.E_m
        cte_red = cte.cte_red

    # Esto es un arreglo temporal para dar cuenta que hay un tiempo hasta que los iones de oxígeno se muevan. En la expresión original se multiplica por 2 lo he quitado para ver si sale algo mejor
    # Calcular velocidad de iones con manejo robusto usando numpy
    # try:
    #     senoh = np.sinh(
    #         (cte_red * E_field * gamma_drift) / (2 * 8.617333262145e-5 * temperature)
    #     )
    #     exp_velocity = np.exp(-E_m / (8.617333262145e-5 * temperature))
    # except Exception as e:
    #     print(f"Error en cálculo de velocidad de iones: {e}")
    #     sys.exit(1)

    # E_field_abs = abs(E_field * 10e-9)
    # velocity_map = [
    #     (0.7, 5.1e-07),
    #     (0.5, 2.6e-07),
    # ]

    # oxigen_velocity = 0
    # for threshold, velocity in velocity_map:
    #     if E_field_abs > threshold:
    #         oxigen_velocity = velocity
    #         break

    # Esto es un arreglo temporal para dar cuenta que hay un tiempo hasta que los iones de oxígeno se muevan
    if abs(E_field * (10e-9)) > 0.5:
        # En la expresión original se multiplica por 2 lo he quitado para ver si sale algo mejor
        oxigen_velocity = 3e-07  # 2 * t_0 * cte_red * (senoh * exp_velocity)
    elif abs(E_field * (10e-9)) > 0.7:
        oxigen_velocity = 5.2e-07  # 2 * t_0 * cte_red * (senoh * exp_velocity)
    else:
        oxigen_velocity = (
            0  # para que no se mueva hasta q se alcance un potencial concreto
        )

    # Calculo la cantidad de "casillas" que se moverá el ion de oxígeno
    displacement = int(round((oxigen_velocity * paso_temp) / grid_size))

    # Generar nueva matriz vacía para estado actualizado
    oxygen_state_new = np.zeros_like(oxygen_state)

    if displacement > 0:
        oxygen_state_new[:, displacement:] = oxygen_state[:, :-displacement]
    else:
        # Sin desplazamiento, copiar directamente
        oxygen_state_new = oxygen_state.copy()

    return oxygen_state_new, oxigen_velocity
