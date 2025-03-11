import numpy as np
import math
import sys

from RRAM import Constants as cte
from .Constants import k_b_ev


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


def Generate_Oxigen(oxygen_state: np.array, num_oxygen: int):
    """
    Generates random oxygen positions in the given oxygen state matrix.

    Args:
        oxigen_state (np.array): The current oxygen state matrix.
        num_oxigen (int): The number of oxygen to generate.

    Returns:
        np.array: The updated oxygen state matrix with the generated oxygen positions.
    """

    eje_y = oxygen_state.shape[1]
    y = np.zeros(num_oxygen, dtype=int)

    for i in range(num_oxygen):
        # Genero las coordenadas aleatorias para el eje y donde habrá un oxígeno
        y[i] = np.random.randint(0, eje_y)

    # Itero sobre cada par coordenada para asignar el valor de 1 que representa que se generó un oxígeno en esa posición
    for i in range(num_oxygen):
        random_number = np.random.rand()
        if (oxygen_state[y[i], 0] == 0 and random_number < 0.5):
            oxygen_state[y[i], 0] = 1

    # Devuelvo la matriz con los oxígenos generados
    return oxygen_state


def Move_OxygenIons(paso_temp: float, oxygen_state: np.array, temperature: float, E_field: float, grid_size: float, **kwargs):
    """
    Move the oxygen ions in the simulation based on the given parameters.

    Parameters:
        - simu_time (float): The simulation time.
        - oxygen_state (np.array): The matrix representing the state of oxygen ions.
        - temperature (float): The temperature of the system.
        - E_field (float): The electric field strength.
        - atom_size (float): The size of each atom.

    Returns:
    np.array: The updated matrix representing the state of oxygen ions after the movement.
    """

    # Obtengo los valores de las constantes si las estoy pasando como argumentos
    if kwargs:
        # Obtengo el valor de las constantes que necesita la función
        t_0 = float(kwargs.get('vibration_frequency'))
        gamma_drift = float(kwargs.get('drift_coefficient'))
        E_m = float(kwargs.get('migration_energy'))
        cte_red = float(kwargs.get('cte_red'))
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
        senoh = math.sinh((cte_red * E_field * gamma_drift) / (2 * k_b_ev * temperature))
        exp_velocity = math.exp(-E_m / (k_b_ev * temperature))
    except OverflowError as Overflow_exception:
        print("\n Error en el cálculo de la velocidad de los iones de oxígeno, los valores empleados son:")
        print(f"OverflowError: {Overflow_exception}")
        print(f"cte_red: {cte_red}")
        print(f"E_field: {E_field}")
        print(f"gamma_drift: {gamma_drift}")
        print(f"k_b_ev: {k_b_ev}")
        print(f"temperature: {temperature}")
        print(f"E_m: {E_m}")
        sys.exit(1)  # Termina la ejecución del programa con un código de salida 1

    # Esto es un arreglo temporal para dar cuenta que hay un tiempo hasta que los iones de oxígeno se muevan
    if abs(E_field*(10e-9)) > 0.45:
        # En la expresión original se multiplica por 2 lo he quitado para ver si sale algo mejor
        # print(f"EL valor del potencial es: {E_field*(10e-9)}")
        oxigen_velocity = 3e-07  # 2 * t_0 * cte_red * (senoh * exp_velocity)
    else:
        oxigen_velocity = 0  # para que no se mueva hasta q se alcance un potencial concreto

    # Calculo la cantidad de "casillas" que se moverá el ion de oxígeno
    displacement = int(round((oxigen_velocity * paso_temp) / grid_size))
    if displacement > 3:
        displacement = 3

    if displacement == 0:
        pass
        # print("No se mueve")
    else:
        # Recorro la matriz de oxígeno para mover los iones
        for i in range(oxygen_state_before.shape[0]):              # Recorro las filas
            for j in range(oxygen_state_before.shape[1]):
                if oxygen_state_before[j, i] == 1:
                    # Muevo el oxígeno si queda dentro de la matriz
                    if i + displacement < oxygen_state_before.shape[1]:
                        oxygen_state[j, i + displacement] = 1
                        oxygen_state[j, i] = 0
                    else:  # Si se sale de la matriz, lo elimino
                        oxygen_state[j, i] = 0

    return oxygen_state, oxigen_velocity, displacement


def Recombine(actual_state: np.array, oxygen_state: np.array, paso_temp: float, velocidad: float, temp: float, **kwargs) -> np.array:
    """
    Recombines oxygen and actual states based on certain conditions.

    Parameters:
    oxygen_state (np.array): The matrix representing the oxygen state.
    actual_state (np.array): The matrix representing the actual state.

    Returns:
    np.array: The updated actual state after recombination.
    """

    # Duplico la matriz de oxígeno y configuraciones para no modificar la original
    oxygen_state_before = np.copy(oxygen_state)
    actual_state_before = np.copy(actual_state)

    # Calculo la probabilidad de recombinación.
    prob_recom = Prob_Recombination(paso_temp, velocidad, temp, **kwargs)

    # Recorro la matriz de oxígeno para saber en qué posiciones hay oxígeno
    for i in range(oxygen_state_before.shape[0]):
        for j in range(oxygen_state_before.shape[1]):
            # Si hay oxígeno en la posición de la matriz de oxígeno y hay una vacante en la matriz de estado actual
            if oxygen_state_before[i, j] == 1 and actual_state_before[i, j] == 1:
                # Si hay un hueco, calculo la probabilidad de recombinación
                random_number = np.random.rand()    # Genero un número aleatorio
                if random_number < prob_recom:
                    actual_state[i, j] = 0
                    oxygen_state[i, j] = 0

    # # Otra forma de hacerlo es con máscaras dado por copilot
    # # Crear una máscara para las posiciones donde hay oxígeno y una vacante
    # mask = (oxygen_state_before == 1) & (actual_state_before == 1)

    # # Generar una matriz de números aleatorios del mismo tamaño que la máscara
    # random_numbers = np.random.rand(*mask.shape)

    # # Crear una máscara para las posiciones donde ocurre la recombinación
    # recombination_mask = mask & (random_numbers < prob_recom)

    # # Actualizar los estados según la máscara de recombinación
    # actual_state[recombination_mask] = 0
    # oxygen_state[recombination_mask] = 0

    return (actual_state, oxygen_state, prob_recom)


def Prob_Recombination(paso_temporal: float, velocidad: float, temp: float, **kwargs) -> float:

    # Obtengo las constantes necesarias para el cálculo
    if kwargs:
        # Obtengo el valor de las constantes que necesita la función
        t_0 = float(kwargs.get('vibration_frequency'))
        beta_0 = float(kwargs.get('recom_enchancement_factor'))
        E_a = float(kwargs.get('activation_energy'))
        L_p = float(kwargs.get('decaimiento_concentracion'))
        # cte_red = float(kwargs.get('cte_red'))
    else:
        t_0 = cte.t_0
        beta_0 = cte.beta_0
        E_a = cte.E_a
        L_p = cte.L_p
        # cte_red = cte.cte_red

    # Calculo la  probabilidad de recombinación en equilibrio
    prob_in_equilibrio = (paso_temporal * t_0) * (math.exp(-E_a / (k_b_ev * temp)))
    exp_beta = math.exp(-(paso_temporal * velocidad) / L_p) * beta_0

    prob_recom = prob_in_equilibrio * exp_beta

    return prob_recom


# |----------------------------------------------------------------------------------------------------------------------|
# | Esta función es la original del paper: On the Switching Parameter Variation of Metal-Oxide RRAM—Part I:              |
# | Physical Modeling and Simulation Methodologyque no tengo forma de conseguir los resultados que indica                |
# | el paper.                                                                                                            |
# |----------------------------------------------------------------------------------------------------------------------|


# def Simple_recombination(paso_temp: float, pos_x: int, E_field: float, temperature: float, grid_size: float, factor: float):
#     """
#     Calculates the recombination probability Function that calculates the probability of recombination.
#     It is calculated from the equilibrium probability and a factor that contains the ion velocity.

#     Args:
#         simu_time (float): The simulation time.
#         pos_x (int): The position in the x-axis.
#         E_field (float): The electric field.
#         temperature (float): The temperature.
#         grid_size (float): The grid size.
#         factor (float): A factor used in the calculation.

#     Returns:
#         tuple: A tuple containing the recombination probability, oxygen ion velocity, and diffusion function.
#     """

#     Prob_in_equilibrio = (paso_temp * t_0)*(math.exp(-E_a / (k_b_ev * temperature)))

#     senoh = math.sinh((2*elementary_charge * E_field * gamma_drift) / (k_b_ev * temperature))
#     exponencial_velocidad = math.exp(-E_m / (k_b_ev * temperature))

#     Oxigen_Ion_velocity = factor * t_0 * grid_size * (senoh * exponencial_velocidad)

#     Funcion = DifussiveBehaviour(pos_x, Oxigen_Ion_velocity, paso_temp, grid_size)
#     exponencial = math.exp(-(paso_temp * Oxigen_Ion_velocity) / L_p)
#     beta = (beta_0 * exponencial * Funcion)

#     Prob_recom = Prob_in_equilibrio * beta

#     return Prob_recom, Oxigen_Ion_velocity*paso_temp, Funcion
