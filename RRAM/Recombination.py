import math
import numpy as np

from RRAM import Constants as cte
from scipy.constants import elementary_charge
from .Constants import t_0, k_b_ev, E_m, gamma_drift


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


def GenerateOxigen(oxygen_state: np.array, num_oxygen: int):
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
        if oxygen_state[y[i], 0] == 0:
            oxygen_state[y[i], 0] = 1

    # Devuelvo la matriz con los oxígenos generados
    return oxygen_state


def Move_OxygenIons(simu_time: float, oxygen_state: np.array, temperature: float, E_field: float, atom_size: float, factor: float, **kwargs):
    """
    Move the oxygen ions in the simulation based on the given parameters.

    Parameters:
        - simu_time (float): The simulation time.
        - oxygen_state (np.array): The matrix representing the state of oxygen ions.
        - temperature (float): The temperature of the system.
        - E_field (float): The electric field strength.
        - atom_size (float): The size of each atom.
        - factor (float): A factor used to adjust the velocity.

    Returns:
    np.array: The updated matrix representing the state of oxygen ions after the movement.
    """

    # Obtengo los valores de las constantes si las estoy pasando como argumentos
    if kwargs:
        # Obtengo el valor de las constantes que necesita la función
        t_0 = kwargs.get('t_0', cte.t_0)
        gamma_drift = kwargs.get('gamma_drift', cte.gamma_drift)
        E_m = kwargs.get('E_m', cte.E_m)
    else:
        t_0 = cte.t_0
        gamma_drift = cte.gamma_drift
        E_m = cte.E_m

    # Obtengo la velocidad de los iones de oxígeno v = (a/t0)*exp(−Em/kT) sinh(q * γ_drift * F/kT)
    senoh = math.sinh((2*elementary_charge * E_field * gamma_drift) / (k_b_ev * temperature))
    exp_velocity = math.exp(-E_m / (k_b_ev * temperature))

    # el t_0 es el valor de 1/t_0 que lo pongo directamente y "factor" es algo que introduzco a mano para ajustar la velocidad
    oxigen_velocity = factor * t_0 * atom_size * (senoh * exp_velocity)

    # Calculo la cantidad de "casillas" que se moverá el ion de oxígeno
    displacement = int(round(oxigen_velocity * simu_time / atom_size))

    if displacement == 0:
        pass
        # print("No se mueve")
    else:
        # Recorro la matriz de oxígeno para mover los iones
        for i in range(oxygen_state.shape[1]):
            for j in range(oxygen_state.shape[0]):
                if oxygen_state[j, i] == 1:
                    # Muevo el oxígeno
                    if i + displacement < oxygen_state.shape[1]:
                        oxygen_state[j, i + displacement] = 1
                        oxygen_state[j, i] = 0
                    else:  # Si se sale de la matriz, lo coloco en la última posición
                        # oxygen_state[j, oxygen_state.shape[1] - 1] = 1
                        oxygen_state[j, i] = 0

    return oxygen_state, oxigen_velocity, displacement


def Recombine(actual_state: np.array, oxygen_state: np.array):
    """
    Recombines oxygen and actual states based on certain conditions.

    Parameters:
    oxygen_state (np.array): The matrix representing the oxygen state.
    actual_state (np.array): The matrix representing the actual state.

    Returns:
    np.array: The updated actual state after recombination.
    """

    # Recorro la matriz de oxígeno para saber en qué posiciones hay oxígeno
    for i in range(oxygen_state.shape[0]):
        for j in range(oxygen_state.shape[1]):
            # Si hay oxígeno en la posición de la matriz de oxígeno y hay un hueco en la matriz de estado actual
            if oxygen_state[i, j] == 1 and actual_state[i, j] == 1:
                # Si hay un hueco, calculo la probabilidad de recombinación
                prob_recom = np.random.rand()
                # Si la probabilidad es menor a 0.5, recombinan:
                if prob_recom < 0.5:  # Cambiar luego a la probabilidad en equilibrio que menciona en el paper original
                    actual_state[i, j] = 0
                    oxygen_state[i, j] = 0

    return (actual_state, oxygen_state)

# |----------------------------------------------------------------------------------------------------------------------|
# | Esta función es la original del paper: On the Switching Parameter Variation of Metal-Oxide RRAM—Part I:              |
# | Physical Modeling and Simulation Methodologyque no tengo forma de conseguir los resultados que indica                |
# | el paper.                                                                                                            |
# |----------------------------------------------------------------------------------------------------------------------|

# def Simple_recombination(simu_time: float, pos_x: int, E_field: float, temperature: float, grid_size: float, factor: float):
#     """
#     Calculates the recombination probability Function that calculates the probability of recombination.
#     It is calculated from the equilibrium probability and a factor that contains the ion velocity.
#
#     Args:
#         simu_time (float): The simulation time.
#         pos_x (int): The position in the x-axis.
#         E_field (float): The electric field.
#         temperature (float): The temperature.
#         grid_size (float): The grid size.
#         factor (float): A factor used in the calculation.
#
#     Returns:
#         tuple: A tuple containing the recombination probability, oxygen ion velocity, and diffusion function.
#     """
#
#     Prob_in_equilibrio = (simu_time * t_0)*(math.exp(-E_a / (k_b_ev * temperature)))
#
#     senoh = math.sinh((2*elementary_charge * E_field * gamma_drift) / (k_b_ev * temperature))
#     exponencial_velocidad = math.exp(-E_m / (k_b_ev * temperature))
#
#     Oxigen_Ion_velocity = factor * t_0 * grid_size * (senoh * exponencial_velocidad)
#
#     Funcion = DifussiveBehaviour(pos_x, Oxigen_Ion_velocity, simu_time, grid_size)
#     exponencial = math.exp(-(simu_time * Oxigen_Ion_velocity) / L_p)
#     beta = (beta_0 * exponencial * Funcion)
#
#     Prob_recom = Prob_in_equilibrio * beta
#
#     return Prob_recom, Oxigen_Ion_velocity*simu_time, Funcion
