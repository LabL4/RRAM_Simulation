import numpy as np

from scipy.constants import elementary_charge, Boltzmann


def OmhCurrent(DDP: float, Configuration_state, resistance: float = 1) -> float:
    """
    Calculates the Ohmic current based on the given parameters.

    Parameters:
    - DDP (float): The voltage difference across the circuit.
    - Configuration_state: The configuration state of the circuit.
    - resistance (float): The default resistance value (optional, default is 1).

    Returns:
    - float: The calculated Ohmic current.

    """
    # TODO ARREGLAR LA CORRIENTE ÓHMICA
    # Initialize total resistance
    total_resistance = 0
    # Sobre cada columna de la matriz
    for column in Configuration_state.T:
        # Se calcula la resistencia paralela de los elementos de la columna
        parallel_resistance = 0
        for resistance in column:
            if resistance != 0:  # ignoramos circuitos abiertos
                parallel_resistance += 1 / resistance
        if parallel_resistance != 0:  # Se evita dividir por 0
            parallel_resistance = 1 / parallel_resistance

        # Calclamos la resistencia total como suma de las resistencias en serie
        total_resistance += parallel_resistance

    return DDP/total_resistance


def poole_frenkel(temperature: float, electric_field: float,
                  barrera: float = 0.895, beta: float = 2.8E-5, I_0: float = 1e-12) -> float:
    """
    Calculates the current using the Poole-Frenkel equation.

    Args:
        temperature (float): The temperature in Kelvin.
        electric_field (float): The electric field strength in V/m.
        barrera (float, optional): The barrier height in eV. Defaults to 0.895.
        beta (float, optional): The Poole-Frenkel constant. Defaults to 2.8E-5.
        I_0 (float, optional): The saturation current in A. Defaults to 1e-12.

    Returns:
        float: The calculated current in A.
    """

    k_b_ev = Boltzmann / elementary_charge

    exponencial = np.exp((beta * np.sqrt(electric_field) - barrera) / (k_b_ev * temperature))

    I_poole_frenkel = I_0 * electric_field * exponencial

    return I_poole_frenkel
