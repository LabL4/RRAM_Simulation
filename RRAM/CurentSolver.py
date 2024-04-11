import numpy as np

from scipy.constants import elementary_charge, Boltzmann


def OmhCurrent(DDP: float, Configuration_state, resistance: float = 1) -> float:
    # Initialize total resistance
    total_resistance = 0

    # SObre cada columna de la matriz
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


def poole_frenkel(temperature: float, electric_field: np.ndarray,
                  barrera: float = 0.895, beta: float = 2.8E-5, I_0: float = 1e-12) -> np.ndarray:

    k_b_ev = Boltzmann / elementary_charge

    exponencial = np.exp((beta * np.sqrt(electric_field) - barrera) / (k_b_ev * temperature))

    I_poole_frenkel = I_0 * electric_field * exponencial

    # return I_poole_frenkel
