import numpy as np
from math import pi


def OmhCurrent(DDP, State):
    # Initialize total resistance
    total_resistance = 0

    # SObre cada columna de la matriz
    for column in State.T:
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


def poole_frenkel(V, T, eps, N, E):
    """
    Función para calcular la densidad de corriente utilizando el efecto Poole-Frenkel
    V: voltaje aplicado
    T: temperatura absoluta
    eps: permitividad
    N: densidad de trampas
    E: Campo eléctrico
    phi_b: energía de barrera
    """
    k = 1.38e-23  # Constante de Boltzmann
    e = 1.6e-19  # Carga del electrón
    phi_b = 123  # Energía de barrera
    exponencial = np.exp(-2*e*(phi_b - math.sqtr((2*e*E)/pi*eps)) / (k * T))
    J = E*exponencial
    return J
