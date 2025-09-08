import numpy as np


def obtener_puntos_en_curva(v_array, i_array, puntos_x):
    """
    Devuelve un diccionario con las coordenadas (x, y) de los puntos
    más cercanos en la curva definida por v_array e i_array.

    v_array: array de voltajes de la curva
    i_array: array de corrientes de la curva
    puntos_x: diccionario {'label': x}

    Retorna: diccionario {'label': (x, y)}
    """
    puntos = {}
    for label, xp in puntos_x.items():
        idx = np.argmin(np.abs(v_array - xp))  # índice del x más cercano
        yp = i_array[idx]  # valor de y correspondiente
        puntos[label] = (xp, yp)
    return puntos
