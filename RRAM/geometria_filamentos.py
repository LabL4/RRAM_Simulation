"""Geometría de filamentos y utilidades de curvas."""

from typing import List
from typing import Optional
import numpy as np


def acumular_T_max_filamentos(acumulado: List[Optional[float]], T_fils: List[float]) -> List[Optional[float]]:
    """
    Acumula el máximo histórico de temperatura de cada filamento a lo largo de una fase.

    Se llama SOLO desde las ramas que resuelven el mapa térmico FVM, es decir,
    cuando `T_fils` es la temperatura real de cada filamento extraída de su fila
    central. Los pasos sin filamento (`T_fils` = temperatura de Joule del
    dispositivo replicada N veces) no cuentan: no son la temperatura de ningún
    filamento porque el filamento aún no existe.

    Un `None` en el acumulado significa que esa fase nunca llegó a resolver el
    FVM para ese filamento.
    """
    return [nuevo if maximo is None else max(maximo, nuevo) for maximo, nuevo in zip(acumulado, T_fils)]


def generar_configuracion_filamentos(
    n_filas: int,
    n_cols: int,
    num_filamentos: int,
    peso_central: int = 70,
    grosores_filamento: list | int | None = None,
    centros_override: list[int] | None = None,
) -> tuple:
    """
    Calcula los rangos de cada filamento y define las regiones de peso iniciales.

    Argumentos:
        n_filas (int): Número de filas de la matriz (alto del dispositivo).
        n_cols (int): Número de columnas (distancia entre electrodos).
        num_filamentos (int): Número de filamentos a distribuir.
        peso_central (int): Probabilidad asignada a la zona central.
        grosores_filamento (list | int | None): Grosor en celdas para cada filamento.
            Si es lista, un valor por filamento. Si es int, se aplica a todos.
            Si es None, usa el valor fijo heredado de 2.
        centros_override (list[int] | None): Filas centrales explícitas por filamento.
            Si es None, lista vacía o no-lista, usa midpoint automático.
    """
    # Normalizar grosores_filamento a lista de longitud num_filamentos
    if grosores_filamento is None:
        grosores = [2] * num_filamentos
    elif isinstance(grosores_filamento, (int, np.integer)):
        grosores = [int(grosores_filamento)] * num_filamentos
    else:
        grosores_raw = list(grosores_filamento)
        if len(grosores_raw) < num_filamentos:
            grosores_raw += [grosores_raw[-1]] * (num_filamentos - len(grosores_raw))
        grosores = [int(g) for g in grosores_raw[:num_filamentos]]

    usar_override = isinstance(centros_override, list) and len(centros_override) > 0

    filamentos_ranges = []
    regiones_pesos = []
    filas_centrales = []

    ancho_zona = n_filas // num_filamentos

    for n in range(num_filamentos):
        inicio_rango = n * ancho_zona
        fin_rango = (n + 1) * ancho_zona - 1 if n < num_filamentos - 1 else n_filas - 1
        filamentos_ranges.append((inicio_rango, fin_rango))

        if usar_override:
            fila_central = int(centros_override[n])
            if not (inicio_rango <= fila_central <= fin_rango):
                raise ValueError(f"centros_override[{n}]={fila_central} fuera de rango [{inicio_rango}, {fin_rango}]")
        else:
            fila_central = (inicio_rango + fin_rango) // 2
        filas_centrales.append(fila_central)

        grosor = grosores[n]
        x_start = max(0, fila_central - grosor)
        x_end = min(n_filas, fila_central + grosor + 1)

        region = (x_start, x_end, 0, n_cols)
        regiones_pesos.append((region, peso_central))

    return filamentos_ranges, regiones_pesos, filas_centrales


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
