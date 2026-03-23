import pandas as pd
import pickle
import math
import sys
import os

ruta_raiz = os.getcwd() + "/"
sys.path.append(ruta_raiz)

from RRAM import Generation as gn

import numpy as np
import math


def generar_configuracion_filamentos(eje_x, eje_y, num_filamentos, peso_central=70):
    """
    Calcula los rangos de cada filamento y define las regiones de peso iniciales.

    Argumentos:
        eje_x (int): Número de celdas en el eje X (filas).
        eje_y (int): Número de celdas en el eje Y (columnas).
        num_filamentos (int): Número de filamentos a distribuir.
        peso_central (int): Probabilidad asignada a la zona central (por defecto 70).

    Retorna:
        tuple: (filamentos_ranges, regiones_pesos)
    """
    filamentos_ranges = []
    regiones_pesos = []

    # Calculamos el ancho de cada zona dividiendo el espacio total entre el número de filamentos
    ancho_zona = eje_x // num_filamentos

    for n in range(num_filamentos):
        # 1. Calcular el rango (zona) de este filamento
        inicio_rango = n * ancho_zona
        # El último filamento se extiende hasta el final para cubrir el resto por división entera
        fin_rango = (n + 1) * ancho_zona - 1 if n < num_filamentos - 1 else eje_x - 1
        filamentos_ranges.append((inicio_rango, fin_rango))

        # 2. Identificar la fila central de dicha zona
        fila_central = (inicio_rango + fin_rango) // 2

        # 3. Definir la región de alta probabilidad (Fila central + 2 arriba + 2 abajo)
        # x_start: fila_central - 2
        # x_end: fila_central + 3 (se usa +3 porque en el slicing el límite superior es exclusivo)
        x_start = max(0, fila_central - 2)
        x_end = min(eje_x, fila_central + 3)

        # La región cubre todo el ancho del dispositivo (de 0 a eje_y)
        region = (x_start, x_end, 0, eje_y)
        regiones_pesos.append((region, peso_central))

    return filamentos_ranges, regiones_pesos


# Ejemplo de integración en tu bucle de Init_simulation.py:
# for i, row in df_params.iterrows():
#     num_f = 2 # O el número que desees
#     ex, ey, f_ranges, r_pesos = inicializar_dispositivo_dinamico(
#         row["device_size"], row["atom_size"], num_f
#     )
#
#     # Invocación a la función de generación original
#     init_state = gn.initial_state_priv(ex, ey, int(row["num_trampas"]), r_pesos)

carpeta = "Init_data"
archivo_params = os.path.join(carpeta, "simulation_parameters.csv")

if not os.path.exists(archivo_params):
    print("ERROR: No se encuentra simulation_parameters.csv.")
    print("Asegúrate de ejecutar ConfigManager.export_to_init_data() en el Notebook primero.")
    sys.exit(1)

df_params = pd.read_csv(archivo_params)
num_simulations = len(df_params)

print(f"Construyendo estados iniciales para {num_simulations} simulaciones...")

for i, row in df_params.iterrows():
    # Calculamos el tamaño de la matriz a partir de los parámetros físicos
    eje_x = int(math.ceil(row["device_size"] / row["atom_size"]))
    eje_y = int(math.ceil(row["device_size"] / row["atom_size"]))
    num_trampas = int(row["num_trampas"])

    f_ranges, regiones_pesos = generar_configuracion_filamentos(eje_x, eje_y, num_filamentos=2)

    # # Definimos el filamento: ((x_start, x_end, y_start, y_end), peso)
    # regiones_pesos = [
    #     # equiespaciados para identificar bien los filamentos, el utlimo numero no entra en el rango
    #     # Cuatro filamentos
    #     # ((3, 6, 0, eje_x[i]), 50),  # Primera banda (filas 3-6)
    #     # ((13, 16, 0, eje_x[i]), 50),  # Primera banda (filas 3-6)
    #     # ((23, 26, 0, eje_x[i]), 60),  # Segunda banda (filas 15-18)
    #     # ((33, 36, 0, eje_x[i]), 50),  # Tercera banda (filas 30-34)
    #     # Dos filamentos
    #     ((8, 13, 0, eje_x), 70),  # Primera banda (filas 8-12)
    #     ((28, 33, 0, eje_x), 70),  # Segunda banda (filas 28-34)
    #     # Un filamento
    #     # ((17, 24, 0, 12), 75),  # Primera aprte del filamento
    #     # ((17, 24, eje_x - 12, eje_x), 70),  # Segunda parte del filamento
    #     # ((17, 24, 12, eje_x[i] - 12), 30),  # Segunda entre del filamento
    #     # (
    #     # (45, 55, 0, eje_x), 70,
    #     # ),
    # ]

    init_state = gn.initial_state_priv(eje_x, eje_y, num_trampas, regiones_pesos)
    # oxygen_state = Recombination.Init_OxygenState(row["device_size"], row["atom_size"])

    with open(f"{carpeta}/init_state_{i}.pkl", "wb") as f:
        pickle.dump(init_state, f)

    # with open(f"{carpeta}/oxygen_state_{i}.pkl", "wb") as f:
    #     pickle.dump(oxygen_state, f)

print("Estados iniciales generados correctamente.")
