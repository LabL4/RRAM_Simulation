import pandas as pd
import pickle
import math
import sys
import os

ruta_raiz = os.getcwd() + "/"
sys.path.append(ruta_raiz)

from RRAM import Generation as gn

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

    # Definimos el filamento: ((x_start, x_end, y_start, y_end), peso)
    regiones_pesos = [
        # equiespaciados para identificar bien los filamentos, el utlimo numero no entra en el rango
        # Cuatro filamentos
        # ((3, 6, 0, eje_x[i]), 50),  # Primera banda (filas 3-6)
        # ((13, 16, 0, eje_x[i]), 50),  # Primera banda (filas 3-6)
        # ((23, 26, 0, eje_x[i]), 60),  # Segunda banda (filas 15-18)
        # ((33, 36, 0, eje_x[i]), 50),  # Tercera banda (filas 30-34)
        # Dos filamentos
        # ((8, 13, 0, eje_x), 70),  # Primera banda (filas 8-12)
        # ((28, 33, 0, eje_x), 70),  # Segunda banda (filas 28-34)
        # Un filamento
        # ((17, 24, 0, 12), 75),  # Primera aprte del filamento
        # ((17, 24, eje_x - 12, eje_x), 70),  # Segunda parte del filamento
        # ((17, 24, 12, eje_x[i] - 12), 30),  # Segunda entre del filamento
        (
        (45, 55, 0, eje_x), 70,
        ),
    ]

    init_state = gn.initial_state_priv(eje_x, eje_y, num_trampas, regiones_pesos)
    # oxygen_state = Recombination.Init_OxygenState(row["device_size"], row["atom_size"])

    with open(f"{carpeta}/init_state_{i}.pkl", "wb") as f:
        pickle.dump(init_state, f)

    # with open(f"{carpeta}/oxygen_state_{i}.pkl", "wb") as f:
    #     pickle.dump(oxygen_state, f)

print("Estados iniciales generados correctamente.")
