from RRAM import Simulation, utils
from typing import final
from pathlib import Path
from turtle import back


import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import shutil
import sys

matplotlib.use("Agg")

# Asegúrate de que se ha pasado un parámetro
if len(sys.argv) > 1:
    num_simulation = int(sys.argv[1])
    guardar_datos = sys.argv[2]
    num_filamentos = int(sys.argv[3])

    guardar_datos = True if guardar_datos == "True" else False
    print(f"El número de simulacion es: {num_simulation + 1}")
    print(f"Se guardan las configuraciones: {guardar_datos}")
else:
    num_simulation = 1
    guardar_datos = False
    num_filamentos = 4

    print(f"El número de simulaciones es: {num_simulation}")
    print(f"Se guardan las configuraciones: {guardar_datos}")

# Se podria leer fuera y luego pasar el diccionario de cada simulación
sim_parmtrs = utils.read_csv_to_dic("Init_data/simulation_parameters.csv")
params = Simulation.SimulationParameters.from_dict(sim_parmtrs[num_simulation])

sim_cte = utils.read_csv_to_dic("Init_data/simulation_constants.csv")
ctes = Simulation.SimulationConstants.from_dict(sim_cte[num_simulation])

print(params)
print("\n----------------------------------------------------------------------\n")
print(ctes)

# Crear una lista para rastrear si se ha creado el CF para cada rango de filamentos

if num_filamentos == 2:
    filamentos_ranges = [(0, 19), (20, 39)]  # Se incluye el ultimo valor
elif num_filamentos == 4:
    filamentos_ranges = [(0, 9), (10, 19), (20, 29), (30, 39)]
elif num_filamentos == 1:
    filamentos_ranges = [(0, 39)]

CF_creado = np.full(len(filamentos_ranges), False, dtype=bool)

final_state_pp_set = Simulation.PP_set(
    num_simulation=num_simulation + 1,
    params=params,
    sim_ctes=ctes,
    CF_ranges=filamentos_ranges,
    CF_creado=CF_creado,
)

final_state_sp_set = Simulation.SP_set(
    final_state_pp_set=final_state_pp_set,
    num_simulation=num_simulation + 1,
    CF_ranges=filamentos_ranges,
)

final_state_pp_reset = Simulation.PP_reset(
    final_state_sp_set=final_state_sp_set,
    num_simulation=num_simulation + 1,
    CF_ranges=filamentos_ranges,
)

final_state_sp_reset = Simulation.SP_reset(
    final_state_pp_reset=final_state_pp_reset,
    num_simulation=num_simulation + 1,
    CF_ranges=filamentos_ranges,
)
print("Resultados de la destruccion de filamentos:\n")
for key, valor in final_state_sp_reset["roturas_dict"].items():
    print(f"Filamento destruido {key}:")
    for campo, dato in valor.items():
        print(f"  {campo}: {dato}")

desplazamiento = {
    "a": (0.025, 1.0),  # derecha, misma altura
    "b": (+0.005, 0.27),  # izquierda, un poco arriba
    "c": (0.02, 0.35),  # derecha, un poco abajo
    "d": (0.02, 1.0),  # izquierda, misma altura
    "e": (-0.11, 0.66),  # izquierda, misma altura
    "f": (0.025, 0.25),  # izquierda, un poco abajo
    "g": (-0.12, 1),  # derecha, un poco arriba
}

Simulation.simulation_IV(
    num_simulation=num_simulation + 1,
    figures_path=Path.cwd() / "Results" / f"Simulation_{num_simulation}" / "Figures",
    simulation_path=Path.cwd() / "Results" / f"Simulation_{num_simulation + 1}",
    desplazamiento=desplazamiento,
    voltaje_percolacion=final_state_pp_set["voltaje_percolacion"],
    roturas_dict=final_state_sp_reset["roturas_dict"],
)
