import numpy as np  # pyright: ignore[reportMissingImports]

# from RRAM import Plot_PostProcess as pplt
# from RRAM import Representate as rp
from RRAM import Generation as gn
from RRAM import Constants as cte
from RRAM import Recombination
import pandas as pd
from RRAM import *
import shutil
import pickle
import sys
import os

# ruta_raiz = 'C:/Users/Usuario/Documents/GitHub/RRAM_Simulation/'
# ruta_raiz = 'C:/Users/jimdo/Documents/GitHub/RRAM_Simulation/'
# ruta_raiz = '/Users/antonio_lopez_torres/Documents/GitHub/RRAM_Simulation/'  # Ruta en el mac
ruta_raiz = os.getcwd() + "/"
sys.path.append(ruta_raiz)

# Asegúrate de que se ha pasado un parámetro
if len(sys.argv) > 1:
    data_path = sys.argv[1]
    num_simulations = int(sys.argv[2])

    print(f"Ruta de los archivos de datos: {data_path}")
    print(f"El número de simulaciones es: {num_simulations}")

else:
    print("No se ha pasado ningún parámetro.")
    data_path = ruta_raiz + "Initial_data/"
    num_simulations = 1

    print(f"Ruta de los archivos de datos: {data_path}")
    print(f"El número de simulaciones es: {num_simulations}")
# ----------------------------------------------------------------------------------------------------------------------------------

# Defino la carpeta donde se guardan los datos iniciales de la simulación
carpeta = "Init_data"

# Verifica si la carpeta existe
if os.path.exists(carpeta):
    # Elimina la carpeta y su contenido
    shutil.rmtree(carpeta)

# Crea la carpeta de nuevo
os.makedirs(carpeta)

# ----------------------------------------------------------------------------------------------------------------------------------
# Defino los parámetros de la simulación
# ----------------------------------------------------------------------------------------------------------------------------------

device_size = np.ones(num_simulations) * 5e-9  # m antes era 10e-9
# m TODO: Esto se deberia llamarse tamaño del grid mejor
atom_size = np.ones(num_simulations) * 0.125e-9
num_trampas = np.ones(num_simulations, dtype=int) * 70  # 130

priv_y_sup_right = np.ones(num_simulations, dtype=int) * 15
priv_y_inf_right = np.ones(num_simulations, dtype=int) * 15
priv_y_sup_left = np.ones(num_simulations, dtype=int) * 15
priv_y_inf_left = np.ones(num_simulations, dtype=int) * 15

priv_x_right = np.ones(num_simulations, dtype=int) * 20
priv_x_left = np.ones(num_simulations, dtype=int) * 20

total_simulation_time = np.ones(num_simulations) * 10
# time step de mili segundo y milivoltios de step voltaje incluso de 0.01
num_pasos = np.ones(num_simulations, dtype=int) * 10000
voltaje_final = np.ones(num_simulations) * 1.1  # 1.4
voltaje_final_set = np.ones(num_simulations) * 1.1  # 1.1

paso_guardar = np.ones(num_simulations, dtype=int) * 1

init_temp = np.ones(num_simulations) * 300  # K #310
initial_elec_field = np.ones(num_simulations) * 0
initial_voltaje = np.ones(num_simulations) * 0
initial_current = np.ones(num_simulations) * 0
init_simulation_time = np.ones(num_simulations) * 0

# Hago un array con cada valor de los parámetros de la simulación y los vuelco sobre un excel para tener un registro
eje_x = np.round(device_size / atom_size).astype(int)
eje_y = np.round(device_size / atom_size).astype(int)

# Creo un dataframe con los parámetros de la simulación
df = pd.DataFrame(
    columns=[
        "device_size",
        "atom_size",
        "x_size",
        "y_size",
        "num_trampas",
        "priv_y_sup_right",
        "priv_y_inf_right",
        "priv_x_right",
        "priv_y_sup_left",
        "priv_y_sup_left",
        "priv_y_inf_left",
        "priv_x_left",
        "total_simulation_time",
        "num_pasos",
        "voltaje_final",
        "voltaje_final_set",
        "paso_guardar",
        "init_temp",
        "initial_elec_field",
        "initial_voltaje",
        "initial_current",
        "init_simulation_time",
    ]
)

df["device_size"] = device_size
df["atom_size"] = atom_size
df["x_size"] = eje_x
df["y_size"] = eje_y
df["num_trampas"] = num_trampas

df["priv_y_sup_right"] = priv_y_sup_right
df["priv_y_inf_right"] = priv_y_inf_right
df["priv_x_right"] = priv_x_right
df["priv_y_sup_left"] = priv_y_sup_left
df["priv_y_inf_left"] = priv_y_inf_left
df["priv_x_left"] = priv_x_left

df["total_simulation_time"] = total_simulation_time
df["num_pasos"] = num_pasos
df["voltaje_final"] = voltaje_final
df["voltaje_final_set"] = voltaje_final_set

df["paso_guardar"] = paso_guardar
df["init_temp"] = init_temp
df["initial_elec_field"] = initial_elec_field
df["initial_voltaje"] = initial_voltaje
df["initial_current"] = initial_current
df["init_simulation_time"] = init_simulation_time

carpeta_results = ruta_raiz + "Results/"

# Guardo el dataframe en un archivo csv
df.to_csv("Init_data/simulation_parameters.csv", index=False)

for i in range(num_simulations):
    # print(f"Simulación {i}", eje_x[i], eje_y[i], num_trampas[i])
    regiones_pesos = [
        # equiespaciados para identificar bien los filamentos, el utlimo numero no entra en el rango
        # Cuatro filamentos
        # ((3, 6, 0, eje_x[i]), 50),  # Primera banda (filas 3-6)
        # ((13, 16, 0, eje_x[i]), 50),  # Primera banda (filas 3-6)
        # ((23, 26, 0, eje_x[i]), 60),  # Segunda banda (filas 15-18)
        # ((33, 36, 0, eje_x[i]), 50),  # Tercera banda (filas 30-34)
        # Dos filamentos
        # ((8, 13, 0, eje_x[i]), 40),  # Primera banda (filas 8-12)
        # ((28, 33, 0, eje_x[i]), 40),  # Segunda banda (filas 28-34)
        # Un filamento
        ((17, 24, 0, 12), 75),  # Primera aprte del filamento
        ((17, 24, eje_x[i] - 12, eje_x[i]), 70),  # Segunda parte del filamento
        # ((17, 24, 12, eje_x[i] - 12), 30),  # Segunda entre del filamento
    ]

    # Ruta de las imagenes de cada simulación
    ruta_simulation = os.path.join(carpeta_results, f"Init_data/simulation_{i}")
    # os.makedirs(ruta_simulation, exist_ok=True)

    # Estado inicial de la simulación para los oxígenos y el sistema
    init_state = gn.initial_state_priv(
        eje_x[i], eje_y[i], num_trampas[i], regiones_pesos
    )
    oxygen_state = Recombination.Init_OxygenState(device_size[i], atom_size[i])

    # Guardo el estado inicial con el nombre estado inicial mas el número de simulación
    with open("Init_data/init_state_" + str(i) + ".pkl", "wb") as f:
        pickle.dump(init_state, f)

    with open("Init_data/oxygen_state_" + str(i) + ".pkl", "wb") as f:
        pickle.dump(oxygen_state, f)

    ruta_figuras = ruta_raiz + f"pruebas_inicio/simulation_{i}"

    # Representar la región privilegiada
    # RepresentateState(init_state, ruta_figuras + '_init_state.png')
    # rp.plot_privileged_regions(eje_x[i], eje_y[i], regiones_pesos, ruta_figuras)

# ------------------------------------------------------------------------------------------------------------------------------------------------------
# Defino las constantes de la simulación y las guardo en un archivo
# ------------------------------------------------------------------------------------------------------------------------------------------------------

# Characteristic vibration frequency of oxygen ions in HfOx
t_0 = np.ones(num_simulations) * cte.t_0

# Migration energy of oxygen ions in HfOx
# with open(data_path + "E_m.pkl", 'rb') as f:
#     E_m = pickle.load(f)
E_m = np.ones(num_simulations) * cte.E_m

# Constante de red, el paper original propone 0.25 nm
cte_red = np.ones(num_simulations) * cte.cte_red

# Energía de activación en eV
with open(data_path + "E_a.pkl", "rb") as f:
    E_a = pickle.load(f)
# E_a = np.ones(num_simulations) * cte.E_a

# Drift coefficient of oxygen ions due to an external field
# with open(data_path + "drift_coefficient.pkl", 'rb') as f:
#     gamma_drift = pickle.load(f)
gamma_drift = np.ones(num_simulations) * cte.gamma_drift

#
with open(data_path + "factor_generacion.pkl", "rb") as f:
    factor_generacion = pickle.load(f)
# factor_generacion = np.ones(num_simulations) * cte.factor_generacion

# Recombination enhancement factor due to the presence of excessive oxygen ions
beta_0 = np.ones(num_simulations) * cte.beta_0

# Decay length of the oxygen concentration
L_p = np.ones(num_simulations) * cte.L_p

# Coefficient representing the local enhancement factor due to the electric field
with open(data_path + "gamma.pkl", "rb") as f:
    gamma = pickle.load(f)
# gamma = np.ones(num_simulations) * cte.gamma

# Resistance ohmic of the device
with open(data_path + "ohm_resistence.pkl", "rb") as f:
    ohm_resistence = pickle.load(f)
# ohm_resistence = np.ones(num_simulations) * cte.ohm_resistence

# Potential barrier at the metal and insulator interface
pb_metal_insul = np.ones(num_simulations) * cte.pb_metal_insul
pb_metal_insul_reset = np.ones(num_simulations) * cte.pb_metal_insul_reset

# Permitividad relativa del material HfOx
permitividad_relativa = np.ones(num_simulations) * cte.permitividad_relativa
permitividad_relativa_reset = np.ones(num_simulations) * cte.permitividad_relativa_reset

# Término inicial de la ecuación de Poole-Frenkel
with open(data_path + "I_0.pkl", "rb") as f:
    I_0 = pickle.load(f)
# I_0 = np.ones(num_simulations) * cte.I_0

# Término inicial de la ecuación de Poole-Frenkel para el reset
with open(data_path + "I_0_reset.pkl", "rb") as f:
    I_0_reset = pickle.load(f)
# I_0_reset = np.ones(num_simulations) * cte.I_0_reset

# Constante de resistencia térmica en K/W cuando el sistema no percola
with open(data_path + "r_termica_no_percola.pkl", "rb") as f:
    r_termica_no_percola = pickle.load(f)
# r_termica_no_percola = np.ones(num_simulations) * cte.r_termica_no_percola

# Constante de resistencia térmica en K/W cuando el sistema percola
with open(data_path + "r_termica_percola.pkl", "rb") as f:
    r_termica_percola = pickle.load(f)
# r_termica_percola = np.ones(num_simulations) * cte.r_termica_percola

# Constante de resistencia térmica en K/W cuando el sistema percola
with open(data_path + "recombination_energy.pkl", "rb") as f:
    recombination_energy = pickle.load(f)

# Numero oxigenos generados primera parte reset 1     if abs(voltage) > 0.7:
with open(data_path + "num_oxigenos_pp_reset_1.pkl", "rb") as f:
    num_oxigenos_pp_reset_1 = pickle.load(f)

# Numero oxigenos generados primera parte reset 2     if abs(voltage) > 1.1:
with open(data_path + "num_oxigenos_pp_reset_2.pkl", "rb") as f:
    num_oxigenos_pp_reset_2 = pickle.load(f)

    # Constante de resistencia térmica en K/W cuando el sistema percola
with open(data_path + "num_oxigenos_sp_reset.pkl", "rb") as f:
    num_oxigenos_sp_reset = pickle.load(f)
# r_termica_percola = np.ones(num_simulations) * cte.r_termica_percola

# Creo un dataframe nuevo con las constantes de la simulación
df_ctes = pd.DataFrame(
    columns=[
        "vibration_frequency",
        "migration_energy",
        "drift_coefficient",
        "cte_red",
        "recom_enchancement_factor",
        "decaimiento_concentracion",
        "activation_energy",
        "gamma",
        "ohm_resistence",
        "pb_metal_insul",
        "pb_metal_insul_reset",
        "permitividad_relativa",
        "permitividad_relativa_reset",
        "I_0",
        "I_0_reset",
        "r_termica_percola",
        "r_termica_no_percola",
        "factor_generacion",
        "recombination_energy",
        "num_oxigenos_pp_reset_1",
        "num_oxigenos_pp_reset_2",
        "num_oxigenos_sp_reset",
    ]
)

df_ctes["vibration_frequency"] = t_0
df_ctes["migration_energy"] = E_m
df_ctes["drift_coefficient"] = gamma_drift
df_ctes["cte_red"] = cte_red
df_ctes["recom_enchancement_factor"] = beta_0
df_ctes["decaimiento_concentracion"] = L_p
df_ctes["activation_energy"] = E_a
df_ctes["gamma"] = gamma
df_ctes["ohm_resistence"] = ohm_resistence
df_ctes["pb_metal_insul"] = pb_metal_insul
df_ctes["pb_metal_insul_reset"] = pb_metal_insul_reset
df_ctes["permitividad_relativa"] = permitividad_relativa
df_ctes["permitividad_relativa_reset"] = permitividad_relativa_reset
df_ctes["I_0"] = I_0
df_ctes["I_0_reset"] = I_0_reset
df_ctes["r_termica_percola"] = r_termica_percola
df_ctes["r_termica_no_percola"] = r_termica_no_percola
df_ctes["factor_generacion"] = factor_generacion
df_ctes["recombination_energy"] = recombination_energy
df_ctes["num_oxigenos_pp_reset_1"] = num_oxigenos_pp_reset_1
df_ctes["num_oxigenos_pp_reset_2"] = num_oxigenos_pp_reset_2
df_ctes["num_oxigenos_sp_reset"] = num_oxigenos_sp_reset

# Guardo el dataframe de las ctes en un archivo csv
print(df_ctes)

df_ctes.to_csv(ruta_raiz + "/Init_data/simulation_constants.csv", index=False)
