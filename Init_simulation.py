import os
import shutil
import pickle
from RRAM import *
import pandas as pd
from RRAM import Recombination
from RRAM import Constants as cte

# Número de simulaciones que realizo
num_simulations = 1

# Defino la carpeta donde se guardan los datos iniciales de la simulación
carpeta = 'Init_data'

# Verifica si la carpeta existe
if os.path.exists(carpeta):
    # Elimina la carpeta y su contenido
    shutil.rmtree(carpeta)

# Crea la carpeta de nuevo
os.makedirs(carpeta)

# ------------------------------------------------------------------------------------------------------------------------------------------------------
# Defino los parámetros de la simulación
# ------------------------------------------------------------------------------------------------------------------------------------------------------

device_size = np.ones(num_simulations) * 10e-9  # m
atom_size = np.ones(num_simulations) * 0.25e-9  # m
num_trampas = np.ones(num_simulations, dtype=int) * 300

priv_y_sup_right = np.ones(num_simulations, dtype=int) * 10
priv_y_inf_right = np.ones(num_simulations, dtype=int) * 10
priv_x_right = np.ones(num_simulations, dtype=int) * 10
priv_y_sup_left = np.ones(num_simulations, dtype=int) * 10
priv_y_inf_left = np.ones(num_simulations, dtype=int) * 10
priv_x_left = np.ones(num_simulations, dtype=int) * 10

total_simulation_time = np.ones(num_simulations) * 2
num_pasos = np.ones(num_simulations, dtype=int) * 10000
voltaje_final = np.ones(num_simulations) * 1
paso_guardar = np.ones(num_simulations, dtype=int) * 1

init_temp = np.ones(num_simulations) * 350
initial_elec_field = np.ones(num_simulations) * 0
initial_voltaje = np.ones(num_simulations) * 0
initial_current = np.ones(num_simulations) * 0
init_simulation_time = np.ones(num_simulations) * 0

# Hago un array con cada valor de los parámetros de la simulación y los vuelco sobre un excel para tener un registro
eje_x = np.round(device_size / atom_size).astype(int)
eje_y = np.round(device_size / atom_size).astype(int)

# Creo un dataframe con los parámetros de la simulación
df = pd.DataFrame(columns=['device_size', 'atom_size', 'x_size', 'y_size', 'num_trampas',
                           'priv_y_sup_right', 'priv_y_inf_right', 'priv_x_right', 'priv_y_sup_left', 'priv_y_sup_left', 'priv_y_inf_left', 'priv_x_left',
                           'total_simulation_time', 'num_pasos', 'voltaje_final', 'paso_guardar',
                           'init_temp', 'initial_elec_field', 'initial_voltaje', 'initial_current', 'init_simulation_time'])

df['device_size'] = device_size
df['atom_size'] = atom_size
df['x_size'] = eje_x
df['y_size'] = eje_y
df['num_trampas'] = num_trampas

df['priv_y_sup_right'] = priv_y_sup_right
df['priv_y_inf_right'] = priv_y_inf_right
df['priv_x_right'] = priv_x_right
df['priv_y_sup_left'] = priv_y_sup_left
df['priv_y_inf_left'] = priv_y_inf_left
df['priv_x_left'] = priv_x_left

df['total_simulation_time'] = total_simulation_time
df['num_pasos'] = num_pasos
df['voltaje_final'] = voltaje_final
df['paso_guardar'] = paso_guardar
df['init_temp'] = init_temp
df['initial_elec_field'] = initial_elec_field
df['initial_voltaje'] = initial_voltaje
df['initial_current'] = initial_current
df['init_simulation_time'] = init_simulation_time

# Guardo el dataframe en un archivo csv
df.to_csv('Init_data/simulation_parameters.csv', index=False)

for i in range(num_simulations):
    # Defino la región que quiero privilegiar y su peso
    regiones_pesos = [
        ((priv_y_sup_right[i], eje_x[i]-priv_y_inf_right[i], eje_y[i]-priv_x_right[i], eje_y[i]),
         50),
        ((priv_y_sup_left[i], eje_x[i]-priv_y_inf_left[i], 0, priv_x_left[i]), 50),
    ]

    # Estado inicial de la simulación para los oxígenos y el sistema
    init_state = Generation.initial_state_priv(eje_x[i], eje_y[i], num_trampas[i], regiones_pesos)
    RepresentateState(init_state, 'Init_data/init_state_' + str(i))
    oxygen_state = Recombination.Init_OxygenState(device_size[i], atom_size[i])

    # Guardo el estado inicial con el nombre estado inicial mas el número de simulación
    with open('Init_data/init_state_' + str(i) + '.pkl', 'wb') as f:
        pickle.dump(init_state, f)

    with open('Init_data/oxygen_state_' + str(i) + '.pkl', 'wb') as f:
        pickle.dump(oxygen_state, f)

# # ------------------------------------------------------------------------------------------------------------------------------------------------------
# # Defino las constantes de la simulación y las guardo en un archivo
# # ------------------------------------------------------------------------------------------------------------------------------------------------------

t_0 = np.ones(num_simulations) * cte.t_0  # Characteristic vibration frequency of oxygen ions in HfOx
E_m = np.ones(num_simulations) * cte.E_m  # Migration energy of oxygen ions in HfOx
gamma_drift = np.ones(num_simulations) * cte.gamma_drift  # Drift coefficient of oxygen ions due to an external field


# Creo un dataframe nuevo con las constantes de la simulación
df_ctes = pd.DataFrame(columns=['vibration_frequency', 'migration_energy', 'drift_coefficient'])

df_ctes['vibration_frequency'] = t_0
df_ctes['migration_energy'] = E_m
df_ctes['drift_coefficient'] = gamma_drift

# Guardo el dataframe de las ctes en un archivo csv
df_ctes.to_csv('Init_data/simulation_constants.csv', index=False)
