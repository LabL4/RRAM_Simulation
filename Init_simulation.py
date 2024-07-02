import pickle
from RRAM import *
import pandas as pd


# Número de simulaciones que realizo
num_simulations = 10

# Defino los parámetros de la simulación

# Es espesor es un  array de unos multiplicados por el tamaño que quiero
device_size = np.ones(num_simulations) * 10e-9  # m
atom_size = np.ones(num_simulations) * 0.25e-9  # m
num_trampas = np.ones(num_simulations, dtype=int) * 300
priv_x = np.ones(num_simulations, dtype=int) * 10
priv_y = np.ones(num_simulations, dtype=int) * 15
total_simulation_time = np.ones(num_simulations) * 1
num_pasos = np.ones(num_simulations, dtype=int) * 10000
voltaje_final = np.ones(num_simulations) * 1
paso_guardar = np.ones(num_simulations, dtype=int) * 1

init_temp = np.ones(num_simulations) * 350
initial_elec_field = np.ones(num_simulations) * 0
initial_voltaje = np.ones(num_simulations) * 0
initial_current = np.ones(num_simulations) * 0
simulation_time = np.ones(num_simulations) * 0

# Hago un array con cada valor de los parámetros de la simulación y los vuelco sobre un excel para tener un registro
eje_x = np.round(device_size / atom_size).astype(int)
eje_y = np.round(device_size / atom_size).astype(int)

# Creo un dataframe con los parámetros de la simulación
df = pd.DataFrame(columns=['device_size', 'atom_size', 'x_size', 'y_size', 'num_trampas', 'priv_x', 'priv_y',
                           'total_simulation_time', 'num_pasos', 'voltaje_final', 'paso_guardar',
                           'init_temp', 'initial_elec_field', 'initial_voltaje', 'initial_current', 'simulation_time'])
df['device_size'] = device_size
df['atom_size'] = atom_size
df['x_size'] = eje_x
df['y_size'] = eje_y
df['num_trampas'] = num_trampas
df['priv_x'] = priv_x
df['priv_y'] = priv_y
df['total_simulation_time'] = total_simulation_time
df['num_pasos'] = num_pasos
df['voltaje_final'] = voltaje_final
df['paso_guardar'] = paso_guardar
df['init_temp'] = init_temp
df['initial_elec_field'] = initial_elec_field
df['initial_voltaje'] = initial_voltaje
df['initial_current'] = initial_current
df['simulation_time'] = simulation_time
df.to_csv('Init_data/parametros_simulacion.csv', index=False)


for i in range(num_simulations):
    # Defino la región que quiero privilegiar y su peso
    regiones_pesos = [
        ((priv_x[i], eje_x[i]-priv_x[i], eje_y[i]-priv_y[i], eje_y[i]),
         50),             # First three columns with higher weight
        ((priv_x[i], eje_x[i]-priv_x[i], 0, priv_y[i]), 50),  # First three columns with higher weight
    ]

    actual_state = Generation.initial_state_priv(eje_x[i], eje_y[i], num_trampas[i], regiones_pesos)
    oxygen_state = Init_OxygenState(device_size[i], atom_size[i])

    # Guardo el estado inicial con el nombre estado inicial mas el número de simulación
    with open('Init_data/actual_state_' + str(i) + '.pkl', 'wb') as f:
        pickle.dump(actual_state, f)

    with open('Init_data/oxygen_state_' + str(i) + '.pkl', 'wb') as f:
        pickle.dump(oxygen_state, f)

    # configuraciones_matriz = np.zeros((int((num_pasos[i] / paso_guardar[i])), eje_x[i], eje_y[i]))
    # oxygen_matrix = np.zeros((int((num_pasos[i] / paso_guardar[i])), eje_x[i], eje_y[i]))
