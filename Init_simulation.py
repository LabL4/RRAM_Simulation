from RRAM import *
import pandas as pd


# Número de simulaciones que realizo
num_simulations = 1

# Defino los parámetros de la simulación

# Es espesor es un  array de unos multiplicados por el tamaño que quiero
espesor_dispositivo = np.ones(num_simulations) * 10e-9  # m
atom_size = np.ones(num_simulations) * 0.25e-9  # m
num_trampas = np.ones(num_simulations) * 300
priv_x = np.ones(num_simulations) * 10
priv_y = np.ones(num_simulations) * 15
total_simulation_time = np.ones(num_simulations) * 1
num_pasos = np.ones(num_simulations) * 10000


eje_x = round(espesor_dispositivo / atom_size)
eje_y = round(espesor_dispositivo / atom_size)

# Defino la región que quiero privilegiar y su peso
regiones_pesos = [
    ((10, eje_x-10, eje_y-15, eje_y), 50),             # First three columns with higher weight
    ((10, eje_x-10, 0, 15), 50),                       # First three columns with higher weight
]

actual_state = Generation.initial_state_priv(eje_x, eje_y, num_trampas, regiones_pesos)

oxygen_state = Init_OxygenState(espesor_dispositivo, atom_size)


paso_temporal = total_simulation_time / num_pasos

# Creo el excel donde voy a sacar todos los datos
df = pd.DataFrame(columns=['Tiempo simulacion', 'desplazamiento'])

voltaje_final = 1
paso_guardar = 1

configuraciones_matriz = np.zeros((int((num_pasos / paso_guardar)), eje_x, eje_y))
oxygen_matrix = np.zeros((int((num_pasos / paso_guardar)), eje_x, eje_y))

# Configuraciones iniciales:
temperatura = 350
Campo_Electrico = 0
voltaje = 0
simulation_time = 0
Corriente = 0
