from RRAM import *
import pandas as pd


espesor_dispositivo = 10e-9        # nm
atom_size = 0.25e-9                # nm

num_trampas = 300

eje_x = round(espesor_dispositivo / atom_size)
eje_y = round(espesor_dispositivo / atom_size)

# Defino la región que quiero privilegiar y su peso
regiones_pesos = [
    ((10, eje_x-10, eje_y-15, eje_y), 50),             # First three columns with higher weight
    ((10, eje_x-10, 0, 15), 50),                       # First three columns with higher weight
]

actual_state = Generation.initial_state_priv(eje_x, eje_y, num_trampas, regiones_pesos)

oxygen_state = Init_OxygenState(espesor_dispositivo, atom_size)

total_simulation_time = 4
num_pasos = 10000
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
