import pickle
from RRAM import *
import pandas as pd
import time as time
from tqdm import tqdm

# comienzo leyendo los datos de la simulación almacenados en un archivo csv dentro de la carpeta Init y los guardo en sus respectivas variables
sim_parmtrs = Montecarlo.read_csv_to_dic("Init_data/simulation_parameters.csv")
sim_ctes = Montecarlo.read_csv_to_dic("Init_data/simulation_constants.csv")


for num_simulation in len(sim_parmtrs):

    # Asigno los valores de los datos de la simulación a las variables correspondientes
    device_size = sim_parmtrs['device_size'][num_simulation]
    atom_size = sim_parmtrs['atom_size'][num_simulation]
    x_size = sim_parmtrs['x_size'][num_simulation]
    y_size = sim_parmtrs['y_size'][num_simulation]
    num_trampas = sim_parmtrs['num_trampas'][num_simulation]

    init_simulation_time = sim_parmtrs['init_simulation_time'][num_simulation]
    total_simulation_time = sim_parmtrs['total_simulation_time'][num_simulation]
    num_pasos = sim_parmtrs['num_pasos'][num_simulation]
    paso_guardar = sim_parmtrs['paso_guardar'][num_simulation]

    voltaje_final = sim_parmtrs['voltaje_final'][num_simulation]
    voltaje_inicial = sim_parmtrs['initial_voltaje'][num_simulation]
    current_inicial = sim_parmtrs['initial_current'][num_simulation]
    init_temp = sim_parmtrs['init_temp'][num_simulation]
    initial_elec_field = sim_parmtrs['initial_elec_field'][num_simulation]


# paso_temporal = total_simulation_time / num_pasos

# # Creo el vector de datos como una matriz de num_pasos filas y las columnas necesarias (x,y,probabilidad recombionacion, velocidad)
# colunm_number = 4

# data = np.zeros((num_pasos, colunm_number))

# # Creo el excel donde voy a sacar todos los datos
# df = pd.DataFrame(columns=['Tiempo simulacion', 'velocidad', 'desplazamiento', 'prob_generacion'])


# # Creo el excel donde voy a sacar todos los datos
# for k in tqdm(range(1, num_pasos+1)):

#     # Guardo el estado anterior
#     last_state = actual_state

#     simulation_time = paso_temporal * k

#     # Calculo la corriente
#     voltaje += voltaje_final * paso_temporal

#     # Obtengo la corrriente, antes decido cual usar comprobando si ha percolado o no
#     # TODO: Revisar la corriente óhmica que no funciona
#     if Percolation.is_path(actual_state):
#         # Si ha percolado uso la corriente de percolación
#         # Corriente = CurentSolver.OmhCurrent(Temperatura, Campo_Electrico)
#         print("Ha percolado")
#         break
#     else:
#         # Si no ha percolado uso la corriente de campo
#         # TODO: REVISAR QUE LA CORRIENTE TIENE LAS UNIDADES CORRECTAS PORQUE NO CUADRAN VALORES.
#         Corriente = CurentSolver.poole_frenkel(temperatura, Campo_Electrico)/(1e-10)

#     # Obtengo los valores del campo eléctrico y la temperatura
#     Campo_Electrico = SimpleElectricField(voltaje, espesor_dispositivo)
#     # Temperatura = Temperature_Joule(voltaje, Corriente, T_0=350)

#     # Calculo la probabilidad de generación o recombinación para ello recorro toda la matriz
#     for i in range(eje_x):
#         for j in range(eje_y):
#             if actual_state[i, j] == 0:
#                 # TODO: REVISAR PROBABILIDAD QUE A VECES SALE MAYOR DE 1
#                 # TODO: HACER UN REESCALADO DE LOS VALORES PARA EVITAR TENER QUE TRABAJAR CON NUMEROS TAN GRANDES
#                 prob_generacion = Generation.generation(paso_temporal, Campo_Electrico, temperatura)
#                 random_number = np.random.rand()
#                 if random_number < prob_generacion:
#                     actual_state[i, j] = 1  # Generación

#     # Genero los oxígenos
#     oxygen_state = GenerateOxigen(oxygen_state, 30)

#     # Muevo los oxígenos
#     oxygen_state, velocidad, desplazamiento = Move_OxygenIons(
#         simulation_time, oxygen_state, temperatura, Campo_Electrico, atom_size, factor=1)

#     data[k-1] = np.array([simulation_time, velocidad, desplazamiento, prob_generacion])

#     # Obtengo la nueva configuración
#     actual_state, oxygen_state = Recombination.Recombine(actual_state, oxygen_state)

#     # Guardo el estado actual CADA paso_guardar PASOS MONTECARLO
#     if k % paso_guardar == 0:
#         configuraciones_matriz[int(k / paso_guardar) - 1] = actual_state
#         oxygen_matrix[int(k / paso_guardar) - 1] = oxygen_state

# # Guardar la lista en un archivo
# with open('Configuraciones.pkl', 'wb') as f:
#     pickle.dump(configuraciones_matriz, f)
# with open('Oxigeno.pkl', 'wb') as f:
#     pickle.dump(oxygen_matrix, f)

# start = time.time()

# # # Suponiendo que 'data' es un array de NumPy que ya contiene tus datos
# # data_filtrados = np.array([fila for fila in data if fila[-1] != 0.0])

# np.savetxt('datos.csv', data, header='tiempo simulacion, velocidad, desplazamiento, prob_generacion',
#            comments='', delimiter=', ')
# end = time.time()

# print(f"Tiempo de creación del txt: {end - start:.3f} segundos")
