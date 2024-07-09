import csv
import numpy as np
from tqdm import tqdm

# |---------------------------------------------------------------------------------------------------------------------------------|
# | En esta sección están todas las fucniones necesarias para realizar la simulación, como el montecarlo, o la lectura de los datos |
# |---------------------------------------------------------------------------------------------------------------------------------|


# def funcion_montecarlo(num_pasos, simulation_time, voltaje_final, factor_externo,
#                        num_trampas=600, atom_size=0.25, espesor_dispositivo=10, temperatura_inicial=350.00,):

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
#     oxygen_state = GenerateOxigen(oxygen_state, 10)

#     # Muevo los oxígenos
#     oxygen_state, velocidad, desplazamiento = Move_OxygenIons(
#         simulation_time, oxygen_state, temperatura, Campo_Electrico, atom_size, factor=1, sim_ctes)

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
#     return data

def read_csv_to_dic(cvs_path: str):
    """
    Reads a CSV file and returns its contents as a list of dictionaries.
    Args:
        cvs_path (str): The name of the CSV file to read.

    Returns:
        list: A list of dictionaries, where each dictionary represents a row in the CSV file.
    """
    with open(cvs_path, mode='r') as archivo:
        reader = csv.DictReader(archivo)
        data = [row for row in reader]
    return data
