import csv
import pickle
import numpy as np

from .findpath import *
from .Constants import *
from .Generation import *
from .Montecarlo import *
from .Percolation import *
from .Temperature import *
from .CurrentSolver import *
from .ElectricField import *

# from tqdm import tqdm


def read_csv_to_dic(cvs_path: str):
    """
    Reads a CSV file and returns its contents as a list of dictionaries.
    Args:
        cvs_path (str): The name of the CSV file to read.

    Returns:
        list: A list of dictionaries, where each dictionary represents a row in the CSV file.
    """
    with open(cvs_path, mode="r") as archivo:
        reader = csv.DictReader(archivo)
        data = [row for row in reader]
    return data


# Función para unir dos archivos pickle y guardar el resultado en un nuevo archivo
def merge_pickles_to_array(file1, file2, output_file):
    try:
        # Cargar datos del primer archivo
        with open(file1, "rb") as f1:
            data1 = pickle.load(f1)
    except (FileNotFoundError, EOFError):
        data1 = np.array([])  # Si no existe, inicializamos como un array vacío

    try:
        # Cargar datos del segundo archivo
        with open(file2, "rb") as f2:
            data2 = pickle.load(f2)
    except (FileNotFoundError, EOFError):
        data2 = np.array([])  # Si no existe, inicializamos como un array vacío

    # Convertir ambos datos en arrays de numpy (si no lo son ya)
    data1 = np.array(data1)
    data2 = np.array(data2)

    # Unir ambos arrays en uno solo
    combined_data = np.concatenate((data1, data2))

    # Guardar los datos combinados en un archivo pickle
    with open(output_file, "wb") as f_out:
        pickle.dump(combined_data, f_out)


# def pp_set(actual_state, voltage, E_field_vector, num_simulation, device_size, paso_temporal, paso_guardar, simulation_time, T_0, resistencia, sim_ctes, ** kwargs):
#    # Obtengo la corrriente, antes decido cual usar comprobando si ha percolado o no
#     if findpath.Percolation.is_path(actual_state):
#         # Copio el estado actual
#         ac = actual_state.copy()
#         resistance_matrix = findpath.find_path(ac)

#         # Si ha percolado uso la corriente de Ohm
#         try:
#             current, resistencia[k] = CurentSolver.OmhCurrent(
#                 voltage, resistance_matrix, **sim_ctes[num_simulation])
#         except Warning:
#             filename = f'Results/Null_Resistance/Configuration_Set_{voltage}_null_resistance.pkl'
#             print("Null resistance matrix in ", filename)
#             RepresentateState(resistance_matrix,
#                               f'Results/Null_Resistance/PS_resistance_matrix_{num_simulation}.png')
#             with open(filename, 'wb') as f:
#                 pickle.dump({"actual_state": ac, "resistance_matrix": resistance_matrix}, f)
#     else:
#         # Si no ha percolado uso la corriente de Poole-Frenkel
#         resistencia[k] = 0
#         mean_field = np.mean(E_field_vector)
#         current = CurentSolver.Poole_Frenkel(temperatura, mean_field, **sim_ctes[num_simulation])*(device_size)

#         # Obtengo los valores del campo eléctrico y la temperatura
#         E_field = SimpleElectricField(voltage, device_size)

#         temperatura = Temperature_Joule(voltage, current, T_0, **sim_ctes[num_simulation])

#         # Genero el vector campo eléctrico
#         for i in range(0, actual_state.shape[0]):
#             E_field_vector[i] = GapElectricField(voltage, i, actual_state, **sim_parmtrs[num_simulation])

#         # Calculo la probabilidad de generación o recombinación para ello recorro toda la matriz
#         for i in range(x_size):
#             prob_generacion = Generation.Generate(
#                 paso_temporal, E_field_vector[i], temperatura, **sim_ctes[num_simulation])
#             for j in range(y_size):
#                 if actual_state[i, j] == 0:
#                     random_number = np.random.rand()
#                     if random_number < prob_generacion:
#                         actual_state[i, j] = 1  # Generación de una vacante

#         data_pp_set[k] = np.array([simulation_time, voltage, current, temperatura, E_field, np.mean(E_field_vector), 0])

#         # Guardo el estado actual CADA paso_guardar PASOS MONTECARLO
#         if k % paso_guardar == 0:
#             config_matrix[int(k / paso_guardar) - 1] = actual_state
