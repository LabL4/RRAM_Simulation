from numpy import save, size
import time as time
import pandas as pd
import logging
import pickle
import shutil
import sys
import os

from RRAM import Plot_PostProcess as pplt
from RRAM import Recombination
from RRAM import exceptions
from tqdm import tqdm
from RRAM import *

import warnings
warnings.filterwarnings("error")

# Asegúrate de que se ha pasado un parámetro
if len(sys.argv) > 1:
    num_simulation = int(sys.argv[1])
    guardar_datos = sys.argv[2]

    guardar_datos = True if guardar_datos == 'True' else False
    print(f"El número de simulacion es: {num_simulation + 1}")
    print(f"Se guardan las configuraciones: {guardar_datos}")
else:
    num_simulation = 10
    guardar_datos = False

    print(f"El número de simulaciones es: {num_simulation+1}")
    print(f"Se guardan las configuraciones: {guardar_datos}")

ruta_raiz = 'C:/Users/Usuario/Documents/GitHub/RRAM_Simulation/'
# ruta_raiz = '/Users/antonio_lopez_torres/Documents/GitHub/RRAM_Simulation/'  # Ruta en el mac
sys.path.append(ruta_raiz)
# endregion

# region Definición de valores iniciales y constantes de la simulación

# comienzo leyendo los datos de la simulación almacenados en un archivo csv dentro de la carpeta Init y los guardo en sus respectivas variables
sim_parmtrs = Montecarlo.read_csv_to_dic("Init_data/simulation_parameters.csv")
sim_ctes = Montecarlo.read_csv_to_dic("Init_data/simulation_constants.csv")

# Defino la carpeta donde se guardan los datos iniciales de la simulación
carpeta_results = 'Results'
simulation_path = os.path.join(carpeta_results, f'simulation_{num_simulation + 1}/')
figures_path = os.path.join(carpeta_results, 'Figures')

# Crea la carpeta de nuevo
os.makedirs(simulation_path, exist_ok=True)
os.makedirs(simulation_path + 'Figures', exist_ok=True)

set_simulation_path = os.path.join(carpeta_results, f'simulation_{num_simulation + 1}/set/')
os.makedirs(set_simulation_path, exist_ok=True)

reset_simulation_path = os.path.join(carpeta_results, f'simulation_{num_simulation + 1}/reset/')
os.makedirs(reset_simulation_path, exist_ok=True)

# Defino las cabeceras de los archivos csv
header_files = 'Tiempo simulacion [s],Voltaje [V],Intensidad [A],Temperatura [K],Campo Simple [V/m],Campo Gap medio [V/m],Velocidad [m/s]'

# Pongo el nombre de la simulación y un salto de línea
print(f"\nSimulacion {num_simulation + 1}")

# Asigno los valores de los datos de la simulación a las variables correspondientes
device_size = float(sim_parmtrs[num_simulation]['device_size'])
atom_size = float(sim_parmtrs[num_simulation]['atom_size'])
x_size = int(sim_parmtrs[num_simulation]['x_size'])
y_size = int(sim_parmtrs[num_simulation]['y_size'])
num_trampas = int(sim_parmtrs[num_simulation]['num_trampas'])

init_simulation_time = float(sim_parmtrs[num_simulation]['init_simulation_time'])
total_simulation_time = float(sim_parmtrs[num_simulation]['total_simulation_time'])
num_pasos = int(sim_parmtrs[num_simulation]['num_pasos'])
paso_guardar = int(sim_parmtrs[num_simulation]['paso_guardar'])

voltaje_max_simulation = float(sim_parmtrs[num_simulation]['voltaje_final'])
voltaje_final_set = float(sim_parmtrs[num_simulation]['voltaje_final_set'])

voltage = float(sim_parmtrs[num_simulation]['initial_voltaje'])
current = float(sim_parmtrs[num_simulation]['initial_current'])
temperatura = float(sim_parmtrs[num_simulation]['init_temp'])
E_field = float(sim_parmtrs[num_simulation]['initial_elec_field'])

Resistencia_serie = 25
current = 0

# Leo los estados iniciales de la simulación
with open('Init_data/init_state_' + str(num_simulation) + '.pkl', 'rb') as f:
    actual_state = pickle.load(f)
with open('Init_data/oxygen_state_' + str(num_simulation) + '.pkl', 'rb') as f:
    oxygen_state = pickle.load(f)
RepresentateState(actual_state, simulation_path + 'Figures/initial_pp_set_' + str(num_simulation))

# Defino las matrices donde guardo las configuración del sistema y la de los oxígenos
config_matrix_pp_set = np.zeros((int((num_pasos / paso_guardar)), x_size, y_size))

# Defino el paso temporal
paso_temporal = total_simulation_time / num_pasos
paso_potencial = voltaje_max_simulation / num_pasos
factor_generacion = float(sim_ctes[num_simulation]['factor_generacion'])
print("\nEl valor de la resistencia de cada casilla es: ", sim_ctes[num_simulation]['ohm_resistence'])
print("El valor de gamma es: ", sim_ctes[num_simulation]['gamma'])
print("El valor de resistencia termica no percola es: ", sim_ctes[num_simulation]['r_termica_no_percola'])
print("El valor de resistencia termica percola es: ", sim_ctes[num_simulation]['r_termica_percola'])
print("El valor de I_0 es: ", sim_ctes[num_simulation]['I_0'])
print("El valor de E_a es: ", sim_ctes[num_simulation]['activation_energy'])
print("El valor del factor de generacion es; ", sim_ctes[num_simulation]['factor_generacion'])

# Creo el vector de diferencias de potencial
vector_ddp = np.arange(0.000, voltaje_max_simulation + paso_potencial, paso_potencial)

# Creo el vector de datos como una matriz de num_pasos filas y las columnas necesarias
colunm_number = 7
data_pp_set = np.zeros((num_pasos, colunm_number))

# Inicializo el campo eléctrico
E_field_vector = np.zeros((actual_state.shape[0]))
num_vacantes = np.zeros(num_pasos+1)
resistencia = np.zeros(num_pasos+1)
T_0 = float(sim_parmtrs[num_simulation]['init_temp'])
# endregion

sistema_percola = False
num_max_vacantes = (device_size/atom_size)**2

print(f"\nComienza la primera parte del set")
# region primera parte del set
# for k in tqdm(range(0, num_pasos)):
for k in (range(0, num_pasos)):
    # Guardo el estado anterior
    last_state = actual_state

    # Actualizo el tiempo de simulación
    simulation_time = paso_temporal * k

    # Actualizo el voltaje
    voltage = vector_ddp[k]

    num_vacantes[k] = np.sum(actual_state)

    # Si se llena el 90 del espacio de la matriz salto a otra simulación
    if np.sum(actual_state) > int(0.9*num_max_vacantes):
        print("\nSe ha llenado el 90% del espacio de la matriz en la iteración: ", k)
        # Represento la temperatura
        save_path = simulation_path + f'Figures/LLENADO_temperature_pp_set_{num_simulation+1}'
        RepresentateState(actual_state, simulation_path +
                          f'Figures/LLENADO_configuration_pp_set_{num_simulation + 1}.png')

        # region representar temperatura
        # Crear un array de ejemplo
        data_pp_set[k-1:] = np.nan      # Añadir valores nulos a partir de la fila k

        # Eliminar filas con valores nulos
        data_pp_set = data_pp_set[~np.isnan(data_pp_set).any(axis=1)]

        # Obtengo los valores de tempertura voltaje
        datos_temperatura = data_pp_set[:, 3]
        datos_voltage = data_pp_set[:, 1]

        # Represento la temperatura
        fig, axes = plt.subplots()

        pplt.config_ax(axes)
        pplt.config_ax(axes)

        axes.set_xlabel('Voltaje [V]')
        axes.set_ylabel('Temperatura [K]')
        global_tittle = 'Temperatura vs Voltaje'
        axes.set_title(global_tittle, fontsize=18, pad=15)

        axes.plot(datos_voltage, datos_temperatura, color='blue', label='Temperatura [K]')
        fig.savefig(save_path + '.pdf', bbox_inches='tight')
        # endregion

        raise exceptions.MaxVacantesException()
    if voltage > voltaje_final_set:
        print("\nSe ha superado el voltaje de ruptura en la iteracion: ", k)

        # Verifica si el sistema ha percolado    
        if not sistema_percola:
            raise exceptions.NoPercolationException()

        k_ruptura = k

        voltaje_max_set = vector_ddp[k]
        config_matrix_recortada = config_matrix_pp_set[k, :, :]
        tiempo_pp_set = paso_temporal * (k - 1)  # Le quitamos un paso porque se ha superado el voltaje de ruptura
        resistencia_copy = resistencia.copy()

        print("Voltaje final set", voltaje_max_set, 'en el tiempo ', tiempo_pp_set, "\n")

        # Crear un array de ejemplo
        data_pp_set[k-1:] = np.nan      # Añadir valores nulos a partir de la fila k
        num_vacantes[k:] = np.nan       # Añadir valores nulos a partir de la fila k
        resistencia[k:] = np.nan        # Añadir valores nulos a partir de la fila k

        # Eliminar filas con valores nulos
        data_pp_set = data_pp_set[~np.isnan(data_pp_set).any(axis=1)]
        num_vacantes = num_vacantes[~np.isnan(num_vacantes)]
        resistencia = resistencia[~np.isnan(resistencia)]

        # RepresentateState(resistance_matrix, simulation_path + f'Figures/final_pp_set_resistance_{num_simulation+1}.png')
        break
    # Obtengo la corrriente, antes decido cual usar comprobando si ha percolado o no
    if Percolation.is_path(actual_state):
        if sistema_percola is False:
            print("\nEl sistema ha percolado en la iteración: ", k, " que corresponde con el voltaje: ", voltage)
            voltage_percolacion = voltage
        sistema_percola = True

        # Cambio la probabilidad de generación de vacantes para controlar la percolación
        sim_ctes[num_simulation]['gamma'] = str(float(sim_ctes[num_simulation]['gamma']) / factor_generacion)

        # Copio el estado actual
        ac = actual_state.copy()
        resistance_matrix = findpath.find_path(ac)

        voltage_RRAM = voltage  # - (current * Resistencia_serie)
        # Si ha percolado uso la corriente de Ohm

        try:
            current, resistencia[k] = CurentSolver.OmhCurrent(
                voltage_RRAM, resistance_matrix, **sim_ctes[num_simulation])

            # Hago la guarrada de ajustar la corriente
            voltaje_division = (voltaje_final_set - voltage_percolacion) / 5

            # Ajusto cierta progresion en la resistencia
            if voltage_percolacion < voltage < voltage_percolacion + voltaje_division:
                current = current / 4
                print('Region 0 con voltaje entre ' + str(voltage_percolacion) +
                      ' y ' + str(voltage_percolacion + voltaje_division))
            elif voltage_percolacion + voltaje_division < voltage < voltage_percolacion + 1.5*voltaje_division:
                current = current / 3.5
                print('Region 1 con voltaje entre ' + str(voltage_percolacion + voltaje_division) +
                      ' y ' + str(voltage_percolacion + 1.5*voltaje_division))
            elif voltage_percolacion + 1.5*voltaje_division < voltage < voltage_percolacion + 2*voltaje_division:
                current = current / 2
                print('Region 2 con voltaje entre ' + str(voltage_percolacion + 1.5*voltaje_division) +
                      ' y ' + str(voltage_percolacion + 2*voltaje_division))
            elif voltage_percolacion + 2*voltaje_division < voltage < voltage_percolacion + 3*voltaje_division:
                current = current / 1.3
                print('Region 3 con voltaje entre ' + str(voltage_percolacion + 2*voltaje_division) +
                      ' y ' + str(voltage_percolacion + 3*voltaje_division))
            elif voltage_percolacion + 3*voltaje_division < voltage < voltage_percolacion + 4*voltaje_division:
                current = current / 1.1
                print('Region 4 con voltaje entre ' + str(voltage_percolacion + 3*voltaje_division) +
                      ' y ' + str(voltage_percolacion + 4*voltaje_division))
            elif voltage_percolacion + 4*voltaje_division < voltage < voltaje_final_set:
                current = current / 1.0
                print('Region 5 con voltaje entre ' + str(voltage_percolacion +
                      4*voltaje_division) + ' y ' + str(voltaje_final_set))
        except Warning:

            filename = simulation_path + f'Null_Resistance/Configuration_Set_{voltage}_null_resistance.pkl'
            print("Null resistance matrix in ", filename)
            RepresentateState(resistance_matrix, simulation_path +
                              f'Figures/Null_Resistance/NULL_resistance_matrix_pp_set_{num_simulation+1}.png')
            with open(filename, 'wb') as f:
                pickle.dump({"actual_state": ac, "resistance_matrix": resistance_matrix}, f)

    else:
        sistema_percola = False

        # Cambio el valor de gamma para favorer la generación de vacantes
        sim_ctes[num_simulation]['gamma'] = str(float(sim_ctes[num_simulation]['gamma']) / 1)

        # Si no ha percolado uso la corriente de Poole-Frenkel
        resistencia[k] = 0

        mean_field = np.mean(E_field_vector)
        current = CurentSolver.Poole_Frenkel(temperatura, mean_field, **sim_ctes[num_simulation])*(device_size)

    # Obtengo los valores del campo eléctrico y la temperatura
    # voltage_RRAM = voltage - (current * Resistencia_serie)
    E_field = SimpleElectricField(voltage, device_size)
    temperatura = Temperature_Joule(voltage, current, sistema_percola, T_0, ** sim_ctes[num_simulation])

    # Genero el vector campo eléctrico
    for i in range(0, actual_state.shape[0]):
        E_field_vector[i] = GapElectricField(voltage, i, actual_state, **sim_parmtrs[num_simulation])

    # Calculo la probabilidad de generación o recombinación para ello recorro toda la matriz
    for i in range(x_size):
        prob_generacion = Generation.Generate(
            paso_temporal, E_field_vector[i], temperatura, **sim_ctes[num_simulation])
        for j in range(y_size):
            if actual_state[i, j] == 0:
                random_number = np.random.rand()
                if random_number < prob_generacion:
                    actual_state[i, j] = 1  # Generación de una vacante
    data_pp_set[k-1] = np.array([simulation_time, voltage, current, temperatura,
                                E_field, np.mean(E_field_vector), 0])

    # Guardo el estado actual CADA paso_guardar PASOS MONTECARLO
    if k % paso_guardar == 0:
        config_matrix_pp_set[int(k / paso_guardar) - 1] = actual_state
# endregion

# region Guardar datos del Primera parte del set

# Si está activada la opción de guardar los datos:
if guardar_datos:
    # Cuando acaba la simulacion guardo las matrices de configuración
    with open(set_simulation_path + f'Configurations_pp_set_{num_simulation+1}.pkl', 'wb') as f:
        pickle.dump(config_matrix_pp_set, f)

# Cuando acaba la simulacion guardo ele stado final de configuracion
with open(set_simulation_path + f'Last_Configuration_pp_set_{num_simulation+1}.pkl', 'wb') as f:
    pickle.dump(actual_state, f)

np.savetxt(set_simulation_path + f'resultados_pp_set_{num_simulation+1}.csv',
           data_pp_set, header=header_files, delimiter=',')

# Guardo las vacantes generadas en el forming
with open(set_simulation_path + f"Vacantes_resistencia_{num_simulation+1}.txt", "w") as f:
    for v1, v2, v3, v4 in zip(data_pp_set[:, 0], data_pp_set[:, 1], num_vacantes, resistencia):
        f.write(f"{v1} {v2} {v3} {v4}\n")

# Leer el contenido del archivo TXT
with open(set_simulation_path + f"Vacantes_resistencia_{num_simulation+1}.txt", 'r') as file:
    lines = file.readlines()

header_files_extra = 'Tiempo simulacion [s],Voltaje [V],Resistencia [Ohm],Numero de vacantes \n'

# Añadir el texto en la primera fila
lines.insert(0, header_files_extra)

# Escribir el contenido de nuevo en el archivo TXT
with open(set_simulation_path + f"Vacantes_resistencia_{num_simulation+1}.txt", 'w') as file:
    file.writelines(lines)
# endregion

# region Segunda parte del Set

# Estado inicial de la simulación reset para las vacantes
with open(set_simulation_path + f'Last_Configuration_pp_set_{num_simulation+1}.pkl', 'rb') as file:
    # Carga el contenido del archivo
    initial_configuration = pickle.load(file)

# Defino el paso temporal según donde haya acabado la primera parte del set
num_pasos = k_ruptura

# Creo el vector de datos como una matriz de num_pasos filas y las columnas necesarias
data_sp_set = np.zeros((num_pasos, colunm_number))

# Defino las matrices donde guardo las configuración del sistema.
config_matrix_sp_set = np.zeros((int((num_pasos / paso_guardar)), x_size, y_size))

# Creo el vector de diferencias de potencial
vector_ddp = np.arange(voltaje_max_set, 0.000, -paso_potencial)

print("Voltaje inicial sp set", voltaje_max_set)

# Estado iniciales de la simulación para el reset
actual_state = initial_configuration
RepresentateState(actual_state, simulation_path + f'Figures/Initial_configuration_sp_set_{num_simulation+1}.png')

print(f"\n Comienza la segunda parte del set")
# Ciclo para la segunda parte del set
# for k in tqdm(range(0, num_pasos)):
for k in (range(0, num_pasos)):
    # Actualizo el tiempo de simulación
    simulation_time = paso_temporal * k
    # Actualizo el voltaje
    voltage = vector_ddp[k]
    # Obtengo la corrriente, antes decido cual usar comprobando si ha percolado o no

    # Si se llena el 90 del espacio de la matriz salto a otra simulación
    if num_vacantes[k] > int(0.85*num_max_vacantes):
        # Represento la temperatura
        save_path = simulation_path + f'Figures/LLENADO_temperature_pp_set_{num_simulation+1}'
        RepresentateState(actual_state, simulation_path +
                          f'Figures/LLENADO_configuration_pp_set_{num_simulation+1}.png')

        # region representar temperatura
        # Crear un array de ejemplo
        data_pp_set[k-1:] = np.nan      # Añadir valores nulos a partir de la fila k

        # Eliminar filas con valores nulos
        data_pp_set = data_pp_set[~np.isnan(data_pp_set).any(axis=1)]

        # Obtengo los valores de tempertura voltaje
        datos_temperatura = data_pp_set[:, 3]
        datos_voltage = data_pp_set[:, 1]

        # Represento la temperatura
        fig, axes = plt.subplots()

        pplt.config_ax(axes)
        pplt.config_ax(axes)

        axes.set_xlabel('Voltaje [V]')
        axes.set_ylabel('Temperatura [K]')
        global_tittle = 'Temperatura [K] vs Voltaje [V]'
        axes.set_title(global_tittle, fontsize=18, pad=15)

        axes.plot(datos_voltage, datos_temperatura, color='blue', label='Temperatura [K]')
        fig.savefig(save_path + '.pdf', bbox_inches='tight')
        # endregion

        raise exceptions.MaxVacantesException()

    if Percolation.is_path(actual_state):
        sim_ctes[num_simulation]['gamma'] = str(float(sim_ctes[num_simulation]['gamma']) / factor_generacion)
        ac = actual_state.copy()
        resistance_matrix = findpath.find_path(ac)

        # Si ha percolado uso la corriente de Ohm
        voltage_RRAM = voltage  # - (current * Resistencia_serie)
        try:
            current, resistencia[k] = CurentSolver.OmhCurrent(
                voltage_RRAM, resistance_matrix, **sim_ctes[num_simulation])
        except Warning:
            filename = simulation_path + f'Figures/Configuration_Set_{voltage}_null_resistance.pkl'
            RepresentateState(resistance_matrix,
                              simulation_path + f'Figures/PS_resistance_matrix_{num_simulation+1}.png')
            print("Null resistance matrix in ", filename)
            with open(filename, 'wb') as f:
                pickle.dump({"actual_state": ac, "resistance_matrix": resistance_matrix}, f)
    else:
        # Cambio el valor de gamma para favorer la generación de vacantes
        sim_ctes[num_simulation]['gamma'] = str(float(sim_ctes[num_simulation]['gamma']) / 1)
        mean_field = np.mean(E_field_vector)
        # Si no ha percolado uso la corriente de Poole-Frenkel
        current = CurentSolver.Poole_Frenkel(temperatura, mean_field, **sim_ctes[num_simulation])*(device_size)

    # Obtengo los valores del campo eléctrico y la temperatura
    E_field = SimpleElectricField(voltage, device_size)
    temperatura = Temperature_Joule(voltage, current, T_0, **sim_ctes[num_simulation])
    # Genero el vector campo eléctrico
    for i in range(0, actual_state.shape[0]):
        E_field_vector[i] = GapElectricField(voltage, i, actual_state, **sim_parmtrs[num_simulation])
    # Calculo la probabilidad de generación o recombinación para ello recorro toda la matriz
    for i in range(x_size):
        prob_generacion = Generation.Generate(
            paso_temporal, E_field_vector[i], temperatura, **sim_ctes[num_simulation])
        for j in range(y_size):
            if actual_state[i, j] == 0:
                random_number = np.random.rand()
                if random_number < prob_generacion:
                    actual_state[i, j] = 1  # Generación de una vacante
    # Tiempo total de la simulacion
    tiempo_total = simulation_time + tiempo_pp_set
    data_sp_set[k] = np.array([tiempo_total, voltage, current, temperatura, E_field, np.mean(E_field_vector), 0])
    # Guardo el estado actual CADA paso_guardar PASOS MONTECARLO
    if k % paso_guardar == 0:
        config_matrix_sp_set[int(k / paso_guardar) - 1] = actual_state
# endregion

# region Guardar datos de la segunda parte del set

if guardar_datos:
    # Cuando acaba la simulacion guardo las matrices de configuraciones y oxigenos
    with open(set_simulation_path + f'Configurations_sp_set_{num_simulation+1}.pkl', 'wb') as f:
        pickle.dump(config_matrix_sp_set, f)

# Guardo el estado final de la simulación
with open(set_simulation_path + f'Last_Configuration_sp_set_{num_simulation+1}.pkl', 'wb') as f:
    pickle.dump(actual_state, f)

np.savetxt(set_simulation_path + f'Resultados_sp_set_{num_simulation +1}.csv',
           data_sp_set, header=header_files, delimiter=',')
# endregion

# region Región de la primera parte del reset

# Estado inicial de la simulación reset para las vacantes
with open(set_simulation_path + f'Last_Configuration_sp_set_{num_simulation+1}.pkl', 'rb') as file:
    initial_configuration = pickle.load(file)

# Como los voltajes no son simétricos, vuelvo a emplear el voltaje máximo de la simulación
num_pasos = int(sim_parmtrs[num_simulation]['num_pasos'])

# el pp significa primera parte del reset (reset primera parte)
data_pp_reset = np.zeros((num_pasos, colunm_number))

# Defino las matrices donde guardo las configuración del sistema y la de los oxígenos
config_matrix_pp_reset = np.zeros((int((num_pasos / paso_guardar)), x_size, y_size))
oxygen_matrix_pp_reset = np.zeros((int((num_pasos / paso_guardar)), x_size, y_size))

# Creo el vector de diferencias de potencial
vector_ddp = np.arange(0.000, -(voltaje_max_simulation + paso_potencial), -paso_potencial)

# Estado iniciales de la simulación para el reset
initial_configuration_reset = actual_state
initial_oxygen_reset = oxygen_state

# Vuelvo a definir el vector de resistencia total
resistencia = np.zeros(num_pasos+1)


RepresentateState(initial_configuration_reset, simulation_path +
                  f'Figures/Initial_pp_reset_configuration_{num_simulation+1}.png')

RepresentateState(initial_configuration_reset, 'C:/Users/Usuario/Documents/GitHub/RRAM_Simulation/Estados_antes_reset/' +
                  f'Initial_pp_reset_configuration_{num_simulation+1}.png')

print(f"\n Comienza la primera parte del reset")

# Durante el reset la generación debe ser más baja por lo que cambio el valor de la constante gamma para desfavaorecer la generación
sim_ctes[num_simulation]['gamma'] = str(float(sim_ctes[num_simulation]['gamma']) / 5)

# Ciclo para la primera parte del reset
# for k in tqdm(range(0, num_pasos)):
for k in (range(0, num_pasos)):
    # Actualizo el tiempo de simulación
    simulation_time = paso_temporal * k
    # Actualizo el voltaje
    voltage = vector_ddp[k]

    # Si se llena el 90 del espacio de la matriz salto a otra simulación
    if np.sum(actual_state) > int(0.85*num_max_vacantes):
        # Represento la temperatura
        save_path = simulation_path + f'Figures/LLENADO_temperature_pp_set_{num_simulation+1}'
        RepresentateState(actual_state, simulation_path +
                          f'Figures/LLENADO_configuration_pp_set_{num_simulation+1}.png')

        # region representar temperatura
        # Crear un array de ejemplo
        data_pp_set[k-1:] = np.nan      # Añadir valores nulos a partir de la fila k

        # Eliminar filas con valores nulos
        data_pp_set = data_pp_set[~np.isnan(data_pp_set).any(axis=1)]

        # Obtengo los valores de tempertura voltaje
        datos_temperatura = data_pp_set[:, 3]
        datos_voltage = data_pp_set[:, 1]

        # Represento la temperatura
        fig, axes = plt.subplots()

        pplt.config_ax(axes)
        pplt.config_ax(axes)

        axes.set_xlabel('Voltaje [V]')
        axes.set_ylabel('Temperatura [K]')
        global_tittle = 'Temperatura vs Voltaje'
        axes.set_title(global_tittle, fontsize=18, pad=15)

        axes.plot(datos_voltage, datos_temperatura, color='blue', label='Temperatura [K]')
        fig.savefig(save_path + '.pdf', bbox_inches='tight')
        # endregion

        raise exceptions.MaxVacantesExcception()

    # Obtengo la corrriente, antes decido cual usar comprobando si ha percolado o no
    if Percolation.is_path(actual_state):
        # Obtengo los caminos de percolación
        ac = actual_state.copy()
        resistance_matrix = findpath.find_path(ac)
        # Si ha percolado uso la corriente de Ohm
        try:
            current, resistencia[k] = CurentSolver.OmhCurrent(
                voltage, resistance_matrix, **sim_ctes[num_simulation])
            current = abs(current)
        except Warning:
            filename = reset_simulation_path + f'Configuration_pp_reset_{voltage}_null_resistance.pkl'
            print("Null resistance matrix in ", filename)
            RepresentateState(resistance_matrix, simulation_path +
                              f'Figures/NULL_resistance_pp_reset_{num_simulation+1}.png')
            with open(filename, 'wb') as f:
                pickle.dump({"actual_state": ac, "resistance_matrix": resistance_matrix}, f)
    else:
        # Si no ha percolado uso la corriente de Poole-Frenkel
        current = abs(CurentSolver.Poole_Frenkel(temperatura, np.mean(
            E_field_vector), **sim_ctes[num_simulation])*(device_size))

    # Obtengo los valores del campo eléctrico y la temperatura
    E_field = abs(SimpleElectricField(voltage, device_size))
    temperatura = Temperature_Joule(voltage, current, T_0, **sim_ctes[num_simulation])

    # Genero el vector campo eléctrico
    for i in range(0, actual_state.shape[0]):
        E_field_vector[i] = abs(GapElectricField(voltage, i, actual_state, **sim_parmtrs[num_simulation]))
    # Calculo la probabilidad de generación o recombinación para ello recorro toda la matriz
    for i in range(x_size):
        prob_generacion = Generation.Generate(
            paso_temporal, E_field_vector[i], temperatura, **sim_ctes[num_simulation])
        for j in range(y_size):
            if actual_state[i, j] == 0:
                random_number = np.random.rand()
                if random_number < prob_generacion:
                    actual_state[i, j] = 1  # Generación de una vacante

    # Genero los oxígenos
    if abs(voltage) > 0.5:
        oxygen_state = Recombination.Generate_Oxigen(oxygen_state, 1)

    if abs(voltage) > 0.9:
        oxygen_state = Recombination.Generate_Oxigen(oxygen_state, 5)

    # Muevo los oxígenos
    oxygen_state, velocidad, desplazamiento = Recombination.Move_OxygenIons(
        paso_temporal, oxygen_state, temperatura, E_field, atom_size, **sim_ctes[num_simulation])

    # Obtengo la nueva configuración
    actual_state, oxygen_state, pro_recombination = Recombination.Recombine(
        actual_state, oxygen_state, paso_temporal, velocidad, temperatura, **sim_ctes[num_simulation])

    # Tiempo total de la simulacion
    tiempo_total = simulation_time + 2 * tiempo_pp_set
    data_pp_reset[k] = np.array([tiempo_total, voltage, current, temperatura,
                                E_field, np.mean(E_field_vector), desplazamiento])
    # Guardo el estado actual CADA paso_guardar PASOS MONTECARLO
    if k % paso_guardar == 0:
        config_matrix_pp_reset[int(k / paso_guardar) - 1] = actual_state
        oxygen_matrix_pp_reset[int(k / paso_guardar) - 1] = oxygen_state
# endregion

# region Guardar datos de la primera parte del reset
if guardar_datos:
    # Cuando acaba la simulacion guardo las matrices de configuraciones y oxigenos
    with open(reset_simulation_path + f'Configurations_pp_reset_{num_simulation+1}.pkl', 'wb') as f:
        pickle.dump(config_matrix_pp_reset, f)

    with open(reset_simulation_path + f'Oxygen_pp_reset_{num_simulation+1}.pkl', 'wb') as f:
        pickle.dump(oxygen_matrix_pp_reset, f)

# Estado inicial de la simulación reset segunda parte para las vacantes
with open(reset_simulation_path + f'Last_Configuration_pp_reset_{num_simulation+1}.pkl', 'wb') as file:
    pickle.dump(actual_state, file)

# Estado inicial para el reset segunda parte de los oxígenos
with open(reset_simulation_path + f'Last_Oxygen_pp_reset_{num_simulation+1}.pkl', 'wb') as file:
    pickle.dump(oxygen_state, file)

np.savetxt(reset_simulation_path +
           f'resultados_pp_reset_{num_simulation +1}.csv', data_pp_reset, header=header_files, delimiter=',')
tiempo_pp_reset = simulation_time
# endregion


# region Región de la segunda parte del reset
print(f"\n Comienza la segunda parte del reset")

# Estado inicial de la simulación reset para las vacantes
with open(reset_simulation_path + f'Last_Configuration_pp_reset_{num_simulation+1}.pkl', 'rb') as file:
    initial_configuration = pickle.load(file)

# Estado inicial para el reset de los oxígenos
with open(reset_simulation_path + f'Last_Oxygen_pp_reset_{num_simulation+1}.pkl', 'rb') as file:
    initial_oxygen_reset = pickle.load(file)

# Número de pasos total de la simlación
num_pasos = int(sim_parmtrs[num_simulation]['num_pasos'])

# el sp significa segunda parte del reset (reset primera parte)
data_sp_reset = np.zeros((num_pasos, colunm_number))

# Defino las matrices donde guardo las configuración del sistema y la de los oxígenos
config_matrix_sp_reset = np.zeros((int((num_pasos + 1 / paso_guardar)), x_size, y_size))
oxygen_matrix_sp_reset = np.zeros((int((num_pasos + 1 / paso_guardar)), x_size, y_size))

# Creo el vector de diferencias de potencial
vector_ddp = np.arange(-voltaje_max_simulation, 0.000 + paso_potencial, paso_potencial)

# Estado iniciales de la simulación para la segunda parte del reset
# initial_configuration_reset = actual_state
# initial_oxygen_reset = oxygen_state

RepresentateState(actual_state, simulation_path + f'Figures/Initial_configuration_sp_reset_{num_simulation+1}.png')
RepresentateState(oxygen_state, simulation_path +
                  f'Figures/Initial_oxygen_sp_reset_{num_simulation+1}.png', color=(0.878, 0.227, 0.370))

sim_ctes[num_simulation]['gamma'] = str(float(sim_ctes[num_simulation]['gamma']) / 5)

# Ciclo para la segunda parte del reset
# for k in tqdm(range(0, num_pasos)):  # son num_pasos + 1 iteraciones
for k in (range(0, num_pasos)):  # son num_pasos + 1 iteraciones
    # Actualizo el tiempo de simulación
    simulation_time = paso_temporal * k

    # Actualizo el voltaje
    voltage = vector_ddp[k]

    # Obtengo la corrriente, antes decido cual usar comprobando si ha percolado o no
    if Percolation.is_path(actual_state):
        # Obtengo los caminos de percolación
        ac = actual_state.copy()
        resistance_matrix = findpath.find_path(ac)
        # Si ha percolado uso la corriente de Ohm
        try:
            current, resistencia[k] = CurentSolver.OmhCurrent(
                voltage, resistance_matrix, **sim_ctes[num_simulation])
            current = abs(current)
        except Warning:
            filename = reset_simulation_path + f'Configuration_sp_reset_{voltage}_null_resistance.pkl'
            print("Null resistance matrix in ", filename)
            RepresentateState(resistance_matrix, simulation_path +
                              f'Figures/PR_resistance_matrix_{num_simulation+1}.png')
            with open(filename, 'wb') as f:
                pickle.dump({"actual_state": ac, "resistance_matrix": resistance_matrix}, f)
    else:
        # Si no ha percolado uso la corriente de Poole-Frenkel
        current = abs(CurentSolver.Poole_Frenkel(temperatura, np.mean(
            E_field_vector), **sim_ctes[num_simulation])*(device_size))
    # Obtengo los valores del campo eléctrico y la temperatura
    E_field = abs(SimpleElectricField(voltage, device_size))
    temperatura = Temperature_Joule(voltage, current, T_0, **sim_ctes[num_simulation])

    # Genero el vector campo eléctrico
    for i in range(0, actual_state.shape[0]):
        E_field_vector[i] = abs(GapElectricField(voltage, i, actual_state, **sim_parmtrs[num_simulation]))

    # Calculo la probabilidad de generación o recombinación para ello recorro toda la matriz
    for i in range(x_size):
        prob_generacion = Generation.Generate(
            paso_temporal, E_field_vector[i], temperatura, **sim_ctes[num_simulation])
        for j in range(y_size):
            if actual_state[i, j] == 0:
                random_number = np.random.rand()
                if random_number < prob_generacion:
                    actual_state[i, j] = 1  # Generación de una vacante

    if abs(voltage) > 0.5:
        # Genero los oxígenos
        oxygen_state = Recombination.Generate_Oxigen(oxygen_state, 1)

    # Muevo los oxígenos
    oxygen_state, velocidad, desplazamiento = Recombination.Move_OxygenIons(
        paso_temporal, oxygen_state, temperatura, E_field, atom_size, **sim_ctes[num_simulation])

    # Obtengo la nueva configuración
    actual_state, oxygen_state, pro_recombination = Recombination.Recombine(
        actual_state, oxygen_state, paso_temporal, velocidad, temperatura, **sim_ctes[num_simulation])

    # Tiempo total de la simulacion
    tiempo_total = simulation_time + 2 * tiempo_pp_set + tiempo_pp_reset
    data_sp_reset[k] = np.array([tiempo_total, voltage, current, temperatura,
                                 E_field, np.mean(E_field_vector), desplazamiento])
    # Guardo el estado actual CADA paso_guardar PASOS MONTECARLO
    if k % paso_guardar == 0:
        config_matrix_sp_reset[int(k / paso_guardar) - 1] = actual_state
        oxygen_matrix_sp_reset[int(k / paso_guardar) - 1] = oxygen_state

# endregion

# region Guardar datos del reset segunda parte
if guardar_datos:
    # Cuando acaba la simulacion guardo las matrices de configuraciones y oxigenos
    with open(reset_simulation_path + f'Configurations_sp_reset_{num_simulation+1}.pkl', 'wb') as f:
        pickle.dump(config_matrix_sp_reset, f)

    with open(reset_simulation_path + f'Oxygen_sp_reset_{num_simulation+1}.pkl', 'wb') as f:
        pickle.dump(oxygen_matrix_sp_reset, f)

np.savetxt(reset_simulation_path +
           f'resultados_sp_reset_{num_simulation +1}.csv', data_sp_reset, header=header_files, delimiter=',')

RepresentateState(actual_state, simulation_path + f'Figures/final_Configuration_sp_reset_{num_simulation+1}.png')
RepresentateState(oxygen_state, simulation_path +
                  f'Figures/final_Oxygen_sp_reset_{num_simulation+1}.png', color=(0.878, 0.227, 0.370))
# endregion

# region Unir todos los datos en un solo archivo csv TODO: necesita revision cuando se unen los datos
df_pset = pd.read_csv(set_simulation_path + f'Resultados_pp_set_{num_simulation +1}.csv')
df_sset = pd.read_csv(set_simulation_path + f'Resultados_sp_set_{num_simulation +1}.csv')
df_preset = pd.read_csv(reset_simulation_path + f'resultados_pp_reset_{num_simulation +1}.csv')
df_sreset = pd.read_csv(reset_simulation_path + f'resultados_sp_reset_{num_simulation +1}.csv')

# Concatenar los DataFrames sin duplicar el encabezado
data_frame_simulation = pd.concat([df_pset, df_sset, df_preset, df_sreset])

# Guardar el DataFrame combinado en un archivo CSV
data_frame_simulation.to_csv(f'Results/Datos_simulacion_completa_{num_simulation +1}.csv', index=False)
print("Todos los archivos CSV se han combinado y guardado exitosamente.")
# endregion

# region Representar datos
data_path_pp_set = set_simulation_path + f'resultados_pp_set_{num_simulation +1}.csv'
data_path_sp_set = set_simulation_path + f'resultados_sp_set_{num_simulation +1}.csv'
data_path_pp_reset = reset_simulation_path + f'resultados_pp_reset_{num_simulation +1}.csv'
data_path_sp_reset = reset_simulation_path + f'resultados_sp_reset_{num_simulation +1}.csv'

df_pset = pd.read_csv(data_path_pp_set, dtype=float)
df_sset = pd.read_csv(data_path_sp_set, dtype=float)
df_preset = pd.read_csv(data_path_pp_reset, dtype=float)
df_sreset = pd.read_csv(data_path_sp_reset, dtype=float)

global_tittle = 'Intensidad vs Voltaje'

save_path = simulation_path + f'Figures/Intensidad_Voltaje_simulation_{num_simulation+1}'

i_ps = np.array(df_pset['Intensidad [A]'])
i_ss = np.array(df_sset['Intensidad [A]'])
i_pr = np.array(df_preset['Intensidad [A]'])
i_sr = np.array(df_sreset['Intensidad [A]'])
v_ps = np.array(df_pset['Voltaje [V]'])
v_ss = np.array(df_sset['Voltaje [V]'])
v_pr = np.array(df_preset['Voltaje [V]'])
v_sr = np.array(df_sreset['Voltaje [V]'])

fig, axes = plt.subplots()

pplt.config_ax(axes)

axes.set_xlabel('Voltaje [V]')
axes.set_ylabel('Intensidad [A]')
axes.set_yscale('log')

axes.set_title(global_tittle, fontsize=18, pad=15)

axes.scatter(v_ps, i_ps, color='blue', s=0.2, label='Primera parte Set')
axes.scatter(v_ss, i_ss, color='red', s=0.2, label='Segunda parte Set')
axes.scatter(v_pr, i_pr, color='green', s=0.2, label='Primera parte Reset')
axes.scatter(v_sr, i_sr, color='pink', s=0.2, label='Segunda parte Reset')

plt.legend()

# Ruta proporcionada
ruta_exp_data = ruta_raiz + 'Datos_Experimentales/Ciclos_Experimentales'
ruta_archivo_set = ruta_exp_data + '/Cycle_p_1000.txt'
ruta_archivo_reset = ruta_exp_data + '/Cycle_n_1000.txt'

# Leer datos del archivo
data_set = np.loadtxt(ruta_archivo_set)
data_reset = np.loadtxt(ruta_archivo_reset)
# Asumimos que los datos están en dos columnas: x e y

x_set = data_set[:, 0]
y_set = data_set[:, 1]

# Data reset
x_reset = data_reset[:, 0]
y_reset = abs(data_reset[:, 1])

# Añado los valores de las variables que estoy cambiando, para eso tengo q ver dentro de la carpeta de init_data el nombre de cada documento, si el nombre conindice con una variable del diccionario, añado el valor que está tomando en la simulación en la representación

# Leer los valores de las variables desde los archivos en Initial_data
init_data_path = 'Initial_data'
variables = {}

for filename in os.listdir(init_data_path):
    if filename.endswith('.pkl'):
        variable_name = filename.split('.')[0]
        with open(os.path.join(init_data_path, filename), 'rb') as f:
            variables[variable_name] = pickle.load(f)

# Crear el subtítulo con los valores de las variables
espacios = ' ' * 3  # Define el número de espacios que deseas añadir

# Construye la lista de subtítulos con espacios y saltos de línea cada tres valores
subtitles = []
for i, (variable_name, value) in enumerate(variables.items()):
    subtitles.append(f'{variable_name} = {value[num_simulation]}')
    if (i + 1) % 3 == 0:
        subtitles.append('\n')
    else:
        subtitles.append(espacios)

# Une los subtítulos en una sola cadena
subtitle = ''.join(subtitles).strip()
fig.suptitle(subtitle, fontsize=11, y=1.05)  # Ajusta el valor de y según sea necesario

# Crear la gráfica scatter
axes.plot(x_set, y_set, 'black', label='Set experimental')
axes.plot(x_reset, y_reset, 'black', label='Reset experimental')

# fig.savefig(save_path + '.pdf', bbox_inches='tight')
# fig.savefig(figures_path + f'/Intensidad_Voltaje_simulation_{num_simulation + 1}.pdf', bbox_inches='tight')
fig.savefig(figures_path + f'/Intensidad_Voltaje_simulation_{num_simulation + 1}.png', bbox_inches='tight')

# plt.show()
# endregion
