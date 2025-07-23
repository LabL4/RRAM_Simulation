from RRAM import Plot_PostProcess as pplt
import matplotlib.pyplot as plt
from RRAM import Recombination
from numpy import save, size # type: ignore
from RRAM import exceptions
from tqdm import tqdm
import time as time
import pandas as pd
from RRAM import *
import logging
import pickle
import shutil
import sys
import os

import warnings
warnings.filterwarnings("error")

# region Definición de valores iniciales y constantes de la simulación

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

ruta_raiz = os.getcwd()+"/"
# ruta_raiz = 'C:/Users/jimdo/Documents/GitHub/RRAM_Simulation/' # Pc personal
# ruta_raiz = 'C:/Users/Usuario/Documents/GitHub/RRAM_Simulation/'
# ruta_raiz = '/Users/antonio_lopez_torres/Documents/GitHub/RRAM_Simulation/'  # Ruta en el mac
sys.path.append(ruta_raiz)


def config_ax(ax):
    # ax.grid(which='major', color='#DDDDDD', linewidth=0.8, zorder=-1)
    # ax.grid(which='minor', color='#DEDEDE', linestyle=':', linewidth=0.5, zorder=-1)
    ax.minorticks_on()
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)


def setup_plt(plt, latex=True, scaling=1):
    plt.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "text.usetex": latex,
            "font.family": "fourier",
            "text.latex.preamble": "\n".join([
                r"\usepackage[utf8]{inputenc}",
                r"\usepackage[T1]{fontenc}",
                r"\usepackage{siunitx}",
            ])
        }
    )

    SMALL_SIZE = 8 * scaling
    MEDIUM_SIZE = 10 * scaling
    BIGGER_SIZE = 11 * scaling

    plt.rc('font', size=SMALL_SIZE)
    plt.rc('axes', titlesize=SMALL_SIZE)
    plt.rc('axes', labelsize=MEDIUM_SIZE)
    plt.rc('xtick', labelsize=SMALL_SIZE)
    plt.rc('ytick', labelsize=SMALL_SIZE)
    plt.rc('legend', fontsize=SMALL_SIZE)
    plt.rc('figure', titlesize=BIGGER_SIZE)
    plt.rc('axes', titlesize=BIGGER_SIZE*1.05)


setup_plt(plt, latex=True, scaling=2)

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
header_files = 'Tiempo simulacion [s],Voltaje [V],Intensidad [A],Temperatura [K],Campo Simple [V/m],Campo Gap medio [V/m],Velocidad [m/s],Densidad Filamento'

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
RepresentateState(actual_state, 0, 0, simulation_path + 'Figures/initial_pp_set_' + str(num_simulation))

# Defino las matrices donde guardo las configuración del sistema y la de los oxígenos
config_matrix_pp_set = np.zeros((int((num_pasos / paso_guardar)), x_size, y_size))

# Defino el paso temporal
paso_temporal = total_simulation_time / num_pasos
paso_potencial = voltaje_max_simulation / num_pasos
factor_generacion = float(sim_ctes[num_simulation]['factor_generacion'])
print("El paso temporal es: ", paso_temporal)
print("El paso del potencial es: ", paso_potencial)
print("El factor de generación es: ", factor_generacion)
print("\nEl valor de la resistencia de cada casilla es: ", sim_ctes[num_simulation]['ohm_resistence'])
print("El valor de gamma es: ", sim_ctes[num_simulation]['gamma'])
print("El valor de resistencia termica no percola es: ", sim_ctes[num_simulation]['r_termica_no_percola'])
print("El valor de resistencia termica percola es: ", sim_ctes[num_simulation]['r_termica_percola'])
print("El valor de I_0 es: ", sim_ctes[num_simulation]['I_0'])
print("El valor de E_a es: ", sim_ctes[num_simulation]['activation_energy'])
print("El valor de E_r es: ", sim_ctes[num_simulation]['recombination_energy'])
print("El valor del factor de generacion es: ", sim_ctes[num_simulation]['factor_generacion'])


# Creo el vector de diferencias de potencial
vector_ddp = np.arange(0.000, voltaje_max_simulation + paso_potencial, paso_potencial)

# Creo el vector de datos como una matriz de num_pasos filas y las columnas necesarias
colunm_number = 8
data_pp_set = np.zeros((num_pasos, colunm_number))

# Inicializo el campo eléctrico
E_field_vector = np.zeros((actual_state.shape[0]))
num_vacantes = np.zeros(num_pasos+1)
resistencia = np.zeros(num_pasos+1)
array_filament_density = np.zeros(num_pasos+1)

g_valor_list = []

voltage_vector = np.zeros(num_pasos+1)

T_0 = float(sim_parmtrs[num_simulation]['init_temp'])

sistema_percola = False
num_max_vacantes = (device_size/atom_size)**2
paso_guardar_2 = 100
# endregion

# region primera parte del set
print(f"\nComienza la primera parte del set")
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
        print("\nSe ha llenado el 90% del espacio de la matriz en la iteración: ",
              k, " que corresponde al voltaje: ", voltage)

        # Verifica si el sistema ha percolado    
        if not sistema_percola:
            raise exceptions.NoPercolationException()

        k_ruptura = k

        voltaje_max_set = vector_ddp[k]
        config_matrix_recortada = config_matrix_pp_set[k, :, :]
        tiempo_pp_set = paso_temporal * (k - 1)  # Le quitamos un paso porque se ha superado el voltaje de ruptura
        resistencia_copy = resistencia.copy()

        print("Voltaje final set", voltaje_max_set, 'en el tiempo ', tiempo_pp_set, 'en la iteración ', k_ruptura, "\n")

        # Crear un array de ejemplo
        data_pp_set[k-1:] = np.nan      # Añadir valores nulos a partir de la fila k
        num_vacantes[k:] = np.nan       # Añadir valores nulos a partir de la fila k
        resistencia[k:] = np.nan        # Añadir valores nulos a partir de la fila k
        voltage_vector[k:] = np.nan  # Añadir valores nulos a partir de la fila k

        # Eliminar filas con valores nulos
        data_pp_set = data_pp_set[~np.isnan(data_pp_set).any(axis=1)]
        num_vacantes = num_vacantes[~np.isnan(num_vacantes)]
        resistencia = resistencia[~np.isnan(resistencia)]
        voltage_vector = voltage_vector[~np.isnan(voltage_vector)]
        
        # RepresentateState(resistance_matrix,k,paso_potencial, simulation_path + f'Figures/final_pp_set_resistance_{num_simulation+1}.png')
        break
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

        # RepresentateState(resistance_matrix,k,paso_potencial, simulation_path + f'Figures/final_pp_set_resistance_{num_simulation+1}.png')
        break
    
    # Obtengo la corrriente, antes decido cual usar comprobando si ha percolado o no
    if Percolation.is_path(actual_state):
        if sistema_percola is False:
        
            voltaje_percolacion = voltage   # Guardo el voltaje de percolación

            print("\nEl sistema ha percolado en la iteración: ", k, " que corresponde con el voltaje: ", voltaje_percolacion)
            
            # Cambio la probabilidad de generación de vacantes para controlar la percolación
            sim_ctes[num_simulation]['gamma'] = str(float(sim_ctes[num_simulation]['gamma']) / factor_generacion)
            print("Nueva gamma cuando percola el sistema: ", sim_ctes[num_simulation]['gamma'])
            
            num_divisiones = round(((voltaje_final_set - 0.2) - voltaje_percolacion) / paso_potencial)

            # Generar valores exponenciales
            valor_resistencia_celda = np.exp(np.linspace(np.log(13), np.log(0.5), num=num_divisiones))
            # print("El vector de resistencias es: ", valor_resistencia_celda)
            
            # Generar ruido aleatorio distinto para cada elemento
            # loc es la media, scale es la desviación estándar
            ruido = np.exp(np.random.normal(loc=-0.1, scale=0.25, size=num_divisiones))

            # Añadir el ruido a cada término del array
            valor_resistencia_celda += ruido
            # print("El vector de resistencias es: ", valor_resistencia_celda)
            indice_resistencia = 0  # Indice de la resistencia
        sistema_percola = True



        # Copio el estado actual
        ac = actual_state.copy()
        resistance_matrix = findpath.find_path(ac)
        densidad_filamento = np.sum(resistance_matrix) / (x_size * y_size)

        if voltaje_percolacion < voltage < (voltaje_final_set - 0.25):
            sim_ctes[num_simulation]['ohm_resistence'] = valor_resistencia_celda[indice_resistencia]
            indice_resistencia = indice_resistencia + 1
            intensidad_lineal = current
        else:
            sim_ctes[num_simulation]['ohm_resistence'] = 1.5

        # - (current * Resistencia_serie) # esto hay q quitarlo no se  pero como tengo cosas referidas a esa variable es mas delicado
        voltage_RRAM = voltage
        # Si ha percolado uso la corriente de Ohm
        try:
            current, resistencia[k] = CurentSolver.OmhCurrent(voltage, resistance_matrix, **sim_ctes[num_simulation])# type: ignore

        except Warning:
            filename = simulation_path + f'Null_Resistance/Configuration_Set_{voltage}_null_resistance.pkl'
            print("Null resistance matrix in ", filename)
            RepresentateState(resistance_matrix, k, paso_potencial, simulation_path +
                              f'Figures/Null_Resistance/NULL_resistance_matrix_pp_set_{num_simulation+1}.png')
            with open(filename, 'wb') as f:
                pickle.dump({"actual_state": ac, "resistance_matrix": resistance_matrix}, f)

    else:
        sistema_percola = False

        # Cambio el valor de gamma para favorer la generación de vacantes        # Si no ha percolado uso la corriente de Poole-Frenkel
        resistencia[k] = 0

        mean_field = np.mean(E_field_vector)
        current = CurentSolver.Poole_Frenkel(temperatura, mean_field, **sim_ctes[num_simulation])*(device_size)# type: ignore
        densidad_filamento = 0

    # Obtengo los valores del campo eléctrico y la temperatura
    # voltage_RRAM = voltage - (current * Resistencia_serie)
    E_field = SimpleElectricField(voltage, device_size)
    temperatura = Temperature_Joule(voltage, current, sistema_percola, T_0, ** sim_ctes[num_simulation])

    # Genero el vector campo eléctrico
    for i in range(0, actual_state.shape[0]):
        E_field_vector[i] = GapElectricField(voltage, i, actual_state, **sim_parmtrs[num_simulation])

    # Calculo la probabilidad de generación o recombinación para ello recorro toda la matriz
    for i in range(x_size):
        prob_generacion = Generation.Generate(paso_temporal, E_field_vector[i], temperatura, **sim_ctes[num_simulation])
        for j in range(y_size):
            if actual_state[i, j] == 0:
                if np.sum(actual_state) < int(0.5*num_max_vacantes): # antes era un 0.6
                    # Compruebo si tiene una vacante cerca
                    if Generation.vecinos_horizontales(actual_state, i, j): #np.sum(actual_state[i-1:i+1, j-1:j+1]) > 0:
                        prob_generacion = prob_generacion * 1.05
                    else:
                        prob_generacion = prob_generacion * 0.9
                else:
                    prob_generacion = 0 # LO hago para que no se generen más vacantes y no se llene el sistema
                    
                random_number = np.random.rand()
                if random_number < prob_generacion:
                    actual_state[i, j] = 1  # Generación de una vacante
    
    if 0.45 < voltage < (voltaje_final_set - 0.2):
        # print("El resultado de la division es: ", k % paso_guardar_2)
        if k % paso_guardar_2 == 0:
            data_pp_set[k-1] = np.array([simulation_time, voltage, current, temperatura,E_field, np.mean(E_field_vector), 0, densidad_filamento])
            g_valor_list.append(Generation.evalutate_g(actual_state, size_grid=40))


    else:
        data_pp_set[k-1] = np.array([simulation_time, voltage, current, temperatura, E_field, np.mean(E_field_vector), 0, densidad_filamento])
        g_valor_list.append(Generation.evalutate_g(actual_state, size_grid=40))

    # Guardo el estado actual CADA paso_guardar PASOS MONTECARLO
    if k % paso_guardar == 0:
        config_matrix_pp_set[int(k / paso_guardar) - 1] = actual_state
    
# endregion

# region Guardar datos del Primera parte del set
data_pp_set = data_pp_set[~np.all(data_pp_set == 0, axis=1)]  # Elimino los valores nulos

# Si está activada la opción de guardar los datos:
if guardar_datos:
    # Cuando acaba la simulacion guardo las matrices de configuración
    with open(set_simulation_path + f'Configurations_pp_set_{num_simulation+1}.pkl', 'wb') as f:
        pickle.dump(config_matrix_pp_set, f)

# Cuando acaba la simulacion guardo el estado final de configuracion
with open(set_simulation_path + f'Last_Configuration_pp_set_{num_simulation+1}.pkl', 'wb') as f:
    pickle.dump(actual_state, f)

np.savetxt(set_simulation_path + f'resultados_pp_set_{num_simulation+1}.csv', data_pp_set, header=header_files, delimiter=',')
print("El fichero de resultados_pp_set contiene ", data_pp_set.shape[0], ' filas y ', data_pp_set.shape[1], ' columnas')

g_pp_set = np.array(g_valor_list[1:])

np.savetxt(set_simulation_path + f'g_pp_set_{num_simulation+1}.txt', g_pp_set, delimiter=',', fmt='%.0f')
print("El g en el pp set contiene ", g_pp_set.shape[0], ' filas y ', g_pp_set.shape[1], ' columnas')

# Guardo las vacantes generadas en el forming
with open(set_simulation_path + f"Vacantes_resistencia_{num_simulation+1}.txt", "w") as f:
    for v1, v2, v3, v4 in zip(data_pp_set[:, 0], data_pp_set[:, 1],  resistencia, num_vacantes):
        f.write(f"{v1} {v2} {v3} {v4}\n")

# Leer el contenido del archivo TXT
with open(set_simulation_path + f"Vacantes_resistencia_{num_simulation+1}.txt", 'r') as file:
    lines = file.readlines()

header_files_extra = 'Tiempo simulacion [s],Voltaje [V],Resistencia [Ohm],Numero de vacantes\n'

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
g_sp_set = np.zeros((num_pasos, x_size))

print("Voltaje inicial sp set", voltaje_max_set)

# Estado iniciales de la simulación para el reset
actual_state = initial_configuration
RepresentateState(actual_state, k, paso_potencial, simulation_path + f'Figures/Final_state_pp_set_{num_simulation+1}.png')

sim_ctes[num_simulation]['gamma'] = str(float(sim_ctes[num_simulation]['gamma']) / factor_generacion)
print("El valor de gamma para la sp set es: ", sim_ctes[num_simulation]['gamma'])

print(f"\n Comienza la segunda parte del set")
# Ciclo para la segunda parte del set
# for k in tqdm(range(0, num_pasos)):
for k in (range(0, num_pasos)):
    # Actualizo el tiempo de simulación
    simulation_time = paso_temporal * k
    # Actualizo el voltaje
    voltage = vector_ddp[k]

    num_vacantes[k] = np.sum(actual_state)
    # Obtengo la corrriente, antes decido cual usar comprobando si ha percolado o no

    # Si se llena el 90 del espacio de la matriz salto a otra simulación
    if num_vacantes[k] > int(0.9*num_max_vacantes):
        # Represento la temperatura
        save_path = simulation_path + f'Figures/LLENADO_temperature_pp_set_{num_simulation+1}'
        RepresentateState(actual_state, k, paso_potencial, simulation_path + f'Figures/LLENADO_configuration_pp_set_{num_simulation+1}.png')

        # region representar temperatura
        # Crear un array de ejemplo
        data_sp_set[k-1:] = np.nan      # Añadir valores nulos a partir de la fila k

        # Eliminar filas con valores nulos
        data_sp_set = data_sp_set[~np.isnan(data_sp_set).any(axis=1)]

        # Obtengo los valores de tempertura voltaje
        datos_temperatura = data_sp_set[:, 3]
        datos_voltage = data_sp_set[:, 1]

        # Represento la temperatura
        fig, axes = plt.subplots(figsize=(12, 9))

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
        # sim_ctes[num_simulation]['gamma'] = str(float(sim_ctes[num_simulation]['gamma']) / factor_generacion)
        # print("\nEl sistema ha percolado en la iteración: ", k, ' el valor de gamma es: ', sim_ctes[num_simulation]['gamma'])
        ac = actual_state.copy()
        resistance_matrix = findpath.find_path(ac)

        filament_density = np.sum(resistance_matrix) / (x_size * y_size*(0.25)*(0.25))

        # Si ha percolado uso la corriente de Ohm
        voltage_RRAM = voltage  # - (current * Resistencia_serie)
        try:
            current, resistencia[k] = CurentSolver.OmhCurrent(
                voltage_RRAM, resistance_matrix, **sim_ctes[num_simulation])# type: ignore
        except Warning:
            filename = simulation_path + f'Figures/Configuration_Set_{voltage}_null_resistance.pkl'
            RepresentateState(resistance_matrix, k, paso_potencial,
                              simulation_path + f'Figures/PS_resistance_matrix_{num_simulation+1}.png')
            print("Null resistance matrix in ", filename)
            with open(filename, 'wb') as f:
                pickle.dump({"actual_state": ac, "resistance_matrix": resistance_matrix}, f)
    else:
        # Cambio el valor de gamma para favorer la generación de vacantes
        sim_ctes[num_simulation]['gamma'] = str(float(sim_ctes[num_simulation]['gamma']) / 1)
        print("Nueva gamma cuando no ha percolado: ", sim_ctes[num_simulation]['gamma'])
        
        mean_field = np.mean(E_field_vector)
        # Si no ha percolado uso la corriente de Poole-Frenkel
        current = CurentSolver.Poole_Frenkel(temperatura, mean_field, **sim_ctes[num_simulation])*(device_size)# type: ignore
        filament_density = 0

    # Obtengo los valores del campo eléctrico y la temperatura
    E_field = SimpleElectricField(voltage, device_size)
    temperatura = Temperature_Joule(voltage, current, T_0, **sim_ctes[num_simulation])# type: ignore
    # Genero el vector campo eléctrico
    for i in range(0, actual_state.shape[0]):
        E_field_vector[i] = GapElectricField(voltage, i, actual_state, **sim_parmtrs[num_simulation])

    # Calculo la probabilidad de generación o recombinación para ello recorro toda la matriz
    for i in range(x_size):
        prob_generacion = Generation.Generate(
            paso_temporal, E_field_vector[i], temperatura, **sim_ctes[num_simulation])
        for j in range(y_size):
            if actual_state[i, j] == 0:
                if np.sum(actual_state) < int(0.8*num_max_vacantes): # antes era un 0.6
                    # Compruebo si tiene una vacante cerca
                    if Generation.tiene_vecinos(actual_state, i, j): #np.sum(actual_state[i-1:i+1, j-1:j+1]) > 0:
                        prob_generacion = prob_generacion * 1
                else:
                    prob_generacion = 0 # LO hago para que no se generen más vacantes y no se llene el sistema

                # Genero un número aleatorio para decidir si se genera una vacante
                random_number = np.random.rand()

                if random_number < prob_generacion:
                    actual_state[i, j] = 1  # Generación de una vacante

    # Tiempo total de la simulacion
    tiempo_total = simulation_time + tiempo_pp_set
    
    data_sp_set[k] = np.array([tiempo_total, voltage, current, temperatura, E_field,np.mean(E_field_vector), 0, filament_density])
    g_sp_set[k] = Generation.evalutate_g(actual_state, size_grid=40)
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

np.savetxt(set_simulation_path + f'Resultados_sp_set_{num_simulation +1}.csv', data_sp_set, header=header_files, delimiter=',')
print("El fichero de resultados_sp_set contiene ", data_sp_set.shape[0], ' filas y ', data_sp_set.shape[1], ' columnas')

# Guardo los resultados del set completos combinando los resultados de las dos partes
data_sp_set_final = np.concatenate((data_pp_set, data_sp_set), axis=0)
np.savetxt(set_simulation_path + f'Resultados_set_{num_simulation +1}.csv', data_sp_set_final, header=header_files, delimiter=',')
print("El fichero de resultados set contiene ", data_sp_set.shape[0], ' filas y ', data_sp_set.shape[1], ' columnas')

np.savetxt(set_simulation_path + f'g_sp_set_{num_simulation+1}.txt', g_sp_set, delimiter=',', fmt='%.0f')

# Guardo los valores de g del proceso de set, combinando los vectores de g de las dos partes
g_set = np.concatenate((g_pp_set, g_sp_set), axis=0)

print("El g en el set contiene ", g_set.shape[0], ' filas y ', g_set.shape[1], ' columnas')
np.savetxt(set_simulation_path + f'g_set_{num_simulation+1}.txt', g_set, delimiter=',', fmt='%.0f')
print("El fichero de g del set contiene ", g_set.shape[0], ' filas y ', g_set.shape[1], ' columnas')
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
vector_ddp = np.arange(0.000, - (voltaje_max_simulation + paso_potencial), -paso_potencial)

# Estado iniciales de la simulación para el reset
initial_configuration_reset = actual_state
initial_oxygen_reset = oxygen_state

# Vuelvo a definir el vector de resistencia total
resistencia = np.zeros(num_pasos+1)
g_pp_reset = np.zeros((num_pasos, x_size))

RepresentateState(initial_configuration_reset, k, paso_potencial, simulation_path +f'Figures/Final_state_sp_set_{num_simulation+1}.png')


print(f"\n Comienza la primera parte del reset")

# Durante el reset la generación debe ser más baja por lo que cambio el valor de la constante gamma para desfavaorecer la generación
sim_ctes[num_simulation]['gamma'] = str(float(sim_ctes[num_simulation]['gamma']) / 10)  # 5) #antes habia un 3

# Ciclo para la primera parte del reset
# for k in tqdm(range(0, num_pasos)):
for k in (range(0, num_pasos)):
    # Actualizo el tiempo de simulación
    simulation_time = paso_temporal * k
    # Actualizo el voltaje
    voltage = vector_ddp[k]

    # Si se llena el 90 del espacio de la matriz salto a otra simulación
    if np.sum(actual_state) > int(0.9*num_max_vacantes):
        # Represento la temperatura
        save_path = simulation_path + f'Figures/LLENADO_temperature_pp_set_{num_simulation+1}'
        RepresentateState(actual_state, k, paso_potencial, simulation_path +
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

        raise exceptions.MaxVacantesException()

    # Obtengo la corrriente, antes decido cual usar comprobando si ha percolado o no
    if Percolation.is_path(actual_state):
        # Obtengo los caminos de percolación
        ac = actual_state.copy()
        resistance_matrix = findpath.find_path(ac)
        filament_density = np.sum(resistance_matrix) / (x_size * y_size*(0.25)*(0.25))

        # Si ha percolado uso la corriente de Ohm
        try:
            current, resistencia[k] = CurentSolver.OmhCurrent(
                voltage, resistance_matrix, **sim_ctes[num_simulation])# type: ignore
            current = abs(current)
        except Warning:
            filename = reset_simulation_path + f'Configuration_pp_reset_{voltage}_null_resistance.pkl'
            print("Null resistance matrix in ", filename)
            RepresentateState(resistance_matrix, k, paso_potencial, simulation_path +
                              f'Figures/NULL_resistance_pp_reset_{num_simulation+1}.png')
            with open(filename, 'wb') as f:
                pickle.dump({"actual_state": ac, "resistance_matrix": resistance_matrix}, f)
    else:
        # Si no ha percolado uso la corriente de Poole-Frenkel
        current = abs(CurentSolver.Poole_Frenkel(temperatura, np.mean(
            E_field_vector), **sim_ctes[num_simulation])*(device_size))# type: ignore
        filament_density = 0

    # Obtengo los valores del campo eléctrico y la temperatura
    E_field = abs(SimpleElectricField(voltage, device_size))
    temperatura = Temperature_Joule(voltage, current, T_0, **sim_ctes[num_simulation])# type: ignore

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
                                 E_field, np.mean(E_field_vector), desplazamiento, filament_density])

    g_pp_reset[k] = Generation.evalutate_g(actual_state, size_grid=40)
    
    # Guardo el estado actual CADA paso_guardar PASOS MONTECARLO
    if k % 1 == 0:  # Arreglo rapido para q lo guarde siempre
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

RepresentateTwoStates(actual_state, oxygen_state, k, paso_potencial, simulation_path + f'Figures/Final_state_pp_reset_{num_simulation+1}.png')

np.savetxt(reset_simulation_path + f'resultados_pp_reset_{num_simulation +1}.csv', data_pp_reset, header=header_files, delimiter=',')
print("El fichero de resultados_pp_reset contiene ", data_pp_reset.shape[0], ' filas y ', data_pp_reset.shape[1], ' columnas')

np.savetxt(reset_simulation_path + f'g_pp_reset_{num_simulation+1}.txt', g_pp_reset, delimiter=',', fmt='%.0f')
print("El g en el pp reset contiene ", g_pp_reset.shape[0], ' filas y ', g_pp_reset.shape[1], ' columnas')


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


sim_ctes[num_simulation]['gamma'] = str(float(sim_ctes[num_simulation]['gamma']) / 5)

g_sp_reset = np.zeros((num_pasos, x_size))

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
        filament_density = np.sum(resistance_matrix) / (x_size * y_size*(0.25)*(0.25))
        # Si ha percolado uso la corriente de Ohm
        try:
            current, resistencia[k] = CurentSolver.OmhCurrent(
                voltage, resistance_matrix, **sim_ctes[num_simulation])# type: ignore
            current = abs(current)
        except Warning:
            filename = reset_simulation_path + f'Configuration_sp_reset_{voltage}_null_resistance.pkl'
            print("Null resistance matrix in ", filename)
            RepresentateState(resistance_matrix, k, paso_potencial, simulation_path +
                              f'Figures/PR_resistance_matrix_{num_simulation+1}.png')
            with open(filename, 'wb') as f:
                pickle.dump({"actual_state": ac, "resistance_matrix": resistance_matrix}, f)
    else:
        # Si no ha percolado uso la corriente de Poole-Frenkel
        current = abs(CurentSolver.Poole_Frenkel(temperatura, np.mean(
            E_field_vector), **sim_ctes[num_simulation])*(device_size))# type: ignore

        filament_density = 0

    # Obtengo los valores del campo eléctrico y la temperatura
    E_field = abs(SimpleElectricField(voltage, device_size))
    temperatura = Temperature_Joule(voltage, current, T_0, **sim_ctes[num_simulation])# type: ignore

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

    oxygen_state = Recombination.Generate_Oxigen(oxygen_state, 5)

    # Muevo los oxígenos
    oxygen_state, velocidad, desplazamiento = Recombination.Move_OxygenIons(
        paso_temporal, oxygen_state, temperatura, E_field, atom_size, **sim_ctes[num_simulation])

    # Obtengo la nueva configuración
    actual_state, oxygen_state, pro_recombination = Recombination.Recombine(
        actual_state, oxygen_state, paso_temporal, velocidad, temperatura, **sim_ctes[num_simulation])

    # Tiempo total de la simulacion
    tiempo_total = simulation_time + 2 * tiempo_pp_set + tiempo_pp_reset

    data_sp_reset[k] = np.array([tiempo_total, voltage, current, temperatura,
                                 E_field, np.mean(E_field_vector), desplazamiento, filament_density])
    
    g_sp_reset[k] = Generation.evalutate_g(actual_state, size_grid=40)
    
    # Guardo el estado actual CADA paso_guardar PASOS MONTECARLO

    if k % 1 == 0:  # Arreglo rapido para q lo guarde siempre
        config_matrix_sp_reset[int(k / paso_guardar) - 1] = actual_state
        oxygen_matrix_sp_reset[int(k / paso_guardar) - 1] = oxygen_state

# endregion

# region Guardar datos del reset segunda parte
if guardar_datos:
    with open(reset_simulation_path + f'Configurations_sp_reset_{num_simulation+1}.pkl', 'wb') as f:
        pickle.dump(config_matrix_sp_reset, f)

    with open(reset_simulation_path + f'Oxygen_sp_reset_{num_simulation+1}.pkl', 'wb') as f:
        pickle.dump(oxygen_matrix_sp_reset, f)

np.savetxt(reset_simulation_path +
           f'resultados_sp_reset_{num_simulation +1}.csv', data_sp_reset, header=header_files, delimiter=',')
print("El fichero de resultados_sp_reset contiene ", data_sp_reset.shape[0], ' filas y ', data_sp_reset.shape[1], ' columnas')

# Obtengo todos los datos del reset en un único fichero
dat_reset = np.concatenate((data_pp_reset, data_sp_reset), axis=0)
np.savetxt(reset_simulation_path + f'Resultados_reset_{num_simulation +1}.csv', dat_reset, header=header_files, delimiter=',')
print("El fichero de resultados reset contiene ", dat_reset.shape[0], ' filas y ', dat_reset.shape[1], ' columnas')


# Obtengo los valores de g del proceso de reset, combinando los vectores de g de las dos partes
np.savetxt(reset_simulation_path + f'g_sp_reset_{num_simulation+1}.txt', g_sp_reset, delimiter=',', fmt='%.0f')
print("El g en el sp reset contiene ", g_sp_reset.shape[0], ' filas y ', g_sp_reset.shape[1], ' columnas')

g_reset = np.concatenate((g_pp_reset,g_sp_reset), axis=0)
np.savetxt(reset_simulation_path + f'g_reset_{num_simulation+1}.txt', g_reset, delimiter=',', fmt='%.0f')
print("El g en el reset contiene ", g_reset.shape[0], ' filas y ', g_reset.shape[1], ' columnas')

# Guardo el estado final de la simulación
RepresentateTwoStates(actual_state, oxygen_state, k, paso_potencial, simulation_path +
                      f'Figures/Final_state_sp_reset_{num_simulation+1}.png')
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

global_tittle = 'Intensidad [A] vs Voltaje [V]'

save_path = simulation_path + f'Figures/Intensidad_Voltaje_simulation_{num_simulation+1}'

i_ps = np.array(df_pset['Intensidad [A]'])
i_ss = np.array(df_sset['Intensidad [A]'])
i_pr = np.array(df_preset['Intensidad [A]'])
i_sr = np.array(df_sreset['Intensidad [A]'])

v_ps = np.array(df_pset['Voltaje [V]'])
v_ss = np.array(df_sset['Voltaje [V]'])
v_pr = np.array(df_preset['Voltaje [V]'])
v_sr = np.array(df_sreset['Voltaje [V]'])

# Uno en un solo array las intensidades y los voltajes de las curvas de SET
i_set = np.concatenate((i_ps, i_ss))
v_set = np.concatenate((v_ps, v_ss))

# Uno en un solo array las intensidades y los voltajes de las curvas de RESET
i_reset = np.concatenate((i_pr, i_sr))
v_reset = np.concatenate((v_pr, v_sr))


plot_IV(v_set, i_set, v_reset, i_reset, num_simulation)

save_path = simulation_path + f'Figures/Densidad_Filamento_{num_simulation+1}'

densidad_filamento = np.concatenate((np.array(df_pset['Densidad Filamento']), np.array(
    df_sset['Densidad Filamento']), df_preset['Densidad Filamento'], np.array(df_sreset['Densidad Filamento'])
))

# Tiempo de simulacion:
t_ps = np.array(df_pset['# Tiempo simulacion [s]'])
t_ss = np.array(df_sset['# Tiempo simulacion [s]'])
t_pr = np.array(df_preset['# Tiempo simulacion [s]'])
t_sr = np.array(df_sreset['# Tiempo simulacion [s]'])

tiempo = np.concatenate((t_ps, t_ss, t_pr, t_sr))

setup_plt(plt, latex=True, scaling=2)
fig, axes = plt.subplots(figsize=(12,9))
config_ax(axes)

# Configurar etiquetas y título
axes.set_xlabel(r"Voltage (\si{\V})")  # (\si{\nano\meter^{-1}})
axes.set_ylabel(r"Conductive filament density (number vancancies in filament/\si{\nano\meter^{2}})", fontsize=12)
axes.set_title(fr"Filament Density", pad=20)

axes.scatter(tiempo, densidad_filamento, s=2.5)

fig.savefig(save_path + '.png', bbox_inches='tight', dpi=300)


# endregion
