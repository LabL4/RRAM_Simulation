# region definicion de importaciones
import os
import sys
import pickle
import shutil
import time as time
import pandas as pd

from RRAM import *
from tqdm import tqdm
from RRAM import exceptions
from RRAM import Recombination
from RRAM import Plot_PostProcess as pplt

import warnings
warnings.filterwarnings("error")

# ruta_raiz = 'C:/Users/Usuario/Documents/GitHub/RRAM_Simulation/'
ruta_raiz = '/Users/antonio_lopez_torres/Documents/GitHub/RRAM_Simulation/' # Ruta en el mac
sys.path.append(ruta_raiz)
# endregion

# region Definición de valores iniciales y constantes de la simulación

# comienzo leyendo los datos de la simulación almacenados en un archivo csv dentro de la carpeta Init y los guardo en sus respectivas variables
sim_parmtrs = Montecarlo.read_csv_to_dic("Init_data/simulation_parameters.csv")
sim_ctes = Montecarlo.read_csv_to_dic("Init_data/simulation_constants.csv")

# Defino la carpeta donde se guardan los datos iniciales de la simulación
carpeta_results = 'Results'

# Verifica si la carpeta existe
if os.path.exists(carpeta_results):
    # Elimina la carpeta y su contenido
    shutil.rmtree(carpeta_results)

# Crea la carpeta de nuevo
os.makedirs(carpeta_results)

# # Ruta de la subcarpeta
# ruta_set = os.path.join(carpeta_results, 'set')
# ruta_reset = os.path.join(carpeta_results, 'reset')
# ruta_figures = os.path.join(carpeta_results, 'Figures')

# # Crear la subcarpeta s del set y reset
# os.makedirs(ruta_set, exist_ok=True)
# os.makedirs(ruta_reset, exist_ok=True)
# os.makedirs(ruta_figures, exist_ok=True)

# Defino las cabeceras de los archivos csv
header_files ='Tiempo simulacion [s],Voltaje [V],Intensidad [A],Temperatura [K],Campo Simple [V/m],Campo Gap medio [V/m],Velocidad [m/s]'

# endregion

# quiero un bucle que recorra todas las simulaciones desde 0 hasta la longitud de sim_parmtrs-1
for num_simulation in range(len(sim_parmtrs)):  
    # region Definición de variables

    # Creo la carpeta de la simulación
    simulation_path = os.path.join(carpeta_results, f'simulation_{num_simulation}/')
    os.makedirs(simulation_path, exist_ok=True)
    os.makedirs(simulation_path + 'Figures', exist_ok=True)
    
    set_simulation_path = os.path.join(carpeta_results, f'simulation_{num_simulation}/set/')
    os.makedirs(set_simulation_path, exist_ok=True)
    
    reset_simulation_path = os.path.join(carpeta_results, f'simulation_{num_simulation}/reset/')
    os.makedirs(reset_simulation_path, exist_ok=True)
    
    # Pongo el nombre de la simulación y un salto de línea
    print(f"\n Simulación {num_simulation + 1}")

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
    
    # df_sim_parmtrs = pd.DataFrame(sim_parmtrs[num_simulation])
    # df_sim_ctes = pd.DataFrame(sim_ctes[num_simulation])
    
    # # Guardar los DataFrames en un archivo de Excel con nombres específicos
    # with pd.ExcelWriter(os.path.join(ruta_simulation, 'simulacion_datos.xlsx')) as writer:
    #     df_sim_parmtrs.to_excel(writer, sheet_name='sim_parmtrs', index=False)
    #     df_sim_ctes.to_excel(writer, sheet_name='sim_ctes', index=False)

    # Leo los estados iniciales de la simulación
    with open('Init_data/init_state_' + str(num_simulation) + '.pkl', 'rb') as f:
        actual_state = pickle.load(f)
        
    with open('Init_data/oxygen_state_' + str(num_simulation) + '.pkl', 'rb') as f:
        oxygen_state = pickle.load(f)
    
    RepresentateState(actual_state, simulation_path + 'Figures/initial_pp_set_' + str(num_simulation))

    # with open('Init_data/oxygen_state_' + str(num_simulation) + '.pkl', 'rb') as f:
    #     oxygen_state = pickle.load(f)

    # Defino las matrices donde guardo las configuración del sistema y la de los oxígenos
    config_matrix_pp_set = np.zeros((int((num_pasos / paso_guardar)), x_size, y_size))

    # Defino el paso temporal
    paso_temporal = total_simulation_time / num_pasos
    paso_potencial = voltaje_max_simulation / num_pasos
    
    print(f"\nEl paso temporal es de {paso_temporal} s")
    print(f"El paso del potencial es de {paso_potencial} s")
    print("\nEl valor de la resistencia de cada casilla es: ", sim_ctes[num_simulation]['ohm_resistence'])
    print("El valor de gamma es: ", sim_ctes[num_simulation]['gamma'])
    
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

    # region primera parte del set
    for k in tqdm(range(0, num_pasos)):
        # Guardo el estado anterior
        last_state = actual_state

        # Actualizo el tiempo de simulación
        simulation_time = paso_temporal * k

        # Actualizo el voltaje
        voltage = vector_ddp[k]

        num_vacantes[k] = np.sum(actual_state)

        if voltage > voltaje_final_set:
            print("\nSe ha superado el voltaje de ruptura en la iteracion: ", k)
            
            # Verifica si el sistema ha percolado    
            if not sistema_percola:
                raise exceptions.NoPercolationException()

            k_ruptura = k
            voltaje_max_set = vector_ddp[k]
            config_matrix_recortada = config_matrix_pp_set[k, :, :]
            tiempo_pp_set = paso_temporal * (k - 1) # Le quitamos un paso porque se ha superado el voltaje de ruptura

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

            # RepresentateState(resistance_matrix, simulation_path + f'Figures/final_pp_set_resistance_{num_simulation}.png')
            break

        # Obtengo la corrriente, antes decido cual usar comprobando si ha percolado o no
        if Percolation.is_path(actual_state):
            sistema_percola = True
            
            # Cambio la probabilidad de generación de vacantes
            sim_ctes[num_simulation]['gamma'] = str(float(sim_ctes[num_simulation]['gamma']) / 10)
            
            # Copio el estado actual
            ac = actual_state.copy()
            resistance_matrix = findpath.find_path(ac)

            # Si ha percolado uso la corriente de Ohm
            try:
                current, resistencia[k] = CurentSolver.OmhCurrent(voltage, resistance_matrix, **sim_ctes[num_simulation])
            except Warning:
                filename = simulation_path + f'Null_Resistance/Configuration_Set_{voltage}_null_resistance.pkl'
                print("Null resistance matrix in ", filename)
                RepresentateState(resistance_matrix,simulation_path + f'Figures/Null_Resistance/NULL_resistance_matrix_pp_set_{num_simulation}.png')
                with open(filename, 'wb') as f:
                    pickle.dump({"actual_state": ac, "resistance_matrix": resistance_matrix}, f)
        else:
            # Si no ha percolado uso la corriente de Poole-Frenkel
            resistencia[k] = 0
            mean_field = np.mean(E_field_vector)
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

        data_pp_set[k-1] = np.array([simulation_time, voltage, current, temperatura,
                                    E_field, np.mean(E_field_vector), 0])

        # Guardo el estado actual CADA paso_guardar PASOS MONTECARLO
        if k % paso_guardar == 0:
            config_matrix_pp_set[int(k / paso_guardar) - 1] = actual_state

    # endregion

    # region Guardar datos del Primera parte del set

    # Cuando acaba la simulacion guardo ele stado final de configuracion
    with open(set_simulation_path + f'Last_Configuration_pp_set_{num_simulation}.pkl', 'wb') as f:
        pickle.dump(actual_state, f)

    # Cuando acaba la simulacion guardo las matrices de configuración
    with open(set_simulation_path + f'Configurations_pp_set_{num_simulation}.pkl', 'wb') as f:
        pickle.dump(config_matrix_pp_set, f)

    np.savetxt(set_simulation_path + f'resultados_pp_set_{num_simulation}.csv', data_pp_set, header=header_files, delimiter=',')

    # Guardo las vacantes generadas en el forming
    with open(set_simulation_path + f"Vacantes_resistencia_{num_simulation}.txt", "w") as f:
        for v1, v2, v3, v4 in zip(data_pp_set[:, 0], data_pp_set[:, 1], num_vacantes, resistencia):
            f.write(f"{v1} {v2} {v3} {v4}\n")

    # Leer el contenido del archivo TXT
    with open(set_simulation_path + f"Vacantes_resistencia_{num_simulation}.txt", 'r') as file:
        lines = file.readlines()

    header_files_extra = 'Tiempo simulacion [s],Voltaje [V],Resistencia [Ohm],Numero de vacantes \n'

    # Añadir el texto en la primera fila
    lines.insert(0, header_files_extra)

    # Escribir el contenido de nuevo en el archivo TXT
    with open(set_simulation_path + f"Vacantes_resistencia_{num_simulation}.txt", 'w') as file:
        file.writelines(lines)

    # endregion

    # region Segunda parte del Set

    # Estado inicial de la simulación reset para las vacantes
    with open(set_simulation_path + f'Last_Configuration_pp_set_{num_simulation}.pkl', 'rb') as file:
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

    RepresentateState(actual_state, simulation_path + f'Figures/Initial_configuration_sp_set_{num_simulation}.png')

    # Cambio la probabilidad de generación de vacantes
    sim_ctes[num_simulation]['gamma'] = str(float(sim_ctes[num_simulation]['gamma']) / 10)
    
    print(f"\n Comienza la segunda parte del set")
    # Ciclo para la segunda parte del set
    for k in tqdm(range(0, num_pasos)):
        # Actualizo el tiempo de simulación
        simulation_time = paso_temporal * k

        # Actualizo el voltaje
        voltage = vector_ddp[k]

        # Obtengo la corrriente, antes decido cual usar comprobando si ha percolado o no
        if Percolation.is_path(actual_state):
            sim_ctes[num_simulation]['gamma'] = str(float(sim_ctes[num_simulation]['gamma']) / 10)

            ac = actual_state.copy()
            resistance_matrix = findpath.find_path(ac)

            # Si ha percolado uso la corriente de Ohm
            try:
                current, resistencia[k] = CurentSolver.OmhCurrent(
                    voltage, resistance_matrix, **sim_ctes[num_simulation])
            except Warning:
                filename = simulation_path + f'Figures/Configuration_Set_{voltage}_null_resistance.pkl'
                RepresentateState(resistance_matrix,
                                  simulation_path + f'Figures/PS_resistance_matrix_{num_simulation}.png')
                print("Null resistance matrix in ", filename)
                with open(filename, 'wb') as f:
                    pickle.dump({"actual_state": ac, "resistance_matrix": resistance_matrix}, f)
        else:
            # Si no ha percolado uso la corriente de Poole-Frenkel
            mean_field = np.mean(E_field_vector)
            current = CurentSolver.Poole_Frenkel(
                temperatura, mean_field, **sim_ctes[num_simulation])*(device_size)

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

    # Cuando acaba la simulacion guardo las matrices de configuraciones y oxigenos
    with open(set_simulation_path + f'Configurations_sp_set_{num_simulation}.pkl', 'wb') as f:
        pickle.dump(config_matrix_sp_set, f)

    # Guardo el estado final de la simulación
    with open(set_simulation_path + f'Last_Configuration_sp_set_{num_simulation}.pkl', 'wb') as f:
        pickle.dump(actual_state, f)

    np.savetxt(set_simulation_path + f'Resultados_sp_set_{num_simulation}.csv', data_sp_set, header=header_files, delimiter=',')

    # endregion

    # region Región de la primera parte del reset

    # Estado inicial de la simulación reset para las vacantes
    with open(set_simulation_path + f'Last_Configuration_sp_set_{num_simulation}.pkl', 'rb') as file:
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

    RepresentateState(initial_configuration_reset, simulation_path + f'Figures/Initial_pp_reset_configuration_{num_simulation}.png')
    print(f"\n Comienza la primera parte del reset")

    sim_ctes[num_simulation]['gamma'] = str(float(sim_ctes[num_simulation]['gamma']) / 15)

    # Ciclo para la primera parte del reset
    for k in tqdm(range(0, num_pasos)):
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
                filename = reset_simulation_path + f'Configuration_pp_reset_{voltage}_null_resistance.pkl'
                print("Null resistance matrix in ", filename)
                RepresentateState(resistance_matrix, simulation_path + f'Figures/NULL_resistance_pp_reset_{num_simulation}.png')
                with open(filename, 'wb') as f:
                    pickle.dump({"actual_state": ac, "resistance_matrix": resistance_matrix}, f)
        else:
            # Si no ha percolado uso la corriente de Poole-Frenkel
            current = abs(CurentSolver.Poole_Frenkel(temperatura, np.mean(E_field_vector), **sim_ctes[num_simulation])*(device_size))

        # Obtengo los valores del campo eléctrico y la temperatura
        E_field = abs(SimpleElectricField(voltage, device_size))

        temperatura = Temperature_Joule(voltage, current, T_0, **sim_ctes[num_simulation])

        # Genero el vector campo eléctrico
        for i in range(0, actual_state.shape[0]):
            E_field_vector[i] = abs(GapElectricField(voltage, i, actual_state, **sim_parmtrs[num_simulation]))

        # Calculo la probabilidad de generación o recombinación para ello recorro toda la matriz
        for i in range(x_size):
            prob_generacion = Generation.Generate(paso_temporal, E_field_vector[i], temperatura, **sim_ctes[num_simulation])
            for j in range(y_size):
                if actual_state[i, j] == 0:
                    random_number = np.random.rand()
                    if random_number < prob_generacion:
                        actual_state[i, j] = 1  # Generación de una vacante

        # Genero los oxígenos
        oxygen_state = Recombination.Generate_Oxigen(oxygen_state, 5)

        # Muevo los oxígenos
        oxygen_state, velocidad, desplazamiento = Recombination.Move_OxygenIons(
            paso_temporal, oxygen_state, temperatura, E_field, atom_size, **sim_ctes[num_simulation])

        # Obtengo la nueva configuración
        actual_state, oxygen_state, pro_recombination = Recombination.Recombine(
            actual_state, oxygen_state, paso_temporal, velocidad, temperatura, **sim_ctes[num_simulation])

        # Tiempo total de la simulacion
        tiempo_total = simulation_time + 2 * tiempo_pp_set

        data_pp_reset[k] = np.array([tiempo_total, voltage, current, temperatura, E_field, np.mean(E_field_vector), desplazamiento])

        # Guardo el estado actual CADA paso_guardar PASOS MONTECARLO
        if k % paso_guardar == 0:
            config_matrix_pp_reset[int(k / paso_guardar) - 1] = actual_state
            oxygen_matrix_pp_reset[int(k / paso_guardar) - 1] = oxygen_state
    # endregion

    # region Guardar datos de la primera parte del reset

    # Cuando acaba la simulacion guardo las matrices de configuraciones y oxigenos
    with open(reset_simulation_path + f'Configurations_pp_reset_{num_simulation}.pkl', 'wb') as f:
        pickle.dump(config_matrix_pp_reset, f)
    with open(reset_simulation_path + f'Oxygen_pp_reset_{num_simulation}.pkl', 'wb') as f:
        pickle.dump(oxygen_matrix_pp_reset, f)

    # Estado inicial de la simulación reset segunda parte para las vacantes
    with open(reset_simulation_path + f'Last_Configuration_pp_reset_{num_simulation}.pkl', 'wb') as file:
        pickle.dump(actual_state, file)

    # Estado inicial para el reset segunda parte de los oxígenos
    with open( reset_simulation_path + f'Last_Oxygen_pp_reset_{num_simulation}.pkl', 'wb') as file:
        pickle.dump(oxygen_state, file)

    np.savetxt(reset_simulation_path + f'resultados_pp_reset_{num_simulation}.csv', data_pp_reset, header=header_files, delimiter=',')

    tiempo_pp_reset = simulation_time
    # endregion

    # region Región de la segunda parte del reset

    print(f"\n Comienza la segunda parte del reset")

    # Estado inicial de la simulación reset para las vacantes
    with open( reset_simulation_path + f'Last_Configuration_pp_reset_{num_simulation}.pkl', 'rb') as file:
        initial_configuration = pickle.load(file)

    # Estado inicial para el reset de los oxígenos
    with open( reset_simulation_path + f'Last_Oxygen_pp_reset_{num_simulation}.pkl', 'rb') as file:
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

    RepresentateState(actual_state, simulation_path + f'Figures/Initial_configuration_sp_reset_{num_simulation}.png')
    RepresentateState(oxygen_state, simulation_path + f'Figures/Initial_oxygen_sp_reset_{num_simulation}.png', color=(0.878, 0.227, 0.370))

    sim_ctes[num_simulation]['gamma'] = str(float(sim_ctes[num_simulation]['gamma']) / 15)
    
    # Ciclo para la segunda parte del reset
    for k in tqdm(range(0, num_pasos)): # son num_pasos + 1 iteraciones
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
                RepresentateState(resistance_matrix, simulation_path + f'Figures/PR_resistance_matrix_{num_simulation}.png')
                with open(filename, 'wb') as f:
                    pickle.dump({"actual_state": ac, "resistance_matrix": resistance_matrix}, f)
        else:
            # Si no ha percolado uso la corriente de Poole-Frenkel
            current = abs(CurentSolver.Poole_Frenkel(temperatura, np.mean(E_field_vector), **sim_ctes[num_simulation])*(device_size))

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
        oxygen_state = Recombination.Generate_Oxigen(oxygen_state, 10)

        # Muevo los oxígenos
        oxygen_state, velocidad, desplazamiento = Recombination.Move_OxygenIons(paso_temporal, oxygen_state, temperatura, E_field, atom_size, **sim_ctes[num_simulation])

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

    # Cuando acaba la simulacion guardo las matrices de configuraciones y oxigenos
    with open(reset_simulation_path + f'Configurations_sp_reset_{num_simulation}.pkl', 'wb') as f:
        pickle.dump(config_matrix_sp_reset, f)
    with open(reset_simulation_path + f'Oxygen_sp_reset_{num_simulation}.pkl', 'wb') as f:
        pickle.dump(oxygen_matrix_sp_reset, f)

    np.savetxt(reset_simulation_path + f'resultados_sp_reset_{num_simulation}.csv', data_sp_reset, header=header_files, delimiter=',')

    RepresentateState(actual_state,simulation_path + f'Figures/final_Configuration_sp_reset_{num_simulation}.png')
    RepresentateState(oxygen_state,simulation_path + f'Figures/final_Oxygen_sp_reset_{num_simulation}.png',color=(0.878, 0.227, 0.370))

    # endregion

    # region Unir todos los datos en un solo archivo csv TODO: necesita revision cuando se unen los datos

    df_pset = pd.read_csv(set_simulation_path + f'Resultados_pp_set_{num_simulation}.csv')
    df_sset = pd.read_csv(set_simulation_path + f'Resultados_sp_set_{num_simulation}.csv')
    df_preset = pd.read_csv(reset_simulation_path + f'resultados_pp_reset_{num_simulation}.csv')
    df_sreset = pd.read_csv(reset_simulation_path + f'resultados_sp_reset_{num_simulation}.csv')

    # Concatenar los DataFrames sin duplicar el encabezado
    data_frame_simulation = pd.concat([df_pset, df_sset, df_preset, df_sreset])

    # Guardar el DataFrame combinado en un archivo CSV
    data_frame_simulation.to_csv(f'Results/Datos_simulacion_completa_{num_simulation}.csv', index=False)

    print("Todos los archivos CSV se han combinado y guardado exitosamente.")

    # endregion

    # region Representar datos
    data_path_pp_set = set_simulation_path + 'resultados_pp_set_0.csv'
    data_path_sp_set = set_simulation_path + 'resultados_sp_set_0.csv'
    data_path_pp_reset = reset_simulation_path + 'resultados_pp_reset_0.csv'
    data_path_sp_reset = reset_simulation_path + 'resultados_sp_reset_0.csv'

    df_pset = pd.read_csv(data_path_pp_set, dtype=float)
    df_sset = pd.read_csv(data_path_sp_set, dtype=float)
    df_preset = pd.read_csv(data_path_pp_reset, dtype=float)
    df_sreset = pd.read_csv(data_path_sp_reset, dtype=float)

    global_tittle = 'Intensidad vs Voltaje'
    save_path = simulation_path + f'Figures/Intensidad_Voltaje_simulation_{num_simulation}'

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
    axes.plot(v_ps, i_ps, color='blue', label='Primera parte Set')
    axes.plot(v_ss, i_ss, color='red', label='Segunda parte Set')
    axes.plot(v_pr, i_pr, color='green', label='Primera parte Reset')
    axes.plot(v_sr, i_sr, color='pink', label='Segunda parte Reset')

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

    # Crear la gráfica scatter
    axes.plot(x_set, y_set, 'black', label='Set experimental')
    axes.plot(x_reset, y_reset, 'black', label='Reset experimental')
    
    fig.savefig(save_path + '.pdf', bbox_inches='tight')
    # plt.show()
    
    # endregion
