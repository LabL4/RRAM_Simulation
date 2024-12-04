# region definicion de importaciones
import os
import pickle
import shutil
import time as time
import pandas as pd

from tqdm import tqdm

from RRAM import *
from RRAM import Recombination

import warnings
warnings.filterwarnings("error")

# endregion

# region Definición de valores iniciales y cosntantes de la simulación

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

# Ruta de la subcarpeta
ruta_set = os.path.join(carpeta_results, 'set')
ruta_reset = os.path.join(carpeta_results, 'reset')
ruta_figures = os.path.join(carpeta_results, 'Figures')

# Crear la subcarpeta s del set y reset
os.makedirs(ruta_set, exist_ok=True)
os.makedirs(ruta_reset, exist_ok=True)
os.makedirs(ruta_figures, exist_ok=True)


header_files = 'Tiempo simulacion [s],Voltaje [V],Intensidad [A],Temperatura [K],Campo Simple [V/m],Campo Gap medio [V/m],Velocidad [m/s]'

# endregion

# quiero un bucle que recorra todas las simulaciones desde 0 hasta la longitud de sim_parmtrs-1
for num_simulation in range(len(sim_parmtrs)):

    # region Definición de variables

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

    voltaje_reset = float(sim_parmtrs[num_simulation]['voltaje_final'])

    voltage = float(sim_parmtrs[num_simulation]['initial_voltaje'])
    current = float(sim_parmtrs[num_simulation]['initial_current'])
    temperatura = float(sim_parmtrs[num_simulation]['init_temp'])
    E_field = float(sim_parmtrs[num_simulation]['initial_elec_field'])

    # Leo los estados iniciales de la simulación
    with open('Init_data/init_state_' + str(num_simulation) + '.pkl', 'rb') as f:
        actual_state = pickle.load(f)

    with open('Init_data/oxygen_state_' + str(num_simulation) + '.pkl', 'rb') as f:
        oxygen_state = pickle.load(f)

    # Defino las matrices donde guardo las configuración del sistema y la de los oxígenos
    config_matrix_pp_set = np.zeros((int((num_pasos / paso_guardar)), x_size, y_size))
    # oxygen_matrix_pp_set = np.zeros((int((num_pasos / paso_guardar)), x_size, y_size))

    # Defino el paso temporal
    paso_temporal = total_simulation_time / num_pasos

    # Creo el vector de diferencias de potencial
    vector_ddp = np.linspace(0, voltaje_reset, num_pasos + 1)

    # Creo el vector de datos como una matriz de num_pasos filas y las columnas necesarias
    colunm_number = 7
    data_pp_set = np.zeros((num_pasos-1, colunm_number))

    # Inicializo el campo eléctrico
    E_field_vector = np.zeros((actual_state.shape[0]))

    num_vacantes = np.zeros(num_pasos)
    resistencia = np.zeros(num_pasos)

    T_0 = float(sim_parmtrs[num_simulation]['init_temp'])

    # endregion

    # region primera parte del set

    for k in tqdm(range(1, num_pasos+1)):
        # Guardo el estado anterior
        last_state = actual_state

        # Actualizo el tiempo de simulación
        simulation_time = paso_temporal * k

        # Actualizo el voltaje
        voltage = vector_ddp[k]

        num_vacantes[k] = np.sum(actual_state)

        if voltage > 2.3:
            print("\nSe ha superado el voltaje de ruptura en la iteracion: ", k)
            k_ruptura = k
            voltaje_inicial_reset = vector_ddp[k]
            simulation_time_forming = simulation_time
            config_matrix_recortada = config_matrix_pp_set[k, :, :]

            print("Voltaje final forming", voltaje_inicial_reset, 'en el tiempo ', simulation_time_forming, "\n")

            # Crear un array de ejemplo
            data_pp_set[k-1:] = np.nan  # Añadir valores nulos a partir de la fila k
            num_vacantes[k:] = np.nan  # Añadir valores nulos a partir de la fila k
            resistencia[k:] = np.nan  # Añadir valores nulos a partir de la fila k

            # Eliminar filas con valores nulos
            data_pp_set = data_pp_set[~np.isnan(data_pp_set).any(axis=1)]
            num_vacantes = num_vacantes[~np.isnan(num_vacantes)]
            resistencia = resistencia[~np.isnan(resistencia)]

            RepresentateState(resistance_matrix, f'Results/set/resistance_{num_simulation}_end_pp_set.png')

            break

        # Obtengo la corrriente, antes decido cual usar comprobando si ha percolado o no
        if Percolation.is_path(actual_state):
            # Copio el estado actual
            ac = actual_state.copy()
            resistance_matrix = findpath.find_path(ac)

            # Si ha percolado uso la corriente de Ohm
            try:
                current, resistencia[k] = CurentSolver.OmhCurrent(
                    voltage, resistance_matrix, **sim_ctes[num_simulation])
            except Warning:
                filename = f'Results/Null_Resistance/Configuration_Set_{voltage}_null_resistance.pkl'
                print("Null resistance matrix in ", filename)
                RepresentateState(resistance_matrix,
                                  f'Results/Null_Resistance/PS_resistance_matrix_{num_simulation}.png')
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
    with open(f'Results/set/Last_Configuration_pp_set_{num_simulation}.pkl', 'wb') as f:
        pickle.dump(actual_state, f)

    # Cuando acaba la simulacion guardo las matrices de configuración
    with open(f'Results/set/Configurations_pp_set_{num_simulation}.pkl', 'wb') as f:
        pickle.dump(config_matrix_pp_set, f)

    np.savetxt(f'Results/set/resultados_pp_set_{num_simulation}.csv', data_pp_set, header=header_files, delimiter=',')

    # Guardo las vacantes generadas en el forming
    with open(f"Results/set/Vacantes_resistencia_{num_simulation}.txt", "w") as f:
        for v1, v2, v3, v4 in zip(data_pp_set[:, 0], data_pp_set[:, 1], num_vacantes, resistencia):
            f.write(f"{v1} {v2} {v3} {v4}\n")

    # Leer el contenido del archivo TXT
    with open(f"Results/set/Vacantes_resistencia_{num_simulation}.txt", 'r') as file:
        lines = file.readlines()

    header_files_extra = 'Tiempo simulacion [s],Voltaje [V],Resistencia [Ohm],Numero de vacantes \n'

    # Añadir el texto en la primera fila
    lines.insert(0, header_files_extra)

    # Escribir el contenido de nuevo en el archivo TXT
    with open(f"Results/set/Vacantes_resistencia_{num_simulation}.txt", 'w') as file:
        file.writelines(lines)

    # endregion

    # region Segunda parte del Set

    # Estado inicial de la simulación reset para las vacantes
    with open(f'Results/set/Last_Configuration_pp_set_{num_simulation}.pkl', 'rb') as file:
        # Carga el contenido del archivo
        initial_configuration = pickle.load(file)

    # NUMERO DE PASOS QUE SE HA dado en el forming. Lo pongo igual en el reset para que los potenciales sean los mismos
    num_pasos = k_ruptura

    # Creo el vector de datos como una matriz de num_pasos filas y las columnas necesarias
    data_sset = np.zeros((num_pasos, colunm_number))

    # Defino las matrices donde guardo las configuración del sistema y la de los oxígenos
    config_matrix_sp_set = np.zeros((int((num_pasos / paso_guardar)), x_size, y_size))
    # oxygen_matrix_sp_set = np.zeros((int((num_pasos / paso_guardar)), x_size, y_size))

    # Cambio la probabilidad de generación de vacantes
    sim_ctes[num_simulation]['gamma'] = '3'

    # Creo el vector de diferencias de potencial
    vector_ddp = np.linspace(voltaje_inicial_reset, 0, num_pasos)

    # Estado iniciales de la simulación para el reset
    actual_state = initial_configuration

    RepresentateState(actual_state, f'Results/set/Initial_configuration_sp_set_{num_simulation}.png')

    print(f"\n Comienza la segunda parte del set")
    # Ciclo para la segunda parte del set
    for k in tqdm(range(0, num_pasos)):
        # Actualizo el tiempo de simulación
        simulation_time = paso_temporal * (k+1)

        # Actualizo el voltaje
        voltage = vector_ddp[k]

        # Obtengo la corrriente, antes decido cual usar comprobando si ha percolado o no
        if Percolation.is_path(actual_state):
            sim_ctes[num_simulation]['gamma'] = '0.3'

            ac = actual_state.copy()
            resistance_matrix = findpath.find_path(ac)

            # Si ha percolado uso la corriente de Ohm
            try:
                current, resistencia[k] = CurentSolver.OmhCurrent(
                    voltage, resistance_matrix, **sim_ctes[num_simulation])
            except Warning:
                filename = f'Results/Null_Resistance/Configuration_Set_{voltage}_null_resistance.pkl'
                RepresentateState(resistance_matrix,
                                  f'Results/Null_Resistance/PS_resistance_matrix_{num_simulation}.png')
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
        tiempo_total = simulation_time + simulation_time_forming

        data_sset[k] = np.array([tiempo_total, voltage, current, temperatura, E_field, np.mean(E_field_vector), 0])

        # Guardo el estado actual CADA paso_guardar PASOS MONTECARLO
        if k % paso_guardar == 0:
            config_matrix_sp_set[int(k / paso_guardar) - 1] = actual_state
    # endregion

    # region Guardar datos de la segunda parte del set

    # Cuando acaba la simulacion guardo las matrices de configuraciones y oxigenos
    with open(f'Results/set/Configurations_sp_set_{num_simulation}.pkl', 'wb') as f:
        pickle.dump(config_matrix_sp_set, f)

    # Guardo el estado final de la simulación
    with open(f'Results/set/Last_Configuration_sp_set_{num_simulation}.pkl', 'wb') as f:
        pickle.dump(actual_state, f)

    np.savetxt(f'Results/set/Resultados_sp_set_{num_simulation}.csv', data_sset,
               header=header_files,
               comments=' ', delimiter=',')

    # endregion

    # region Región de la primera parte del reset

    # Estado inicial de la simulación reset para las vacantes
    with open(f'Results/set/Last_Configuration_sp_set_{num_simulation}.pkl', 'rb') as file:
        initial_configuration = pickle.load(file)

    # NUMERO DE PASOS QUE SE HA dado en el forming. Lo pongo igual en el reset para que los potenciales sean los mismos
    num_pasos = k_ruptura

    # el pp significa primera parte del reset (reset primera parte)
    data_pp_reset = np.zeros((num_pasos - 1, colunm_number))

    # Defino las matrices donde guardo las configuración del sistema y la de los oxígenos
    config_matrix_pp_reset = np.zeros((int((num_pasos / paso_guardar)), x_size, y_size))
    oxygen_matrix_pp_reset = np.zeros((int((num_pasos / paso_guardar)), x_size, y_size))

    # Creo el vector de diferencias de potencial
    vector_ddp = np.linspace(0, -voltaje_inicial_reset, num_pasos)

    # Estado iniciales de la simulación para el reset
    initial_configuration_reset = actual_state
    initial_oxygen_reset = oxygen_state

    RepresentateState(initial_configuration_reset, f'Results/reset/Initial_pp_reset_configuration_{num_simulation}.png')
    print(f"\n Comienza la primera parte del reset")

    sim_ctes[num_simulation]['gamma_drift'] = '10'
    sim_ctes[num_simulation]['gamma'] = '0.3'

    # Ciclo para la primera parte del reset
    for k in tqdm(range(0, num_pasos-1)):
        # Actualizo el tiempo de simulación
        simulation_time = paso_temporal * (k+1)

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
                filename = f'Results/reset/Configuration_Set_{voltage}_null_resistance.pkl'
                print("Null resistance matrix in ", filename)
                RepresentateState(resistance_matrix,
                                  f'Results/reset/PR_resistance_matrix_{num_simulation}.png')
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
        oxygen_state = Recombination.Generate_Oxigen(oxygen_state, 15)

        # Muevo los oxígenos
        oxygen_state, velocidad = Recombination.Move_OxygenIons(
            paso_temporal, oxygen_state, temperatura, E_field, atom_size, **sim_ctes[num_simulation])

        # Obtengo la nueva configuración
        actual_state, oxygen_state, pro_recombination = Recombination.Recombine(
            actual_state, oxygen_state, paso_temporal, velocidad, temperatura, **sim_ctes[num_simulation])

        # Tiempo total de la simulacion
        tiempo_total = simulation_time + 2 * simulation_time_forming

        data_pp_reset[k] = np.array([tiempo_total, voltage, current, temperatura,
                                     E_field, np.mean(E_field_vector), 0])

        # Guardo el estado actual CADA paso_guardar PASOS MONTECARLO
        if k % paso_guardar == 0:
            config_matrix_pp_reset[int(k / paso_guardar) - 1] = actual_state
            oxygen_matrix_pp_reset[int(k / paso_guardar) - 1] = oxygen_state
    # endregion

    # region Guardar datos del reset primera parte

    # Cuando acaba la simulacion guardo las matrices de configuraciones y oxigenos
    with open(f'Results/reset/Configurations_pp_reset_{num_simulation}.pkl', 'wb') as f:
        pickle.dump(config_matrix_pp_reset, f)
    with open(f'Results/reset/Oxygen_pp_reset_{num_simulation}.pkl', 'wb') as f:
        pickle.dump(oxygen_matrix_pp_reset, f)

    # Estado inicial de la simulación reset segunda parte para las vacantes
    with open(f'Results/reset/Last_Configuration_pp_reset_{num_simulation}.pkl', 'wb') as file:
        pickle.dump(actual_state, file)

    # Estado inicial para el reset segunda parte de los oxígenos
    with open(f'Results/reset/Last_Oxygen_pp_reset_{num_simulation}.pkl', 'wb') as file:
        pickle.dump(oxygen_state, file)

    np.savetxt(f'Results/reset/resultados_pp_reset_{num_simulation}.csv', data_pp_reset,
               header=header_files,
               comments=' ', delimiter=',')

    # endregion

    # region Región del reset segunda parte

    print(f"\n Comienza la segunda parte del reset")

    # Estado inicial de la simulación reset para las vacantes
    with open(f'Results/reset/Last_Configuration_pp_reset_{num_simulation}.pkl', 'rb') as file:
        initial_configuration = pickle.load(file)

    # Estado inicial para el reset de los oxígenos
    with open(f'Results/reset/Last_Oxygen_pp_reset_{num_simulation}.pkl', 'rb') as file:
        initial_oxygen_reset = pickle.load(file)

    # NUMERO DE PASOS QUE SE HA dado en el forming. Lo pongo igual en el reset para que los potenciales sean los mismos
    num_pasos = k_ruptura

    # el sp significa segunda parte del reset (reset primera parte)
    data_sp_reset = np.zeros((num_pasos - 1, colunm_number))

    # Defino las matrices donde guardo las configuración del sistema y la de los oxígenos
    config_matrix_sp_reset = np.zeros((int((num_pasos / paso_guardar)), x_size, y_size))
    oxygen_matrix_sp_reset = np.zeros((int((num_pasos / paso_guardar)), x_size, y_size))

    # Creo el vector de diferencias de potencial
    vector_ddp = np.linspace(-voltaje_inicial_reset, 0, num_pasos)

    # Estado iniciales de la simulación para la segunda parte del reset
    # initial_configuration_reset = actual_state
    # initial_oxygen_reset = oxygen_state

    RepresentateState(actual_state, f'Results/reset/Initial_configuration_sp_reset_{num_simulation}.png')
    RepresentateState(oxygen_state, f'Results/reset/Initial_oxygen_sp_reset_{num_simulation}.png')

    # Ciclo para la segunda parte del reset
    for k in tqdm(range(0, num_pasos-1)):
        # Actualizo el tiempo de simulación
        simulation_time = paso_temporal * (k + 1)

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
                filename = f'Results/reset/Configuration_sp_reset_{voltage}_null_resistance.pkl'
                print("Null resistance matrix in ", filename)
                RepresentateState(resistance_matrix,
                                  f'Results/reset/PR_resistance_matrix_{num_simulation}.png')
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
        oxygen_state = Recombination.Generate_Oxigen(oxygen_state, 15)

        # Muevo los oxígenos
        oxygen_state, velocidad = Recombination.Move_OxygenIons(
            paso_temporal, oxygen_state, temperatura, E_field, atom_size, **sim_ctes[num_simulation])

        # Obtengo la nueva configuración
        actual_state, oxygen_state, pro_recombination = Recombination.Recombine(
            actual_state, oxygen_state, paso_temporal, velocidad, temperatura, **sim_ctes[num_simulation])

        # Tiempo total de la simulacion
        tiempo_total = simulation_time + 3 * simulation_time_forming

        data_sp_reset[k] = np.array([tiempo_total, voltage, current, temperatura,
                                     E_field, np.mean(E_field_vector), 0])

        # Guardo el estado actual CADA paso_guardar PASOS MONTECARLO
        if k % paso_guardar == 0:
            config_matrix_sp_reset[int(k / paso_guardar) - 1] = actual_state
            oxygen_matrix_sp_reset[int(k / paso_guardar) - 1] = oxygen_state
    # endregion

    # region Guardar datos del reset segunda parte

    # Cuando acaba la simulacion guardo las matrices de configuraciones y oxigenos
    with open(f'Results/reset/Configurations_sp_reset_{num_simulation}.pkl', 'wb') as f:
        pickle.dump(config_matrix_sp_reset, f)
    with open(f'Results/reset/Oxygen_sp_reset_{num_simulation}.pkl', 'wb') as f:
        pickle.dump(oxygen_matrix_sp_reset, f)

    np.savetxt(f'Results/reset/resultados_sp_reset_{num_simulation}.csv', data_sp_reset,
               header=header_files,
               comments=' ', delimiter=',')

    RepresentateState(actual_state, f'Results/reset/Configuration_final_sp_reset_{num_simulation}.png')
    RepresentateState(oxygen_state, f'Results/reset/Oxygen_final_sp_reset_{num_simulation}.png')

    # endregion

    # region Unir todos los datos en un solo archivo csv

    df_pset = pd.read_csv(f'Results/set/Resultados_pp_set_{num_simulation}.csv')
    df_sset = pd.read_csv(f'Results/set/Resultados_sp_set_{num_simulation}.csv')
    df_preset = pd.read_csv(f'Results/reset/resultados_pp_reset_{num_simulation}.csv')
    df_sreset = pd.read_csv(f'Results/reset/resultados_sp_reset_{num_simulation}.csv')

    # Concatenar los DataFrames sin duplicar el encabezado
    data_frame_simulation = pd.concat([df_pset, df_sset, df_preset, df_sreset])

    # Guardar el DataFrame combinado en un archivo CSV
    data_frame_simulation.to_csv(f'Results/Datos_simulacion_completa_{num_simulation}.csv', index=False)

    print("Todos los archivos CSV se han combinado y guardado exitosamente.")

    # endregion

    # region Representar datos
    data_full = f'Results/Datos_simulacion_completa_{num_simulation}.csv'

    df = pd.read_csv(data_full, dtype=float)

    intensidad = np.array(df['Intensidad [A]'])
    voltaje = np.array(df['Voltaje [V]'])

    plt.semilogy(df['Voltaje [V]'], abs(df['Intensidad [A]']))
    plt.xlabel("Voltaje")
    plt.ylabel("Resistencia")
    plt.title(r"Intensidad en función del voltaje con $I_0 = {}$".format(sim_ctes[num_simulation]['I_0']))

    # guardo la figura
    plt.savefig(f'Results/Grafico_Intensidad_Voltaje_{num_simulation}.png')

    # represento el estado de los oxígenos a partir de los pkl
    # with open(f'Results/reset/Oxygen_pp_reset_{num_simulation}.pkl', 'rb') as f:
        # oxygen_pp_reset = pickle.load(f)

    # with open(f'Results/reset/Oxygen_sp_reset_{num_simulation}.pkl', 'rb') as f:
        # oxygen_sp_reset = pickle.load(f)

    # Represento el estado de los oxígenos cada 50 pasos
    # num_representar = 50
    # vector_guardar = np.arange(0, num_pasos, paso_guardar)
    # print(vector_guardar)

    # print("Representando el estado de los oxígenos")

    # for i in vector_guardar:
        # RepresentateState(oxygen_pp_reset[i], f'Results/Figures/Oxygen_pp_reset_{num_simulation}_{i}.png')
        # RepresentateState(oxygen_sp_reset[i], f'Results/Figures/Oxygen_sp_reset_{num_simulation}_{i}.png')
    # endregion
    
    # print("Representados los estado de los oxígenos")
