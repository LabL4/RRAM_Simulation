import os
import pickle
import shutil
import time as time
import pandas as pd

from tqdm import tqdm

from RRAM import *
from RRAM import Recombination
from RRAM import Plot_PostProcess as pplt

# region Definición de valores iniciales y cosntantes de la simulación

# comienzo leyendo los datos de la simulación almacenados en un archivo csv dentro de la carpeta Init y los guardo en sus respectivas variables
sim_parmtrs = Montecarlo.read_csv_to_dic("Init_data/simulation_parameters.csv")
sim_ctes = Montecarlo.read_csv_to_dic("Init_data/simulation_constants.csv")

# Defino la carpeta donde se guardan los datos iniciales de la simulación
carpeta = 'Results'

# Verifica si la carpeta existe
if os.path.exists(carpeta):
    # Elimina la carpeta y su contenido
    shutil.rmtree(carpeta)

# Crea la carpeta de nuevo
os.makedirs(carpeta)

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

    voltaje = float(sim_parmtrs[num_simulation]['initial_voltaje'])
    corriente = float(sim_parmtrs[num_simulation]['initial_current'])
    temperatura = float(sim_parmtrs[num_simulation]['init_temp'])
    E_field = float(sim_parmtrs[num_simulation]['initial_elec_field'])

    # Leo los estados iniciales de la simulación
    with open('Init_data/init_state_' + str(num_simulation) + '.pkl', 'rb') as f:
        actual_state = pickle.load(f)

    with open('Init_data/oxygen_state_' + str(num_simulation) + '.pkl', 'rb') as f:
        oxygen_state = pickle.load(f)

    # Defino las matrices donde guardo las configuración del sistema y la de los oxígenos
    config_matrix = np.zeros((int((num_pasos / paso_guardar)), x_size, y_size))
    oxygen_matrix = np.zeros((int((num_pasos / paso_guardar)), x_size, y_size))

    # Defino el paso temporal
    paso_temporal = total_simulation_time / num_pasos

    # Creo el vector de diferencias de potencial
    vector_ddp = np.linspace(0, voltaje_reset, num_pasos + 1)

    # Creo el vector de datos como una matriz de num_pasos filas y las columnas necesarias
    colunm_number = 6
    data = np.zeros((num_pasos, colunm_number))

    # Creo el excel donde voy a sacar todos los datos
    df = pd.DataFrame(columns=['Tiempo simulacion [s]', 'Voltaje [V] ',
                      'Intensidad [A]', 'Temperatura [K]', 'Campo Simple [V/m]', 'Campo Gap medio [V/m]'])

    # Inicializo el campo eléctrico
    E_field_vector = np.zeros((actual_state.shape[0]))

    T_0 = float(sim_parmtrs[num_simulation]['init_temp'])

    # endregion

    # region Forming

    for k in tqdm(range(1, num_pasos+1)):
        # Guardo el estado anterior
        last_state = actual_state

        # Actualizo el tiempo de simulación
        simulation_time = paso_temporal * k

        # Actualizo el voltaje
        voltaje = vector_ddp[k]

        if voltaje > 2.25:
            print("Se ha superado el voltaje de ruptura", k)
            k_ruptura = k
            voltaje_inicial_reset = vector_ddp[k]
            simulation_time_forming = simulation_time
            print("Voltaje final forming", voltaje_inicial_reset, 'en el tiempo ', simulation_time_forming)

            # Crear un array de ejemplo
            data[k:] = np.nan  # Añadir valores nulos a partir de la fila k

            # Eliminar filas con valores nulos
            data = data[~np.isnan(data).any(axis=1)]
            break

        # Obtengo la corrriente, antes decido cual usar comprobando si ha percolado o no
        if Percolation.is_path(actual_state):
            # Si ha percolado uso la corriente de Ohm
            corriente = CurentSolver.OmhCurrent(voltaje, actual_state, **sim_ctes[num_simulation])
        else:
            # Si no ha percolado uso la corriente de Poole-Frenkel
            corriente = CurentSolver.poole_frenkel(temperatura, np.mean(
                E_field_vector), **sim_ctes[num_simulation])*(device_size)

        # Obtengo los valores del campo eléctrico y la temperatura
        E_field = SimpleElectricField(voltaje, device_size)

        temperatura = Temperature_Joule(voltaje, corriente, T_0, **sim_ctes[num_simulation])

        # Genero el vector campo eléctrico
        for i in range(0, actual_state.shape[0]):
            E_field_vector[i] = GapElectricField(voltaje, i, actual_state, **sim_parmtrs[num_simulation])

        # Calculo la probabilidad de generación o recombinación para ello recorro toda la matriz
        for i in range(x_size):
            prob_generacion = Generation.Generate(
                paso_temporal, E_field_vector[i], temperatura, **sim_ctes[num_simulation])
            for j in range(y_size):
                if actual_state[i, j] == 0:
                    random_number = np.random.rand()
                    if random_number < prob_generacion:
                        actual_state[i, j] = 1  # Generación de una vacante

        data[k] = np.array([simulation_time, voltaje, corriente, temperatura, E_field, np.mean(E_field_vector)])

        # Guardo el estado actual CADA paso_guardar PASOS MONTECARLO
        if k % paso_guardar == 0:
            config_matrix[int(k / paso_guardar) - 1] = actual_state
            oxygen_matrix[int(k / paso_guardar) - 1] = oxygen_state

    # endregion

    # region Guardar datos del forming

    # Cuando acaba la simulacion guardo los estados de las matrices de configuracion y oxigenos
    with open(f'Results/Last_Configuration_{num_simulation}.pkl', 'wb') as f:
        pickle.dump(actual_state, f)

    # with open(f'Results/Last_OxygenState_{num_simulation}.pkl', 'wb') as f:
    #     pickle.dump(oxygen_state, f)

    # Cuando acaba la simulacion guardo las matrices de configuraciones y oxigenos
    with open(f'Results/Configurations_forming_{num_simulation}.pkl', 'wb') as f:
        pickle.dump(config_matrix, f)
    # with open(f'Results/Oxygen_forming_{num_simulation}.pkl', 'wb') as f:
        # pickle.dump(oxygen_matrix, f)

    np.savetxt(f'Results/resultados_forming_{num_simulation}.csv', data,
               header='Tiempo simulacion [s], Voltaje [V] , Intensidad [A], Temperatura [K], Campo Simple [V/m], Campo Gap medio [V/m]',
               comments=' ', delimiter=', ')

    # endregion

    # region reset

    # Estado inicial de la simulación reset para las vacantes
    with open(f'Results/Last_Configuration_{num_simulation}.pkl', 'rb') as file:
        # Carga el contenido del archivo
        initial_configuration = pickle.load(file)

    # NUMERO DE PASOS QUE SE HA dado en el forming. Lo pongo igual en el reset para que los potenciales sean los mismos
    num_pasos = k_ruptura

    # Creo el vector de datos como una matriz de num_pasos filas y las columnas necesarias
    colunm_number = 7
    data_reset = np.zeros((num_pasos, colunm_number))

    # Defino las matrices donde guardo las configuración del sistema y la de los oxígenos
    config_matrix_reset = np.zeros((int((num_pasos / paso_guardar)), x_size, y_size))
    oxygen_matrix_reset = np.zeros((int((num_pasos / paso_guardar)), x_size, y_size))
    # Creo el excel donde voy a sacar todos los datos

    # df_reset = pd.DataFrame(columns=['Tiempo simulacion [s]', 'Voltaje [V] ',
    #  'Intensidad [A]', 'Temperatura [K]', 'Campo Simple [V/m]', 'Campo Gap medio [V/m]'])

    # Cmabio la probabilidad del forming
    sim_ctes[num_simulation]['gamma'] = '0.3'

    # Defino el paso temporal
    # paso_temporal = total_simulation_time / num_pasos

    # Creo el vector de diferencias de potencial
    vector_ddp = np.linspace(voltaje_inicial_reset, 0, num_pasos)

    # Estado iniciales de la simulación para el reset
    actual_state = initial_configuration

    RepresentateState(actual_state, f'Results/Initial_reset_configuration_{num_simulation}.png')

    for k in tqdm(range(1, num_pasos)):
        # Actualizo el tiempo de simulación
        simulation_time = paso_temporal * k

        # Actualizo el voltaje
        voltaje = vector_ddp[k]

        # Obtengo la corrriente, antes decido cual usar comprobando si ha percolado o no
        if Percolation.is_path(actual_state):
            # Si ha percolado uso la corriente de Ohm
            corriente = CurentSolver.OmhCurrent(voltaje, actual_state, **sim_ctes[num_simulation])
            # print("Corriente Ohm", corriente)
        else:
            # Si no ha percolado uso la corriente de Poole-Frenkel
            corriente = CurentSolver.poole_frenkel(temperatura, np.mean(
                E_field_vector), **sim_ctes[num_simulation])*(device_size)

        # Obtengo los valores del campo eléctrico y la temperatura
        E_field = SimpleElectricField(voltaje, device_size)

        temperatura = Temperature_Joule(voltaje, corriente, T_0, **sim_ctes[num_simulation])

        # Genero el vector campo eléctrico
        for i in range(0, actual_state.shape[0]):
            E_field_vector[i] = GapElectricField(voltaje, i, actual_state, **sim_parmtrs[num_simulation])

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
        oxygen_state, velocidad = Recombination.Move_OxygenIons(
            paso_temporal, oxygen_state, temperatura, E_field, atom_size, **sim_ctes[num_simulation])

        # Obtengo la nueva configuración
        actual_state, oxygen_state, pro_recombination = Recombination.Recombine(
            actual_state, oxygen_state, paso_temporal, velocidad, temperatura, **sim_ctes[num_simulation])

        # Tiempo total de la simulacion
        tiempo_total = simulation_time + simulation_time_forming

        data_reset[k-1] = np.array([tiempo_total, voltaje, corriente, temperatura,
                                    velocidad, E_field, np.mean(E_field_vector)])

        # Guardo el estado actual CADA paso_guardar PASOS MONTECARLO
        if k % paso_guardar == 0:
            config_matrix_reset[int(k / paso_guardar) - 1] = actual_state
            oxygen_matrix_reset[int(k / paso_guardar) - 1] = oxygen_state

    # Cuando acaba la simulacion guardo las matrices de configuraciones y oxigenos
    with open(f'Results/Configurations_reset_{num_simulation}.pkl', 'wb') as f:
        pickle.dump(config_matrix, f)
    with open(f'Results/Oxygen_reset_{num_simulation}.pkl', 'wb') as f:
        pickle.dump(oxygen_matrix, f)
    # endregion

    # region Guardar datos del reset
    np.savetxt(f'Results/resultados_reset_{num_simulation}.csv', data_reset,
               header='Tiempo simulacion [s], Voltaje [V] , Intensidad [A], Temperatura [K], velocidad [m/s], Campo Simple [V/m], Campo Gap medio [V/m]',
               comments=' ', delimiter=', ')

    # merge_pickles_to_array(f'Results/Oxygen_forming_{num_simulation}.pkl', f'Results/Oxygen_reset_{num_simulation}.pkl',
    #                        f'Results/Oxygen_{num_simulation}.pkl')

    # merge_pickles_to_array(f'Results/Configurations_forming_{num_simulation}.pkl', f'Results/Configurations_reset_{num_simulation}.pkl',
    #                        f'Results/Configurations_{num_simulation}.pkl')

    # endregion

    # potencial = float(sim_ctes[num_simulation]["pb_metal_insul"])
    # ohm_resistence = float(sim_ctes[num_simulation]["ohm_resistence"])
    # I0 = float(sim_ctes[num_simulation]["I_0"])

    # Represento los datos de la simulación
    # pplt.plot_both(f'Results/resultados_{num_simulation}.csv',
    #                col_indices_x=1,
    #                col_indices_y=[2, 2],
    #                y_label='Intensidad [A]',
    #                save_path=f'Results/resultados_intensidad_{num_simulation}',
    #                global_tittle=fr'$I_0$ = {I0:.2e} A $R = ${ohm_resistence:.2e} $\Omega$',
    #                log_scale='y')

    #   global_tittle = fr'$\phi_{{B}}$ = {potencial} eV, $\varepsilon_r$ = {permitividad}, $I_0$ = {I0:.1e} A, $T_0$ = {T_0} K',
