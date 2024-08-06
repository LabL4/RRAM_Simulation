import os
import pickle
import shutil
from re import A
import time as time
import pandas as pd

from RRAM import *
from tqdm import tqdm
from RRAM import Recombination
from RRAM import Plot_PostProcess

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

# quiero un bucle que recorra todas las simulaciones desde 0 hasta la longitud de sim_parmtrs-1

for num_simulation in range(len(sim_parmtrs)):

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

    voltaje_final = float(sim_parmtrs[num_simulation]['voltaje_final'])

    voltaje = float(sim_parmtrs[num_simulation]['initial_voltaje'])
    Corriente = float(sim_parmtrs[num_simulation]['initial_current'])
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
    vector_ddp = np.linspace(0, voltaje_final, num_pasos + 1)

    # Creo el vector de datos como una matriz de num_pasos filas y las columnas necesarias (x,y,probabilidad recombionacion, velocidad)
    colunm_number = 5
    data = np.zeros((num_pasos, colunm_number))
    # Creo el excel donde voy a sacar todos los datos
    df = pd.DataFrame(columns=['Tiempo simulacion', 'velocidad', 'desplazamiento',
                      'prob_generacion', 'probabilidad_recombinacion'])

    # Comienzo la simulación
    for k in tqdm(range(1, num_pasos+1)):
        # Guardo el estado anterior
        last_state = actual_state

        # Actualizo el tiempo de simulación
        simulation_time = paso_temporal * k

        # Calculo la corriente TODO: asasasasasa
        voltaje = vector_ddp[k]

        # voltaje += voltaje_final / paso_temporal

        # Obtengo la corrriente, antes decido cual usar comprobando si ha percolado o no
        # TODO: Revisar la corriente óhmica que no funciona
        if Percolation.is_path(actual_state):
            # Si ha percolado uso la corriente de percolación
            # Corriente = CurentSolver.OmhCurrent(Temperatura, Campo_Electrico)
            print(f"\n Ha percolado")
            break
        else:
            # Si no ha percolado uso la corriente de campo
            # TODO: REVISAR QUE LA CORRIENTE TIENE LAS UNIDADES CORRECTAS PORQUE NO CUADRAN VALORES.
            Corriente = CurentSolver.poole_frenkel(temperatura, E_field)/(1e-10)

        # Obtengo los valores del campo eléctrico y la temperatura
        E_field = SimpleElectricField(voltaje, device_size)
        # temperatura = Temperature_Joule(voltaje, Corriente, T_0=350) TODO: Estoy usando la temperatura constante

        # TODO: REVISAR PROBABILIDAD QUE A VECES SALE MAYOR DE 1
        # TODO: HACER UN REESCALADO DE LOS VALORES PARA EVITAR TENER QUE TRABAJAR CON NUMEROS TAN GRANDES
        prob_generacion = Generation.generation(paso_temporal, E_field, temperatura)

        # TODO: Revisa COMO SE GENERA LA PROBABILIDAD DE GENERACIÓN
        # Calculo la probabilidad de generación o recombinación para ello recorro toda la matriz
        for i in range(x_size):
            for j in range(y_size):
                if actual_state[i, j] == 0:
                    random_number = np.random.rand()
                    if random_number < prob_generacion:
                        actual_state[i, j] = 1  # Generación de una vacante

        if (simulation_time > 9) and (simulation_time < 9.2):
            pass

        # Genero los oxígenos
        oxygen_state = Recombination.Generate_Oxigen(oxygen_state, 5)

        # Muevo los oxígenos
        oxygen_state, velocidad, desplazamiento, senh = Recombination.Move_OxygenIons(
            paso_temporal, oxygen_state, temperatura, E_field, atom_size, **sim_ctes[num_simulation])

        # Obtengo la nueva configuración
        actual_state, oxygen_statem, pro_recombination = Recombination.Recombine(
            actual_state, oxygen_state, paso_temporal, velocidad, temperatura, **sim_ctes[num_simulation])

        data[k-1] = np.array([simulation_time, velocidad, desplazamiento, prob_generacion, pro_recombination])

        # Guardo el estado actual CADA paso_guardar PASOS MONTECARLO
        if k % paso_guardar == 0:
            config_matrix[int(k / paso_guardar) - 1] = actual_state
            oxygen_matrix[int(k / paso_guardar) - 1] = oxygen_state

    # Elimino de la matriz config_matrix las filas que no se han completado que ocurre cuando percola
    config_matrix = np.array([fila for fila in config_matrix if fila.sum() != 0.0])
    oxygen_matrix = np.array([fila for fila in oxygen_matrix if fila.sum() != 0.0])

    # Cuando acaba la simulacion guardo las matrices de configuraciones y oxigenos
    with open(f'Results/Configurations_{num_simulation}.pkl', 'wb') as f:
        pickle.dump(config_matrix, f)
    with open(f'Results/Oxygen_{num_simulation}.pkl', 'wb') as f:
        pickle.dump(oxygen_matrix, f)

    # Cuando percola no se completa la matriz de datos, por lo que la recorto
    data_filtrados = np.array([fila for fila in data if fila[-1] != 0.0])
    np.savetxt(f'Results/resultados_{num_simulation}.csv', data_filtrados, header='tiempo simulacion, velocidad, desplazamiento, prob_generacion, prob recombinacion',
               comments=' ', delimiter=', ')

    # Represento los datos de la simulación

    Plot_PostProcess.Plot_panel(f'Results/resultados_{num_simulation}.csv',
                                title=fr'$\gamma^{{drift}}$ = {sim_ctes[num_simulation]["drift_coefficient"]}, $E_m$ = {sim_ctes[num_simulation]["migration_energy"]} eV')
