import numpy as np
from RRAM import *
from tqdm import tqdm


def funcion_montecarlo(num_pasos, simulation_time, voltaje_final, factor_externo,
                       num_trampas=600, atom_size=0.25, espesor_dispositivo=10, temperatura_inicial=350.00,):

    eje_x = round(espesor_dispositivo / atom_size)
    eje_y = round(espesor_dispositivo / atom_size)

    # Inicializaciones
    paso_temporal = simulation_time / num_pasos

    # Defino la región que quiero privilegiar y su peso
    regiones_pesos = [
        ((10, eje_x-10, eje_y-15, eje_y), 50),             # First three columns with higher weight
        ((10, eje_x-10, 0, 15), 50),                       # First three columns with higher weight
    ]

    actual_state = Generation.initial_state_priv(eje_x, eje_y, num_trampas, regiones_pesos)

    # Configuraciones iniciales:
    temperatura = temperatura_inicial
    campo_electrico = 0
    voltaje = 0

    # Asumiendo que se guarda k, tiempo, i, j, prob_rec, espacio_recorr, funcion_trozos
    data = np.zeros((num_pasos*eje_x*eje_y, 7))

    re_index = 0

    # Comienza el algoritmo de Montecarlo
    for k in tqdm(range(1, num_pasos+1)):

        simulation_time = paso_temporal*k

        # Calculo la corriente
        voltaje += voltaje_final * paso_temporal

        # Obtengo la corriente, antes decido cual usar comprobando si ha percolado o no
        # TODO: Revisar la corriente óhmica que no funciona
        if Percolation.is_path(actual_state):
            # Si ha percolado uso la corriente de percolación
            # Corriente = CurentSolver.OmhCurrent(Temperatura, Campo_Electrico)
            print("Ha percolado")
            break
        else:
            # Si no ha percolado uso la corriente de campo
            # TODO: REVISAR QUE LA CORRIENTE TIENE LAS UNIDADES CORRECTAS PORQUE NO CUADRAN VALORES cuando coloco la superficie
            corriente = CurentSolver.poole_frenkel(temperatura, campo_electrico)

        # Obtengo los valores del campo eléctrico y la temperatura
        campo_electrico = SimpleElectricField(voltaje, espesor_dispositivo*1e-9)
        # Temperatura = Temperature_Joule(voltaje, Corriente, T_0=350)

        # Calculo la probabilidad de generación o recombinación para ello recorro toda la matriz
        for i in range(eje_x):
            for j in range(eje_y):
                if actual_state[i, j] == 0:
                    # TODO: REVISAR PROBABILIDAD QUE A VECES SALE MAYOR DE 1
                    # TODO: HACER UN REESCALADO DE LOS VALORES PARA EVITAR TENER QUE TRABAJAR CON NUMEROS TAN GRANDES
                    prob_generacion = Generation.generation(paso_temporal, campo_electrico, temperatura)
                    random_number = np.random.rand()
                    if random_number < prob_generacion:
                        actual_state[i, j] = 1  # Generación

                if actual_state[i, j] == 1:
                    # TODO: REVISAR PROBABILIDAD QUE A VECES SALE MAYOR DE 1
                    # TODO: HACER UN REESCALADO DE LOS VALORES PARA EVITAR TENER QUE TRABAJAR CON NUMEROS TAN GRANDES
                    prob_rec, espacio_recorr, funcion_trozos = Recombination.recombination(
                        paso_temporal, i, campo_electrico, temperatura, atom_size, factor=factor_externo)
                    data[re_index] = np.array([k, simulation_time, i, j, prob_rec, espacio_recorr, funcion_trozos])
                    re_index += 1

                    # genero un número aleatorio entre 0 y 1
                    random_number = np.random.rand()
                    if random_number < prob_rec:
                        actual_state[i, j] = 0  # Recombinación
    return data
