from . import (
    Recombination,
    ElectricField,
    CurrentSolver,
    Representate,
    Percolation,
    Temperature,
    Generation,
    Montecarlo,
    exceptions,
    io_manager,
    Constants,
    findpath,
    utils,
)
from dataclasses import dataclass, fields

import matplotlib.pyplot as plt
import pandas as pd
import time as time
import numpy as np
import warnings
import pickle
import sys
import os

warnings.filterwarnings("error")


@dataclass
class SimulationParameters:
    device_size: float
    atom_size: float
    x_size: int
    y_size: int
    num_trampas: int
    init_simulation_time: float
    total_simulation_time: float
    num_pasos: int
    paso_guardar: int
    voltaje_final_reset: float
    voltaje_final_set: float
    initial_voltaje: float
    initial_current: float
    initial_elec_field: float
    initial_temperatura: float
    T_0: float

    @staticmethod
    def from_dict(d: dict):
        # Usamos los nombres de los campos y el tipo para castear automáticamente
        field_types = {f.name: f.type for f in fields(SimulationParameters)}
        # Alias para mapeos distintos y repetidos
        mapping = {
            "voltaje_final_reset": "voltaje_final",
            "T_0": "init_temp",
            "initial_temperatura": "init_temp",
        }
        # Para cada atributo, obtener del diccionario, castearlo y aplicar alias si corresponde
        kwargs = {}
        for k in field_types:
            src = mapping.get(k, k)
            kwargs[k] = field_types[k](d[src])
        return SimulationParameters(**kwargs)


def PP_set(
    paso_temporal: float,
    num_pasos: int,
    sim_ctes: list[dict],
    sim_parmtrs: list[dict],
    num_simulation: int,
    simulation_path: str,
    actual_state: np.ndarray,
):
    # Declaro todas las variables que voy a usar exclsivamente en la primera parte (PP) del set.

    ocupacion_max_pp_set = 0.35
    ocupacion_max_sp_set = 0.4

    num_max_vacantes = (device_size / atom_size) ** 2

    # region primera parte del set
    print("\nComienza la primera parte del set")
    # for k in tqdm(range(0, num_pasos)):
    for k in range(0, num_pasos):
        # Guardo el estado anterior
        last_state = actual_state

        # Actualizo el tiempo de simulación
        simulation_time = paso_temporal * k

        # Actualizo el voltaje
        voltage = vector_ddp[k]

        CF_matrix, CF_graph = CurrentSolver.Clean_state_matrix(actual_state)

        max_x, max_y = actual_state.shape
        filamentos = CurrentSolver.Clasificar_CF(
            CF_graph, max_x, max_y, filamentos_ranges
        )

        # print(f"{filamentos} \n")
        if any(~CF_creado):  # mientras haya alguno sin romper
            for i, v in enumerate(
                CurrentSolver.Existe_filamentos(filamentos, len(filamentos_ranges))
            ):
                if v and not CF_creado[i]:
                    CF_creado[i] = True
                    voltage_CF_creado[indice_filamento] = voltage
                    indice_filamento += 1
                    Representate.RepresentateState(
                        actual_state,
                        round(voltage, 3),
                        simulation_path
                        + f"Figures/Filamento_{i + 1}_creado_set_{num_simulation + 1}.png",
                    )
                    print(f"El filamento {i + 1} se ha creado en el voltaje {voltage}")

        num_vacantes[k] = np.sum(actual_state)

        # Si se llena el 90 del espacio de la matriz salto a otra simulación
        if np.sum(actual_state) > int(0.9 * num_max_vacantes):
            print(
                "\nSe ha llenado el 90% del espacio de la matriz en la iteración: ",
                k,
                " que corresponde al voltaje: ",
                voltage,
            )

            # Verifica si el sistema ha percolado
            if not sistema_percola:
                raise exceptions.NoPercolationException()

            k_ruptura = k

            voltaje_max_set = vector_ddp[k]
            config_matrix_recortada = config_matrix_pp_set[k, :, :]
            tiempo_pp_set = paso_temporal * (
                k - 1
            )  # Le quitamos un paso porque se ha superado el voltaje de ruptura
            resistencia_copy = resistencia.copy()

            print(
                "Voltaje final set",
                voltaje_max_set,
                "en el tiempo ",
                tiempo_pp_set,
                "en la iteración ",
                k_ruptura,
                "\n",
            )

            # Crear un array de ejemplo
            data_pp_set[k - 1 :] = np.nan  # Añadir valores nulos a partir de la fila k
            num_vacantes[k:] = np.nan  # Añadir valores nulos a partir de la fila k
            resistencia[k:] = np.nan  # Añadir valores nulos a partir de la fila k
            voltage_vector[k:] = np.nan  # Añadir valores nulos a partir de la fila k

            # Eliminar filas con valores nulos
            data_pp_set = data_pp_set[~np.isnan(data_pp_set).any(axis=1)]
            num_vacantes = num_vacantes[~np.isnan(num_vacantes)]
            resistencia = resistencia[~np.isnan(resistencia)]
            voltage_vector = voltage_vector[~np.isnan(voltage_vector)]

            # RepresentateState(resistance_matrix,voltaje_max_set, simulation_path + f'Figures/final_pp_set_resistance_{num_simulation+1}.png')
            break
        if voltage > voltaje_final_set:
            print(
                "\nSe ha superado el voltaje máximo del set en la iteracion: ",
                k,
                " que corresponde al voltaje: ",
                voltage,
            )

            # Verifica si el sistema ha percolado
            if not sistema_percola:
                raise exceptions.NoPercolationException()

            k_ruptura = k

            voltaje_max_set = vector_ddp[k]
            config_matrix_recortada = config_matrix_pp_set[k, :, :]
            tiempo_pp_set = paso_temporal * (
                k - 1
            )  # Le quitamos un paso porque se ha superado el voltaje de ruptura
            resistencia_copy = resistencia.copy()

            print(
                "Voltaje final set",
                voltaje_max_set,
                "en el tiempo ",
                tiempo_pp_set,
                "\n",
            )

            # Crear un array de ejemplo
            data_pp_set[k - 1 :] = np.nan  # Añadir valores nulos a partir de la fila k
            num_vacantes[k:] = np.nan  # Añadir valores nulos a partir de la fila k
            resistencia[k:] = np.nan  # Añadir valores nulos a partir de la fila k

            # Eliminar filas con valores nulos
            data_pp_set = data_pp_set[~np.isnan(data_pp_set).any(axis=1)]
            num_vacantes = num_vacantes[~np.isnan(num_vacantes)]

            resistencia = resistencia[~np.isnan(resistencia)]

            # RepresentateState(resistance_matrix, round(vector_ddp[k], 3), simulation_path + f'Figures/final_pp_set_resistance_{num_simulation+1}.png')
            break

        # Obtengo la corrriente, antes decido cual usar comprobando si ha percolado o no
        if Percolation.is_path(actual_state):
            if sistema_percola is False:
                voltaje_percolacion = voltage  # Guardo el voltaje de percolación
                print(
                    "\nEl sistema ha percolado en la iteración: ",
                    k,
                    " que corresponde con el voltaje: ",
                    voltaje_percolacion,
                )
                Representate.RepresentateState(
                    actual_state,
                    round(voltaje_percolacion, 3),
                    simulation_path + f"Figures/Percola_state_{num_simulation + 1}.png",
                )

                print(
                    "El sistema ha percolado cuando tiene una ocupación del: ",
                    (np.sum(actual_state) / (num_max_vacantes)) * 100,
                    " %",
                )

                if voltaje_percolacion >= 0.75:
                    # Si el voltaje de percolación es demasiado alto no va a coincidir con los datos experimentales
                    raise exceptions.HighPercolationVoltageException()

                # Cambio la probabilidad de generación de vacantes para controlar la percolación
                sim_ctes[num_simulation]["gamma"] = str(
                    float(sim_ctes[num_simulation]["gamma"]) / factor_generacion
                )
                print(
                    "Nueva gamma cuando percola el sistema: ",
                    sim_ctes[num_simulation]["gamma"],
                )

                r_termica = float(sim_ctes[num_simulation]["r_termica_percola"])

            sistema_percola = True

            # Copio el estado actual
            ac = actual_state.copy()

            # ac_clean_matriz, ac_clean_grid = CurrentSolver.Clean_state_matrix(actual_state)

            # cf_clasificados = CurrentSolver.Clasificar_CF(
            #     CF_graph, x_size, y_size, filamentos_ranges
            # )

            exist_cf = CurrentSolver.Existe_filamentos(
                filamentos, len(filamentos_ranges)
            )

            cf_clean_matrix = CurrentSolver.Eliminar_filamentos_incompletos(
                CF_graph, filamentos_ranges, exist_cf
            )

            densidad_filamento = np.sum(cf_clean_matrix) / (x_size * y_size)

            voltage_RRAM = voltage
            # Si ha percolado uso la corriente de Ohm
            try:
                current, resistencia[k] = CurrentSolver.OmhCurrent(
                    voltage, cf_clean_matrix, **sim_ctes[num_simulation]
                )

            except Warning:
                filename = (
                    simulation_path
                    + f"Null_Resistance/Configuration_Set_{voltage}_null_resistance.pkl"
                )
                print("Null resistance matrix in ", filename)
                Representate.RepresentateState(
                    cf_clean_matrix,
                    round(vector_ddp[k], 3),
                    simulation_path
                    + f"Figures/Null_Resistance/NULL_resistance_matrix_pp_set_{num_simulation + 1}.png",
                )
                with open(filename, "wb") as f:
                    pickle.dump(
                        {"actual_state": ac, "resistance_matrix": cf_clean_matrix}, f
                    )

        else:
            sistema_percola = False

            r_termica = float(sim_ctes[num_simulation]["r_termica_no_percola"])
            # Cambio el valor de gamma para favorer la generación de vacantes        # Si no ha percolado uso la corriente de Poole-Frenkel
            resistencia[k] = 0

            mean_field = np.mean(E_field_vector).item()
            current = CurrentSolver.Poole_Frenkel(
                temperatura, mean_field, **sim_ctes[num_simulation]
            ) * (device_size)  # type: ignore
            densidad_filamento = 0

        # Obtengo los valores del campo eléctrico y la temperatura
        E_field = ElectricField.SimpleElectricField(voltage, device_size)
        temperatura = Temperature.Temperature_Joule(
            voltage, current, sistema_percola, T_0, **sim_ctes[num_simulation]
        )

        # Genero el vector campo eléctrico
        for i in range(0, actual_state.shape[0]):
            E_field_vector[i] = ElectricField.GapElectricField(
                voltage, i, actual_state, **sim_parmtrs[num_simulation]
            )

        # calcular las probabilidades por fila
        prob_generacion_fila = np.minimum(
            [
                Generation.Generate(
                    paso_temporal,
                    E_field_vector[i],
                    temperatura,
                    **sim_ctes[num_simulation],
                )
                for i in range(x_size)
            ],
            1,
        )

        # Calculo la probabilidad de generación o recombinación para ello recorro toda la matriz
        for i in range(x_size):
            base_prob = prob_generacion_fila[i]
            for j in range(y_size):
                if actual_state[i, j] == 0:
                    if np.sum(actual_state) < max_vancantes_pp_set:
                        # Compruebo si tiene una vacate cerca
                        # np.sum(actual_state[i-1:i+1, j-1:j+1]) > 0:
                        if Generation.vecinos_horizontales(actual_state, i, j):
                            prob_generacion = base_prob * 1.1
                        else:
                            prob_generacion = base_prob * 0.9
                    else:
                        if "set_pp_vacantes_limit" not in locals():
                            print(
                                f"\nSe ha llenado el espacio de simulación al {ocupacion_max_pp_set * 100}%, se deja de generar vacantes"
                            )
                            set_pp_vacantes_limit = True
                        prob_generacion = 0  # LO hago para que no se generen más vacantes y no se llene el sistema

                    random_number = np.random.rand()
                    if random_number < prob_generacion:
                        actual_state[i, j] = 1  # Generación de una vacante

        if 1.2 < voltage < (1.6 - 0.2):  # 0.2 SOn valores inventados para q no entre
            # print("El resultado de la division es: ", k % paso_guardar_2)
            if k % paso_guardar_2 == 0:
                data_pp_set[k - 1] = np.array(
                    [
                        simulation_time,
                        voltage,
                        current,
                        temperatura,
                        E_field,
                        np.mean(E_field_vector),
                        0,
                        densidad_filamento,
                    ]
                )
                g_valor_list.append(Generation.evalutate_g(actual_state, size_grid=40))

        else:
            data_pp_set[k - 1] = np.array(
                [
                    simulation_time,
                    voltage,
                    current,
                    temperatura,
                    E_field,
                    np.mean(E_field_vector),
                    0,
                    densidad_filamento,
                ]
            )
            g_valor_list.append(Generation.evalutate_g(actual_state, size_grid=40))

        # Guardo el estado actual CADA paso_guardar PASOS MONTECARLO
        if k % paso_guardar == 0:
            config_matrix_pp_set[int(k / paso_guardar) - 1] = actual_state


if __name__ == "__main__":
    print("Iniciando la simulación RRAM")
