"""Fases del ciclo SET de la simulación RRAM (PP_set y SP_set)."""

import sys
from pathlib import Path
from typing import List

import numpy as np

from . import (
    CurrentSolver,
    ElectricField,
    Generation,
    Percolation,
    Temperature,
    exceptions,
    utils,
)
from .constants_simulation import SimulationConstants
from .filament_tracking import (
    actualizar_parametros_por_filamento,
    procesar_filamentos_creados,
)
from .parameters import SimulationParameters
from .state_updates import update_state_generation
import logging

logger = logging.getLogger(__name__)


def PP_set(
    num_simulation: int,
    params: SimulationParameters,
    sim_ctes: SimulationConstants,
    CF_ranges: List[tuple],
    CF_creado: np.ndarray,
    CF_centros: List[int] | None = None,
    actual_state: np.ndarray | None = None,
):
    """
    Executes the first part (PP) of the simulation set process for a resistive random-access memory (RRAM) device.
    This function simulates the physical processes involved in the formation of conductive filaments (CFs)
    in an RRAM device during the set operation. It initializes the simulation environment, updates the
    state of the system, and records the results at each simulation step.

    Args:
        num_simulation (int): The simulation number (used for naming and saving results).
        params (SimulationParameters): Object containing simulation parameters such as device size,
            total simulation time, number of steps, and voltage settings.
        sim_ctes (SimulationConstants): Object containing simulation constants such as gamma,
            factor of generation, and material properties.
        CF_ranges (List[tuple]): List of tuples defining the ranges for conductive filaments.
        CF_creado (np.ndarray): Boolean array indicating whether each conductive filament has been created.
        CF_centros (List[int] | None): Centro vertical de cada filamento esperado.
        actual_state (np.ndarray | None): Estado inicial precargado. Si es None,
            se carga desde `Init_data/init_state_{num_simulation - 1}` por
            compatibilidad hacia atrás. La nueva arquitectura init→exec→plot
            siempre debería pasarlo explícito.

    Raises:
        exceptions.MaxVacantesException: Raised if the maximum number of vacancies is exceeded.
        exceptions.NoPercolationException: Raised if the system does not percolate.
        exceptions.HighPercolationVoltageException: Raised if the percolation voltage is too high.
        exceptions.NullResistanceException: Raised if the resistance becomes null during the simulation.

    Returns:
        dict: Estado final del PP set con todas las variables necesarias para SP_set.
    """
    np.set_printoptions(threshold=sys.maxsize)
    # Declaro todas las variables que voy a usar exclusivamente en la primera parte (PP) del set.
    rutas = utils.crear_rutas_simulacion(num_simulation=num_simulation, state="set")

    rutas["simulation_path"].mkdir(parents=True, exist_ok=True)
    rutas["figures_path"].mkdir(parents=True, exist_ok=True)
    rutas["data_simulation_path"].mkdir(parents=True, exist_ok=True)

    # Cargo el estado inicial: prioridad al argumento explícito, fallback al disco.
    if actual_state is None:
        actual_state = utils.cargar_estado(Path.cwd() / f"Init_data/init_state_{num_simulation - 1}")
    else:
        actual_state = actual_state.copy()

    sistema_percola = False
    total_vacantes_pp_set = False
    num_pasos_guardar_estado = 100
    cf_clean_matrix = None
    voltaje_percolacion = params.voltaje_final_set

    # AL inicio como la corriente es de tipo poole frenkel, la resitencia ohmica se considera nula
    resistencia = 0.0
    temperatura_anterior = params.init_temp
    pendiente_temperatura = sim_ctes.pendiente_temperatura

    logger.info(f"Los valores de factor vecinos y factor libre son: {sim_ctes.factor_vecinos_pp_set} y {sim_ctes.factor_libre_pp_set} ")

    max_vancantes_pp_set = int(sim_ctes.ocupacion_max_pp_set * params.num_max_vacantes)
    voltage_CF_creado = np.full(len(CF_ranges), 0.0)
    creaciones_dict: dict = {}  # Mismo formato que roturas_dict
    filamentos_previos = 0

    # Inicializo vectores donde almaceno datos y condiciones iniciales
    temperatura = params.init_temp
    current = 0.0
    E_field_vector = np.zeros((actual_state.shape[0]), dtype=np.float64)
    vector_ddp = np.arange(0.000, params.voltaje_final_reset + params.paso_potencial_set, params.paso_potencial_set)
    logger.info(f"El paso de potencial para la parte de set es: {params.paso_potencial_set} ")

    centros_calculados = CF_centros

    num_columnas = 3  # Tiempo, Voltaje, Intensidad

    # Defino la matriz para almacenar los datos
    data_pp_set = np.zeros((params.num_pasos, num_columnas), dtype=np.float64)
    resistencia_vector = np.zeros((params.num_pasos, num_columnas), dtype=np.float64)
    num_vacantes_total = np.zeros((params.num_pasos, num_columnas), dtype=np.float64)

    logger.info(f"El grosor de los filamentos es de {sim_ctes.grosor_filamento} filas")

    # creo la mascara para limitar la generacion a solo la zona del esperado filamento
    limit_CF_witdh_mask_generation = Generation.create_custom_mask(
        state=actual_state, centros_CF=CF_centros, grosor_CF=sim_ctes.grosor_filamento
    )

    # Inicializamos las matrices variables a None para que existan desde el principio
    cf_clean_matrix = None
    probabilidad_matrix = None
    matriz_para_plot_muro = None

    # `filas_intermedias` y `dist_casillas` solo se calculan cuando hay >=2
    # filamentos (muros térmicos entre filamentos). Para 1 filamento se
    # quedan en None y el bloque de muros térmicos se salta.
    filas_intermedias = None
    dist_casillas = None
    mis_perfiles_extraidos = None

    logger.info(f"El valor de gamma es: {sim_ctes.gamma} ")

    logger.info(f"Simulacion {num_simulation} - Primera parte del set")
    all_CFs_created = False

    for k in range(0, params.num_pasos + 1):
        total_vacantes = np.sum(actual_state)

        if total_vacantes > int(params.num_max_vacantes):
            # Si se llena el 90 del espacio de la matriz salto a otra simulación. Ponerlo aqui puede dar el problema de que nada mas empezar esté lleno y de error, pero eso NO debe pasar asi q no me preocupa.
            raise exceptions.MaxVacantesException(k=k - 1, voltage=vector_ddp[k - 1])
        else:
            # Verifica si el sistema ha percolado
            if (k == params.num_pasos - 1) and (not sistema_percola):
                raise exceptions.NoPercolationException()

        # Actualizo el tiempo de simulación y el voltaje
        simulation_time = params.paso_temporal * k
        voltage = vector_ddp[k]

        # Genero el vector campo eléctrico
        for i in range(0, params.x_size):
            E_field_vector[i] = ElectricField.GapElectricField(
                potential=voltage,
                pos_y=i,
                actual_state=actual_state,
                device_size_x=params.device_size_x,
                grid_size=params.atom_size,
            )

        # Verifica si el sistema ha percolado
        if voltage >= params.voltaje_final_set:
            if not sistema_percola:
                raise exceptions.NoPercolationException()

            voltaje_max_set = vector_ddp[k]
            tiempo_pp_set = params.paso_temporal * (
                k - 1
            )  # Le quitamos un paso porque se ha superado el voltaje de ruptura

            logger.info(f"Voltaje final set {voltaje_max_set} en el tiempo {tiempo_pp_set} ")
            # Elimino las filas sobrantes del array de datos y las lleno de nans para eliminarlas luego
            data_pp_set[k - 1 :] = np.nan  # Añadir valores nulos a partir de la fila k
            data_pp_set = data_pp_set[~np.isnan(data_pp_set).any(axis=1)]  # Eliminar filas con valores nulos
            break

        # Obtengo la corrriente, antes decido cual usar comprobando si ha percolado o no
        if Percolation.is_path(actual_state):
            # Si es la primera vez que percola, siste_percola será falso y entra aquí
            if sistema_percola is False:
                voltaje_percolacion = voltage  # Guardo el voltaje de percolación
                ocupacion_percola = np.sum(actual_state)
                logger.info(f"\nEl sistema ha percolado en la iteración: {k}  que corresponde con el voltaje: {round(voltaje_percolacion, 5)}  con una ocupación del: {round((np.sum(actual_state) / (params.num_max_vacantes)), 4) * 100} que corresponde con un numero de vacantes de: {int(np.sum(actual_state))} ")

                if voltaje_percolacion >= sim_ctes.lim_voltage_percolacion:
                    # Si el voltaje de percolación es demasiado alto no va a coincidir con los datos experimentales, y no merece la pena seguir con la simulación
                    raise exceptions.HighPercolationVoltageException(voltage_percola=voltaje_percolacion)

                # Verificar si temperatura es un float y convertirlo a matriz
                if isinstance(temperatura, (float, int)):
                    temperatura_anterior = np.full_like(actual_state, temperatura, dtype=float)

            sistema_percola = True

            actual_state_clean_CF, CF_graph = CurrentSolver.Clean_state_matrix(actual_state)
            filamentos = CurrentSolver.Clasificar_CF(CF_graph, actual_state, CF_ranges)
            exist_cf = CurrentSolver.Existe_filamentos(filamentos, len(CF_ranges))

            # Compruebo si hay filamentos nuevos
            if any(~CF_creado):
                procesar_filamentos_creados(
                    imagen_path=rutas["figures_path"],
                    data_save_path=rutas["data_simulation_path"],
                    existentes=exist_cf,
                    CF_creado=CF_creado,
                    voltage=voltage,
                    voltage_CF_creado=voltage_CF_creado,
                    actual_state=actual_state,
                    num_simulation=num_simulation,
                    creaciones_dict=creaciones_dict,
                    etapa="pp_set",
                )

            filamentos_actuales = sum(CF_creado)

            # Solo entramos si el número de filamentos ha AUMENTADO en este paso
            if filamentos_actuales > filamentos_previos:
                sim_ctes, all_CFs_created, filas_int_temp, dist_temp = actualizar_parametros_por_filamento(
                    filamentos_actuales,
                    filamentos_previos,
                    CF_ranges,
                    CF_centros,  # type: ignore
                    sim_ctes,
                    all_CFs_created,
                )

                # Actualizar variables locales de PP_set si es necesario
                if filas_int_temp is not None:
                    filas_intermedias = filas_int_temp
                    dist_casillas = dist_temp

                # Actualizamos el historial para que no vuelva a entrar en iteraciones futuras
                filamentos_previos = filamentos_actuales

            cf_clean_matrix = CurrentSolver.Eliminar_filamentos_incompletos(CF_graph, CF_ranges, exist_cf, actual_state)

            # Si ha percolado uso la corriente de Ohm
            try:
                current, resistencia = CurrentSolver.OmhCurrent(
                    voltage, cf_clean_matrix, ohm_resistence=sim_ctes.ohm_resistence_set
                )
            except ZeroDivisionError:
                raise exceptions.NullResistanceException(
                    simulation_path=rutas["simulation_path"],
                    voltage=voltage,
                    num_simulation=num_simulation,
                    actual_state=actual_state,
                )

            if all_CFs_created:
                # El sistema percola por lo que resuelvo la ecuación del calor. Primero se obtiene el mapa de materiales
                materials_map = Temperature.crear_matriz_materiales(cf_clean_matrix)

                # Cáculo de las fuentes de calor (el filamento es el que emite calor, el resto no)
                Q_source_map = Temperature.calculate_heat_source(
                    types_map=materials_map,
                    atom_size=params.atom_size,
                    I_total=current,
                    R_cell=sim_ctes.ohm_resistence_set,
                    factor_generar_calor=sim_ctes.factor_generar_calor,
                )

                # Compruebo si temperatura es un float, si es así se lanza una excepción porque no se puede extraer el perfil, si no es así se asume que es una matriz y se lanza la función de extracción
                if isinstance(temperatura_anterior, (float, int)):
                    raise ValueError(
                        "La temperatura no se ha calculado como matriz, no se pueden extraer los perfiles de los filamentos. Se esperaba una matriz de temperaturas, pero se ha recibido un valor escalar."
                    )
                else:
                    mis_perfiles_extraidos = Temperature.extraer_perfiles_filamentos(
                        matriz_temperaturas=temperatura_anterior,
                        filas_centros=centros_calculados,  # type: ignore
                    )

                # =====================================================================
                # 5. CÁLCULO DE LOS PERFILES PARA LOS MUROS Y COLOCACIÓN
                # =====================================================================
                # Solo aplica con >=2 filamentos: hay paredes intermedias.
                # Con 1 filamento, filas_intermedias y dist_casillas siguen None
                # tras `actualizar_parametros_por_filamento`; saltamos los muros.
                if filas_intermedias is not None and dist_casillas is not None:
                    perfiles_muros_calculados = Temperature.calcular_perfiles_muro(
                        perfiles_filamentos=mis_perfiles_extraidos,
                        distancias_casillas=dist_casillas,
                        pendiente_temperatura=pendiente_temperatura,
                        atom_size=params.atom_size,
                        T_ambient=params.init_temp,
                    )

                    matriz_temperaturas_fijas = Temperature.colocar_muro_termico(
                        matriz_molde=actual_state_clean_CF,
                        filas_intermedias=filas_intermedias,
                        perfiles_muros_calculados=perfiles_muros_calculados,
                    )

                    # Añadimos columnas de ceros (donde no hay muro) en las posiciones de los electrodos
                    Ny = matriz_temperaturas_fijas.shape[0]
                    columna_ceros = np.zeros((Ny, 1))
                    matriz_temperaturas_fijas_final = np.hstack([columna_ceros, matriz_temperaturas_fijas, columna_ceros])

                    temperatura = Temperature.solve_thermal_state(
                        types_map=materials_map,
                        Q_map=Q_source_map,
                        thermal_props=sim_ctes.propiedades_termicas,
                        atom_size=params.atom_size,
                        T_ambient=params.init_temp,
                        matriz_muros=matriz_temperaturas_fijas_final,
                    )
                else:
                    # Caso 1 filamento (sin muros): resolver térmica sin matriz_muros.
                    temperatura = Temperature.solve_thermal_state(
                        types_map=materials_map,
                        Q_map=Q_source_map,
                        thermal_props=sim_ctes.propiedades_termicas,
                        atom_size=params.atom_size,
                        T_ambient=params.init_temp,
                        matriz_muros=None,
                    )

                # Actualizo la temperatura anterior para el siguiente paso, NO guardo las columnas primera y ultima ya q corresponden a los electrodos
                temperatura_anterior = temperatura[:, 1:-1]

            else:
                temperatura = Temperature.Temperature_Joule(
                    voltage, current, T_0=params.init_temp, r_termica=sim_ctes.r_termica_no_percola * 5
                )
                # Extiendo el valor para formar una matriz del mismo tamaño que el estado, para que no de error al usarlo en la función de generación si no ha percolado
                temperatura = np.full_like(actual_state, temperatura)

        else:
            sistema_percola = False
            mean_field = np.mean(E_field_vector).item()

            # Obtengo los valores del campo eléctrico y la temperatura
            temperatura = Temperature.Temperature_Joule(
                voltage, current, T_0=params.init_temp, r_termica=sim_ctes.r_termica_no_percola
            )
            # print(f"El valor de la temperatura es {temperatura} K, se usa el modelo de temperatura de Joule\n")

            # TODO Confirmar que la corriente es por device_size_y
            # simple_field = ElectricField.SimpleElectricField(voltage, params.device_size)
            # Si no ha percolado uso la corriente de Poole-Frenkel
            if not total_vacantes_pp_set:
                current = CurrentSolver.Poole_Frenkel(
                    temperatura,
                    mean_field,
                    pb_metal_insul=sim_ctes.pb_metal_insul_set,
                    permitividad_relativa=sim_ctes.permitividad_relativa_set,
                    I_0=sim_ctes.I_0_set,
                ) * (params.device_size_y)

        if total_vacantes < max_vancantes_pp_set:
            # Actualizo el estado del sistema
            actual_state, probabilidad_matrix = update_state_generation(
                actual_state,
                params,
                sim_ctes,
                E_field_vector,
                temperatura,
                sim_ctes.factor_vecinos_pp_set,
                sim_ctes.factor_libre_pp_set,
                max_vancantes_pp_set,
                custom_mask=limit_CF_witdh_mask_generation,
            )

        elif not total_vacantes_pp_set:
            logger.info(f"\nSe ha alcanzado la ocupación máxima del {sim_ctes.ocupacion_max_pp_set * 100}% en la primera parte del set en el paso {k}.")
            total_vacantes_pp_set = True

        # region GUARDAR ESTADO

        # Guardo los datos de la simulación
        data_pp_set[k] = np.array([simulation_time, voltage, current])

        if locals().get("resistencia") is not None:
            resistencia_vector[k] = np.array([k, voltage, resistencia])
        else:
            resistencia_vector[k] = np.array([k, voltage, 0])

        num_vacantes_total[k] = np.array([k, voltage, total_vacantes])

        if k % num_pasos_guardar_estado == 0:
            # Si se ha creado la matriz de temperaturas fijas es porque se ha creado el muro y entonces tiene sentido guardarlo.
            if locals().get("matriz_temperaturas_fijas") is not None:
                matriz_para_plot_muro = np.copy(matriz_temperaturas_fijas)
                for centro, perfil_filamento in zip(centros_calculados, mis_perfiles_extraidos):  # type: ignore
                    if centro is not None and perfil_filamento is not None:
                        matriz_para_plot_muro[centro, :] = perfil_filamento

            # Guardo las variables del estado
            utils.guardar_estado_intermedio(
                ruta_destino=rutas["data_simulation_path"],
                etapa="pp_set",
                num_simulation=num_simulation,
                k=k,
                actual_state=actual_state,
                cf_clean_matrix=locals().get("cf_clean_matrix"),
                temperatura=locals().get("temperatura"),
                probabilidad_matrix=locals().get("probabilidad_matrix"),
                matriz_para_plot_muro=locals().get("matriz_para_plot_muro"),
            )
            # endregion

    # Muestro el valor de temperatura más alto alcanzado en la simulación
    logger.info(f"\nLa temperatura máxima alcanzada en la simulación ha sido de: {round(np.max(temperatura), 4)} K")

    # Si no se fomran los flamentos esperados se decarta la simulación
    if filamentos_actuales < len(CF_ranges):
        raise exceptions.FilamentosNoFormadosException(
            simulation_path=rutas["simulation_path"],
            num_simulation=num_simulation,
            actual_state=actual_state,
            CF_formados=filamentos_actuales,
            CF_esperados=len(CF_ranges),
        )

    data_encabezados = {
        "datos_simulacion": "Tiempo [s],Voltaje [V],Intensidad [A]",
        "vacantes": "paso, Voltaje [V], Total Vacantes",  # Correcto, porque solo hay 1 columna
        "resistencia": "paso, Voltaje [V], Resistencia [Ohm]",  # Correcto, porque solo hay 1 columna
    }

    # Guardo los datos de la simulacion
    utils.guardar_datos(
        save_path_data=rutas["simulation_path"] / f"Data_pp_set_{num_simulation}",
        headers=data_encabezados,
        datos_sim=data_pp_set,
        vacantes=num_vacantes_total,
        resistencia=resistencia_vector,
    )

    np.save(rutas["simulation_path"] / f"Final_state_{num_simulation}_pp_set.npz", actual_state)

    # Se decarta la simulación si no se ha llegado a la resistencia mínima necesaria para la segunda parte del set, ya que no va a coincidir con los datos experimentales.
    # if not (35 <= resistencia <= 55):
    #     # No se ha llegado a la resistencia necesaria para la segunda parte del set, directamelo lo descarto
    #     raise exceptions.LowResistanceException(valor_resistencia=resistencia)

    # Guardo todas las variables del estado final del PP set para usarlas en el PS set
    final_state_pp_set = {
        "actual_state": actual_state,
        "cf_clean_matrix": cf_clean_matrix,
        "sistema_percola": sistema_percola,
        "k_maxima": k,
        "sim_ctes": sim_ctes,
        "params": params,
        "Temperatura_final": temperatura,
        "voltaje_max_set": voltaje_max_set,
        "voltaje_percolacion": voltaje_percolacion,
        "tiempo_pp_set": simulation_time,
        "current_final": current,
        "ocupacion_percola": ocupacion_percola,
        "intensidad_final": current,
        "CF_centros": CF_centros,
        "creaciones_dict": creaciones_dict,
    }

    return final_state_pp_set


def SP_set(
    final_state_pp_set: dict,
    num_simulation: int,
    CF_ranges: List[tuple],
) -> dict:
    """
    Simulates the second part of the "set" process in a resistive switching device.
    This function performs a simulation of the "set" process, updating the state of the system
    based on the applied voltage, electric field, and other parameters. It handles the generation
    of vacancies, checks for percolation, calculates the current, and updates the temperature
    and system state. The results of the simulation are saved to files, and the final state is
    returned for further use.
    Args:
        final_state_pp_set (dict): A dictionary containing the final state of the previous
            simulation step (PP set). It includes the following keys:
            - "actual_state": The current state of the system.
            - "k_maxima": The maximum number of simulation steps.
            - "sistema_percola": Boolean indicating if the system has percolated.
            - "sim_ctes": Simulation constants.
            - "params": Simulation parameters.
            - "voltaje_max_set": Maximum voltage for the set process.
            - "Temperatura_final": Final temperature from the previous step.
        num_simulation (int): The simulation number, used for saving results.
        CF_ranges (List[tuple]): A list of tuples defining the ranges for classifying conductive filaments.
    Returns:
        dict: A dictionary containing the final state of the system after the "set" process.
        It includes the following keys:
            - "actual_state": The updated state of the system.
            - "sim_ctes": Updated simulation constants.
            - "params": Simulation parameters.
            - "Temperatura_final": Final temperature after the "set" process.
    Raises:
        exceptions.MaxVacantesException: If the maximum number of vacancies is exceeded.
        exceptions.NoPercolationException: If the system does not percolate by the end of the simulation.
        exceptions.NullResistanceException: If a division by zero occurs during current calculation.
    Notes:
        - The function saves the simulation data, final state, and figures to specified paths.
        - It uses various helper functions and classes for tasks such as electric field calculation,
          current calculation, and state updates.
        - The simulation stops early if the maximum vacancy occupancy is reached.
    """

    # Extraigo las variables del estado final del PP set
    actual_state = final_state_pp_set["actual_state"]
    logger.info(f"El número inicial de vacantes es: {np.sum(actual_state)}")

    k_max = final_state_pp_set["k_maxima"] - 1
    sistema_percola = final_state_pp_set["sistema_percola"]
    sim_ctes = final_state_pp_set["sim_ctes"]
    params = final_state_pp_set["params"]
    voltaje_max_set = final_state_pp_set["voltaje_max_set"]
    tiempo_pp_set = final_state_pp_set["tiempo_pp_set"]
    current = final_state_pp_set["current_final"]
    max_vancantes_pp_set = final_state_pp_set["ocupacion_percola"]
    current = final_state_pp_set["intensidad_final"]
    CF_centros = final_state_pp_set["CF_centros"]
    cf_clean_matrix = final_state_pp_set["cf_clean_matrix"]

    logger.info(f"El numero de vacantes al inicio del SP set es: {np.sum(actual_state)}")
    logger.info(f"El numero de vacantes al percolar fue : {max_vancantes_pp_set}")

    ocupacion_max_sp_set = 0 + 0  # 0.35
    max_vancantes_sp_set = max_vancantes_pp_set + int(ocupacion_max_sp_set * params.num_max_vacantes)
    total_vacantes_sp_set = False
    num_pasos_guardar_estado = 100
    rutas = utils.crear_rutas_simulacion(num_simulation=num_simulation, state="set")

    temperatura_anterior = final_state_pp_set["Temperatura_final"]

    # Elimino las columnas 0 y ultima de la matriz de temperatura porque corresponden a los electrodos
    temperatura_anterior = final_state_pp_set["Temperatura_final"][:, 1:-1]

    logger.info(f"El paso de potencial para sp set es: {params.paso_potencial_set} ")
    vector_ddp = np.arange(voltaje_max_set, 0.000, -params.paso_potencial_set)

    E_field_vector = np.zeros((actual_state.shape[0]), dtype=np.float64)

    # Defino la matriz para almacenar los datos
    num_columnas = 3  # Tiempo, Voltaje, Intensidad
    data_sp_set = np.zeros((k_max, num_columnas), dtype=np.float64)

    filas_intermedias, dist_casillas = Temperature.calcular_filas_intermedias(CF_centros)

    logger.info(f"Los centros calculados de los filamentos son: {CF_centros}")
    logger.info(f"Las filas intermedias calculadas son: {filas_intermedias}")
    logger.info(f"La distancia entre casillas es: {dist_casillas} ")

    actual_state_clean_CF, _ = CurrentSolver.Clean_state_matrix(actual_state)

    # cf_clean_matrix = CurrentSolver.Eliminar_filamentos_incompletos(CF_graph, CF_ranges, exist_cf)
    # _, cf_clean_matrix = CurrentSolver.limitar_grosor_filamentos(
    #     actual_state,
    #     cf_clean_matrix,
    #     centros_calculados,
    #     sim_ctes.grosor_filamento,
    #     CF_ranges,
    # )

    logger.info(f"Simulacion {num_simulation} - Segunda parte del set")
    for k in range(0, k_max):
        total_vacantes = np.sum(actual_state)

        if total_vacantes > int(params.num_max_vacantes):
            # Si se llena el 90 del espacio de la matriz salto a otra simulación. Ponerlo aqui puede dar el problema de que nada mas empezar esté lleno y de error, pero eso NO debe pasar asi q no me preocupa.
            raise exceptions.MaxVacantesException(k=k - 1, voltage=vector_ddp[k - 1])
        else:
            # Verifica si el sistema ha percolado
            if (k == params.num_pasos - 1) and (not sistema_percola):
                raise exceptions.NoPercolationException()

        # Actualizo el tiempo de simulación y el voltaje
        simulation_time = params.paso_temporal * k
        voltage = vector_ddp[k]

        # Genero el vector campo eléctrico
        for i in range(0, params.x_size):
            E_field_vector[i] = ElectricField.GapElectricField(
                voltage, i, actual_state, device_size_x=params.device_size_x, grid_size=params.atom_size
            )

        # Obtengo la corrriente, comprobado si ha percolado o no
        if Percolation.is_path(actual_state):
            # region

            # TODO: Si el sistema llega al maximo de vacante, como no genera mas, no hace falta recalcular los filamentos, ya que no van a cambiar if total_vacantes < max_vancantes_sp_set: estaba dando error lo he quitado
            # actual_state_clean_CF, CF_graph = CurrentSolver.Clean_state_matrix(actual_state)

            # filamentos = CurrentSolver.Clasificar_CF(CF_graph, params.x_size, params.y_size, CF_ranges)
            # exist_cf = CurrentSolver.Existe_filamentos(filamentos, len(CF_ranges))
            # cf_clean_matrix = CurrentSolver.Eliminar_filamentos_incompletos(CF_graph, CF_ranges, exist_cf)
            # _, cf_clean_matrix = CurrentSolver.limitar_grosor_filamentos(
            #     actual_state,
            #     cf_clean_matrix,
            #     centros_calculados,
            #     sim_ctes.grosor_filamento,
            #     CF_ranges,
            # )

            # TODO: Si el sistema llega al maximo de vacante, como no genera mas, no hace falta recalcular los filamentos, ya que no van a cambiar if total_vacantes < max_vancantes_sp_set
            # print("Número de vacantes en el estado actual limpio de CF:", np.sum(cf_clean_matrix))

            # obtengo los centros de los CF
            # centros_calculados = Temperature.obtener_centro_CF(actual_state_clean_CF, cf_ranges=CF_ranges)

            # before_state = actual_state.copy()  # Copia del estado antes de limitar el grosor de los filamentos
            # before_clean_matrix = (
            #     cf_clean_matrix.copy()
            # )  # Copia de la matriz limpia antes de limitar el grosor de los filamentos
            # # Limito el grosor de los filamentos a un máximo de grosor_filamento celdas
            # new_actual_state, new_cf_clean_matrix = CurrentSolver.limitar_grosor_filamentos(
            #     actual_state,
            #     cf_clean_matrix,
            #     centros_calculados,
            #     sim_ctes.grosor_filamento,
            #     CF_ranges,
            # )

            # vacantes_finales = np.sum(new_actual_state)

            # cf_clean_matrix = new_cf_clean_matrix.copy()
            # actual_state = new_actual_state.copy()

            # if vacantes_finales != total_vacantes:
            #     print(
            #         f"\nSe han eliminado {total_vacantes - vacantes_finales} vacantes para limitar el grosor de los filamentos.\n"
            #     )
            # endregion

            # Si ha percolado uso la corriente de Ohm
            try:
                current, _ = CurrentSolver.OmhCurrent(
                    voltage, cf_clean_matrix, ohm_resistence=sim_ctes.ohm_resistence_set
                )
            except ZeroDivisionError:
                raise exceptions.NullResistanceException(
                    simulation_path=rutas["simulation_path"],
                    voltage=voltage,
                    num_simulation=num_simulation,
                    actual_state=actual_state,
                )

            # El sistema percola por lo que resuelvo la ecuación del calor. Primero se obtiene el mapa de materiales
            materials_map = Temperature.crear_matriz_materiales(cf_clean_matrix)
            # Cáculo de las fuentes de calor (el filamento es el que emite calor, el resto no)
            Q_source_map = Temperature.calculate_heat_source(
                types_map=materials_map,
                atom_size=params.atom_size,
                I_total=current,
                R_cell=sim_ctes.ohm_resistence_set,
                factor_generar_calor=sim_ctes.factor_generar_calor,
            )

            # 4. LLAMAMOS A LA FUNCIÓN DE EXTRACCIÓN
            # Compruebo si temperatura es un float, si es así se lanza una excepción porque no se puede extraer el perfil, si no es así se asume que es una matriz y se lanza la función de extracción
            if isinstance(temperatura_anterior, (float, int)):
                raise ValueError(
                    "La temperatura no se ha calculado como matriz, no se pueden extraer los perfiles de los filamentos. Se esperaba una matriz de temperaturas, pero se ha recibido un valor escalar."
                )
            else:
                mis_perfiles_extraidos = Temperature.extraer_perfiles_filamentos(
                    matriz_temperaturas=temperatura_anterior, filas_centros=CF_centros
                )

            # CÁLCULO DE LOS PERFILES PARA LOS MUROS Y COLOCACIÓN
            perfiles_muros_calculados = Temperature.calcular_perfiles_muro(
                perfiles_filamentos=mis_perfiles_extraidos,
                distancias_casillas=dist_casillas,
                pendiente_temperatura=sim_ctes.pendiente_temperatura,
                atom_size=params.atom_size,
                T_ambient=params.init_temp,
            )
            matriz_temperaturas_fijas = Temperature.colocar_muro_termico(
                matriz_molde=actual_state_clean_CF,
                filas_intermedias=filas_intermedias,
                perfiles_muros_calculados=perfiles_muros_calculados,
            )
            # Añadimos columnas de ceros (donde no hay muro) en las posiciones de los electrodos
            Ny = matriz_temperaturas_fijas.shape[0]
            columna_ceros = np.zeros((Ny, 1))
            matriz_temperaturas_fijas_final = np.hstack([columna_ceros, matriz_temperaturas_fijas, columna_ceros])

            temperatura = Temperature.solve_thermal_state(
                types_map=materials_map,
                Q_map=Q_source_map,
                thermal_props=sim_ctes.propiedades_termicas,
                atom_size=params.atom_size,
                T_ambient=params.init_temp,
                matriz_muros=matriz_temperaturas_fijas_final,
            )

            # Importante para que se actualice el perfil termico de los filamentos
            temperatura_anterior = temperatura[:, 1:-1]

            if k % num_pasos_guardar_estado == 0:
                if "matriz_temperaturas_fijas" in locals():
                    matriz_para_plot_muro = np.copy(matriz_temperaturas_fijas)
                    for centro, perfil_filamento in zip(CF_centros, mis_perfiles_extraidos):
                        if centro is not None and perfil_filamento is not None:
                            matriz_para_plot_muro[centro, :] = perfil_filamento

                # Guardo las variables del estado
                utils.guardar_estado_intermedio(
                    ruta_destino=rutas["data_simulation_path"],
                    etapa="sp_set",
                    num_simulation=num_simulation,
                    k=k,
                    actual_state=actual_state,
                    cf_clean_matrix=locals().get("cf_clean_matrix"),  # Si no existe, devuelve None
                    temperatura=locals().get("temperatura"),
                    matriz_para_plot_muro=locals().get("matriz_para_plot_muro"),
                )

        else:
            # Obtengo los valores del campo eléctrico y la temperatura
            temperatura = Temperature.Temperature_Joule(
                voltage, current, T_0=params.init_temp, r_termica=sim_ctes.r_termica_no_percola
            )
            sistema_percola = False
            mean_field = np.mean(E_field_vector).item()
            # Si no ha percolado uso la corriente de Poole-Frenkel
            current = CurrentSolver.Poole_Frenkel(
                temperatura,
                mean_field,
                pb_metal_insul=sim_ctes.pb_metal_insul_set,
                permitividad_relativa=sim_ctes.permitividad_relativa_set,
                I_0=sim_ctes.I_0_set,
            ) * (params.device_size_y)

        if total_vacantes < max_vancantes_sp_set:
            # Actualizo el estado del sistema
            actual_state, _ = update_state_generation(
                actual_state,
                params,
                sim_ctes,
                E_field_vector,
                temperatura,
                sim_ctes.factor_vecinos_sp_set,
                sim_ctes.factor_libre_sp_set,
                ocupacion_max_sp_set,
            )

        elif not total_vacantes_sp_set:
            logger.info(f"Se ha alcanzado la ocupación máxima del {ocupacion_max_sp_set * 100}% en la primera segunda del set en el paso {k}.")
            total_vacantes_sp_set = True

        # Guardo los datos de la simulación
        data_sp_set[k] = np.array([simulation_time + tiempo_pp_set, voltage, current])

    tiempo_sp_set = simulation_time + tiempo_pp_set

    # Guardo los datos de la simulación
    utils.guardar_datos(
        save_path_data=rutas["simulation_path"] / f"Data_sp_set_{num_simulation}",
        headers={"datos_simulacion": "Tiempo [s],Voltaje [V],Intensidad [A]"},
        datos_sim=data_sp_set,
    )

    # Guardo todas las variables del estado final del PP set para usarlas en el PS set
    final_state_sp_set = {
        "actual_state": actual_state,
        "sim_ctes": sim_ctes,
        "params": params,
        "Temperatura_final": temperatura,
        "percola": sistema_percola,
        "centros_calculados": CF_centros,
        "tiempo_sp_set": tiempo_sp_set,
    }

    np.save(rutas["simulation_path"] / f"Final_state_pp_set_{num_simulation}.npz", actual_state)

    logger.info('Simulación del set finalizada correctamente.')

    return final_state_sp_set
