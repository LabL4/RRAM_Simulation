"""Fases del ciclo RESET de la simulación RRAM (PP_reset y SP_reset)."""

from typing import List

import numpy as np

from . import (
    CurrentSolver,
    ElectricField,
    Percolation,
    Temperature,
    exceptions,
    utils,
)
from .filament_tracking import procesar_filamentos_destruidos
from .state_updates import update_state_recombinate


def PP_reset(
    final_state_sp_set: dict,
    num_simulation: int,
    CF_ranges: List[tuple],
    num_pasos_guardar_estado: int = 100,  # Antes era cada 2000
):
    """
    Simulates the reset process of a resistive switching device, updating the system's state and tracking the evolution of various parameters over time.
    Args:
         - final_state_sp_set (dict): A dictionary containing the final state of the set process, including parameters, constants, temperature, time, and other simulation data.
            Expected keys:
                - "params": Simulation parameters.
                - "sim_ctes": Simulation constants.
                - "Temperatura_final": Final temperature from the set process.
                - "tiempo_sp_set": Time elapsed during the set process.
                - "percola": Boolean indicating if percolation occurred.
                - "actual_state": Current state of the system.

         - num_simulation (int): The simulation number, used for file naming and tracking.
         - CF_ranges (List[tuple]): A list of tuples defining the ranges for conductive filaments (CFs).
    Returns:
        None. The function performs the simulation, updates the system's state, and saves
        intermediate results (e.g., figures, data files) to disk.
    Notes:
        - During the reset process, no vacancies are generated; only recombination occurs.
        - The simulation involves calculating electric fields, currents, and temperature changes based on the system's state and applied voltage.
        - The state of the system is updated iteratively, and intermediate results are saved every 3000 steps.
        - The function handles both Ohmic and Poole-Frenkel conduction mechanisms depending on whether percolation occurs.
        - If a ZeroDivisionError occurs during current calculation, a custom exception is raised to handle null resistance scenarios.
    """

    params = final_state_sp_set["params"]
    sim_ctes = final_state_sp_set["sim_ctes"]
    temperatura = final_state_sp_set["Temperatura_final"]
    tiempo_sp_set = final_state_sp_set["tiempo_sp_set"]
    actual_state = final_state_sp_set["actual_state"]
    CF_centros = final_state_sp_set["centros_calculados"]

    # Parametros termicos
    temperatura_anterior = final_state_sp_set["Temperatura_final"][:, 1:-1]
    pendiente_temperatura = sim_ctes.pendiente_temperatura

    filas_intermedias, dist_casillas = Temperature.calcular_filas_intermedias(CF_centros)

    rutas = utils.crear_rutas_simulacion(num_simulation=num_simulation, state="reset")

    rutas["data_simulation_path"].mkdir(parents=True, exist_ok=True)

    oxygen_state = np.zeros_like(actual_state, dtype=np.int8)

    num_columnas = 3  # Tiempo, Voltaje, Intensidad
    data_pp_reset = np.zeros((params.num_pasos + 1, num_columnas), dtype=np.float64)

    # CUIDADO Configuración de umbrales, tiene q estar ordenado de mayor a menor!!
    oxygen_config = {
        float(sim_ctes.voltaje_gen_oxigeno_pp_2): int(sim_ctes.num_oxigenos_pp_reset_2),
        float(sim_ctes.voltaje_gen_oxigeno_pp_1): int(sim_ctes.num_oxigenos_pp_reset_1),
    }

    velocity_thresholds = {
        float(sim_ctes.voltaje_gen_oxigeno_pp_2): 5.2e-07,
        float(sim_ctes.voltaje_gen_oxigeno_pp_1): 3e-07,
    }

    print("La configuración de generación de oxígeno en la pp reset es:")
    for key, value in oxygen_config.items():
        print(f"  - {key} V: {value} oxígenos")

        print("La configuración de velocidades de oxígeno en la pp reset es:")
    for key, value in velocity_thresholds.items():
        print(f"  - {key} V: {value} m/s")

    CF_destruido = np.full(len(CF_ranges), False, dtype=bool)
    all_df_destruidos = False
    voltage_CF_destruido = np.full(len(CF_ranges), 0.0)

    E_field_vector = np.zeros((actual_state.shape[0]), dtype=np.float64)
    vector_ddp = np.arange(
        -0,
        -(params.voltaje_final_reset + params.paso_potencial_reset),
        -params.paso_potencial_reset,
    )
    print("El paso de potencial para la parte de set es:", params.paso_potencial_reset, "\n")

    CF_destruido_index = 1
    roturas_dict = {}

    print(f"Simulacion {num_simulation} - primera parte del reset")

    # Ciclo para la primera parte del reset
    for k in range(0, params.num_pasos + 1):
        simulation_time = params.paso_temporal * k
        voltage = vector_ddp[k]

        # Obtengo los valores del campo eléctrico
        E_field = abs(ElectricField.SimpleElectricField(voltage, params.device_size_x))

        # Genero el vector campo eléctrico
        for i in range(0, actual_state.shape[0]):
            E_field_vector[i] = abs(
                ElectricField.GapElectricField(
                    voltage, i, actual_state, device_size_x=params.device_size_x, grid_size=params.atom_size
                )
            )

        _, CF_graph = CurrentSolver.Clean_state_matrix(actual_state)

        max_x, max_y = actual_state.shape
        filamentos = CurrentSolver.Clasificar_CF(CF_graph, actual_state, CF_ranges)
        exist_cf = CurrentSolver.Existe_filamentos(filamentos, len(CF_ranges))

        if any(~CF_destruido):  # mientras haya alguno sin romper
            procesar_filamentos_destruidos(
                imagen_path=rutas["figures_path"],
                data_save_path=rutas["data_simulation_path"],
                existentes=exist_cf,
                CF_destruido=CF_destruido,
                voltage=voltage,
                voltage_CF_destruido=voltage_CF_destruido,
                actual_state=actual_state,
                num_simulation=num_simulation,
                roturas_dict=roturas_dict,
                etapa="pp",
            )

        # Obtengo la corrriente, antes decido cual usar comprobando si ha percolado o no
        if Percolation.is_path(actual_state):
            # Obtengo los caminos de percolación
            cf_clean_matrix = CurrentSolver.Eliminar_filamentos_incompletos(CF_graph, CF_ranges, exist_cf, actual_state)
            percola = True

            # Si ha percolado uso la corriente de Ohm
            try:
                current, _ = CurrentSolver.OmhCurrent(
                    voltage, cf_clean_matrix, ohm_resistence=sim_ctes.ohm_resistence_reset
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

            # Cáculo de las fuentes de calor (el filamento)
            Q_source_map = Temperature.calculate_heat_source(
                types_map=materials_map,
                atom_size=params.atom_size,
                I_total=current,
                R_cell=sim_ctes.ohm_resistence_reset,
                factor_generar_calor=sim_ctes.factor_generar_calor,
            )

            if not np.any(CF_destruido):
                # Todos los filamentos intactos: calculamos con muro térmico
                # Extracción del perfil térmico de los filamentos del paso anterior
                if isinstance(temperatura_anterior, (float, int)):
                    raise ValueError(
                        "La temperatura no se ha calculado como matriz, no se pueden extraer los perfiles de los filamentos. Se esperaba una matriz de temperaturas, pero se ha recibido un valor escalar."
                    )
                else:
                    mis_perfiles_extraidos = Temperature.extraer_perfiles_filamentos(
                        matriz_temperaturas=temperatura_anterior, filas_centros=CF_centros
                    )

                # Cálculo de los perfiles para los muros y colocación
                perfiles_muros_calculados = Temperature.calcular_perfiles_muro(
                    perfiles_filamentos=mis_perfiles_extraidos,
                    distancias_casillas=dist_casillas,
                    pendiente_temperatura=pendiente_temperatura,
                    atom_size=params.atom_size,
                    T_ambient=params.init_temp,
                )

                matriz_temperaturas_fijas = Temperature.colocar_muro_termico(
                    matriz_molde=cf_clean_matrix,
                    filas_intermedias=filas_intermedias,
                    perfiles_muros_calculados=perfiles_muros_calculados,
                )

                # Añadimos columnas de ceros (donde no hay muro) en las posiciones de los electrodos
                Ny = matriz_temperaturas_fijas.shape[0]
                columna_ceros = np.zeros((Ny, 1))
                matriz_temperaturas_fijas_final = np.hstack([columna_ceros, matriz_temperaturas_fijas, columna_ceros])

                # Obtengo la matriz de temperatura con el muro térmico
                temperatura = Temperature.solve_thermal_state(
                    types_map=materials_map,
                    Q_map=Q_source_map,
                    thermal_props=sim_ctes.propiedades_termicas,
                    atom_size=params.atom_size,
                    T_ambient=sim_ctes.Temperatura_electrodo,
                    matriz_muros=matriz_temperaturas_fijas_final,
                )

                # Actualizo la temperatura anterior para el siguiente paso
                temperatura_anterior = temperatura[:, 1:-1]

        else:
            percola = False

            # Calculo la temperatura cuando no hay percolación
            temperatura = Temperature.Temperature_Joule(
                voltage, current, T_0=params.init_temp, r_termica=sim_ctes.r_termica_no_percola
            )

            # Si no percola uso la corriente de Poole-Frenkel
            current = abs(
                CurrentSolver.Poole_Frenkel(
                    temperatura,
                    E_field,
                    pb_metal_insul=sim_ctes.pb_metal_insul_reset,
                    permitividad_relativa=sim_ctes.permitividad_relativa_reset,
                    I_0=sim_ctes.I_0_reset,
                )
                * (params.device_size_y)
            )

        # Actualizo el estado del sistema con la recombinación
        actual_state, oxygen_state = update_state_recombinate(
            voltage=voltage,
            E_field=E_field,
            oxygen_config=oxygen_config,
            sim_ctes=sim_ctes,
            params=params,
            actual_state=actual_state,
            oxygen_state=oxygen_state,
            temperatura=temperatura,
            velocity_thresholds=velocity_thresholds,
        )

        # Tiempo total de la simulacion
        tiempo_total = simulation_time + tiempo_sp_set
        data_pp_reset[k] = np.array([tiempo_total, voltage, current])

        # Represento el estado cada X pasos
        if k % num_pasos_guardar_estado == 0:
            matriz_para_plot_muro = np.copy(matriz_temperaturas_fijas)
            for centro, perfil_filamento in zip(CF_centros, mis_perfiles_extraidos):
                if centro is not None and perfil_filamento is not None:
                    matriz_para_plot_muro[centro, :] = perfil_filamento

            # Guardo las variables del estado
            utils.guardar_estado_intermedio(
                ruta_destino=rutas["data_simulation_path"],
                etapa="pp_reset",
                num_simulation=num_simulation,
                k=k,
                actual_state=actual_state,
                cf_clean_matrix=locals().get("cf_clean_matrix"),  # Si no existe, devuelve None
                temperatura=locals().get("temperatura"),
            )

    # Guardo los datos de la simulacion
    utils.guardar_datos(
        save_path_data=rutas["simulation_path"] / f"Data_pp_reset_{num_simulation}",
        headers={"datos_simulacion": "Tiempo [s],Voltaje [V],Intensidad [A]"},
        datos_sim=data_pp_reset,
    )
    print(f"La temperatura final alcanzada en el reset es: {temperatura} K\n")
    np.save(rutas["simulation_path"] / f"Final_state_pp_reset_{num_simulation}.npz", actual_state)

    if sum(CF_destruido) < len(CF_ranges):
        raise exceptions.FilamentosNoDestruidosException(
            simulation_path=rutas["simulation_path"],
            figures_path=rutas["figures_path"],
            voltage=voltage,
            num_simulation=num_simulation,
            actual_state=actual_state,
            CF_destruidos=sum(CF_destruido),
            CF_esperados=len(CF_ranges),
        )

    # Guardo todas las variables del estado final del PP set para usarlas en el PS set
    final_state_pp_reset = {
        "actual_state": actual_state,
        "oxygen_state": oxygen_state,
        "sistema_percola": percola,
        "sim_ctes": sim_ctes,
        "params": params,
        "Temperatura_final": temperatura,
        "voltaje_max_reset": voltage,
        "tiempo_pp_reset": simulation_time,
        "CF_destruido": CF_destruido,
        "voltage_CF_destruido": voltage_CF_destruido,
        "CF_destruido_index": CF_destruido_index,
        "roturas_dict": roturas_dict,
        "temperatura_final": temperatura,
        "centros_calculados": CF_centros,
    }

    return final_state_pp_reset


def SP_reset(
    final_state_pp_reset: dict,
    num_simulation: int,
    CF_ranges: List[tuple],
    num_pasos_guardar_estado: int = 250,
):
    params = final_state_pp_reset["params"]
    sim_ctes = final_state_pp_reset["sim_ctes"]
    temperatura = final_state_pp_reset["Temperatura_final"]
    tiempo_pp_reset = final_state_pp_reset["tiempo_pp_reset"]
    percola = final_state_pp_reset["sistema_percola"]
    actual_state = final_state_pp_reset["actual_state"]
    oxygen_state = final_state_pp_reset["oxygen_state"]
    CF_destruido_index = final_state_pp_reset["CF_destruido_index"]
    CF_destruido = final_state_pp_reset["CF_destruido"]
    voltage_CF_destruido = final_state_pp_reset["voltage_CF_destruido"]
    roturas_dict = final_state_pp_reset["roturas_dict"]
    centros_calculados = final_state_pp_reset["centros_calculados"]

    pendiente_temperatura = sim_ctes.pendiente_temperatura
    filas_intermedias, dist_casillas = Temperature.calcular_filas_intermedias(centros_calculados)

    # FIX: temperatura_anterior se usa dentro del bloque `if Percolation.is_path(...)` para
    # extraer perfiles térmicos de filamentos. En el original NO se inicializaba en SP_reset,
    # provocando NameError la primera vez que el sistema percolaba en esta fase.
    # Se hereda del estado final de PP_reset, recortando los electrodos (columnas 0 y -1).
    if isinstance(temperatura, np.ndarray) and temperatura.ndim == 2:
        temperatura_anterior = temperatura[:, 1:-1]
    else:
        # Si PP_reset terminó sin percolación, temperatura es escalar; el bloque que usa
        # temperatura_anterior detectará el caso con su propio isinstance(...) y elevará
        # ValueError explicativo en lugar de NameError.
        temperatura_anterior = temperatura

    print("Lol voltaje de rotura de pp reset son: ", voltage_CF_destruido)

    rutas = utils.crear_rutas_simulacion(num_simulation=num_simulation, state="reset")

    num_columnas = 3  # Tiempo, Voltaje, Intensidad
    data_sp_reset = np.zeros((params.num_pasos, num_columnas), dtype=np.float64)
    resistencia_vector = np.zeros((params.num_pasos, num_columnas), dtype=np.float64)

    # configuración de generación de oxígeno, si el voltaje supera el umbral se generan el número de oxígenos indicados, si hay varios umbrales se comprueba de mayor a menor y se asigna el número de oxígenos correspondiente al primer umbral que se supere
    oxygen_config = {float(sim_ctes.voltaje_gen_oxigeno_sp): int(sim_ctes.num_oxigenos_sp_reset)}
    velocity_thresholds = {float(sim_ctes.voltaje_gen_oxigeno_sp): 5.2e-07}

    print("La configuración de generación de oxígeno en la sp reset es:")
    for key, value in oxygen_config.items():
        print(f"  - {key} V: {value} oxígenos")

    print("La configuración de velocidades de oxígeno en la sp reset es:")
    for key, value in velocity_thresholds.items():
        print(f"  - {key} V: {value} m/s")

    print("Los filamentos destruidos al inicio del SP reset son: ", CF_destruido)

    E_field_vector = np.zeros((actual_state.shape[0]), dtype=np.float64)
    vector_ddp = np.arange(
        -params.voltaje_final_reset,
        0,
        params.paso_potencial_reset,
    )
    print("El paso de potencial para la parte de set es:", params.paso_potencial_reset, "\n")

    print(f"\nSimulacion {num_simulation} - segunda parte del reset\n")

    # Ciclo para la primera parte del reset
    for k in range(0, params.num_pasos):
        simulation_time = params.paso_temporal * k
        voltage = vector_ddp[k]

        # Obtengo los valores del campo eléctrico y la temperatura
        E_field = abs(ElectricField.SimpleElectricField(voltage, params.device_size_x))

        # Genero el vector campo eléctrico
        for i in range(0, actual_state.shape[0]):
            E_field_vector[i] = abs(
                ElectricField.GapElectricField(
                    voltage, i, actual_state, device_size_x=params.device_size_x, grid_size=params.atom_size
                )
            )

        actual_state_clean_CF, CF_graph = CurrentSolver.Clean_state_matrix(actual_state)

        max_x, max_y = actual_state.shape
        filamentos = CurrentSolver.Clasificar_CF(CF_graph, actual_state, CF_ranges)
        exist_cf = CurrentSolver.Existe_filamentos(filamentos, len(CF_ranges))

        anterior_voltage_CF = voltage_CF_destruido.copy()

        if any(~CF_destruido):  # mientras haya alguno sin romper
            procesar_filamentos_destruidos(
                imagen_path=rutas["figures_path"],
                data_save_path=rutas["data_simulation_path"],
                existentes=exist_cf,
                CF_destruido=CF_destruido,
                voltage=voltage,
                voltage_CF_destruido=voltage_CF_destruido,
                actual_state=actual_state,
                num_simulation=num_simulation,
                roturas_dict=roturas_dict,
                etapa="sp",
            )

        # Obtengo la corrriente, antes decido cual usar comprobando si ha percolado o no
        if Percolation.is_path(actual_state):
            # Obtengo los caminos de percolación
            cf_clean_matrix = CurrentSolver.Eliminar_filamentos_incompletos(CF_graph, CF_ranges, exist_cf, actual_state)
            percola = True

            # Si ha percolado uso la corriente de Ohm
            try:
                current, resistencia = CurrentSolver.OmhCurrent(
                    voltage, cf_clean_matrix, ohm_resistence=sim_ctes.ohm_resistence_reset
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

            # Cáculo de las fuentes de calor (el filamento)
            Q_source_map = Temperature.calculate_heat_source(
                types_map=materials_map,
                atom_size=params.atom_size,
                I_total=current,
                R_cell=sim_ctes.ohm_resistence_reset,
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
                    filas_centros=centros_calculados,
                )

            # =====================================================================
            # 5. CÁLCULO DE LOS PERFILES PARA LOS MUROS Y COLOCACIÓN
            # =====================================================================
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
            # Actualizo la temperatura anterior para el siguiente paso, NO guardo las columnas primera y ultima ya q corresponden a los electrodos
            temperatura_anterior = temperatura[:, 1:-1]

        else:
            percola = False

            # Si no ha percolado uso la corriente de Poole-Frenkel
            # Compruebo que sea un floar, si no lo es se lanza una excepción porque no se puede calcular la corriente
            if not isinstance(temperatura, float):
                raise ValueError(
                    "La temperatura calculada no es un valor escalar, no se puede calcular la corriente de Poole-Frenkel."
                )
            current = abs(
                CurrentSolver.Poole_Frenkel(
                    temperatura,
                    E_field,
                    pb_metal_insul=sim_ctes.pb_metal_insul_reset,
                    permitividad_relativa=sim_ctes.permitividad_relativa_reset,
                    I_0=sim_ctes.I_0_reset,
                )
                * (params.device_size_y)
            )

            temperatura = Temperature.Temperature_Joule(
                voltage, current, params.init_temp, r_termica=sim_ctes.r_termica_no_percola
            )

        # Actualizo el estado del sistema con la recombinación
        actual_state, oxygen_state = update_state_recombinate(
            voltage=voltage,
            E_field=E_field,
            oxygen_config=oxygen_config,
            sim_ctes=sim_ctes,
            params=params,
            actual_state=actual_state,
            oxygen_state=oxygen_state,
            temperatura=temperatura,
            velocity_thresholds=velocity_thresholds,
        )

        # Tiempo total de la simulacion
        tiempo_total = simulation_time + tiempo_pp_reset
        data_sp_reset[k] = np.array([tiempo_total, voltage, current])

        if locals().get("resistencia") is not None:
            resistencia_vector[k] = np.array([k, voltage, resistencia])
        else:
            resistencia_vector[k] = np.array([k, voltage, 0])

        # Represento el estado cada x pasos
        if k % num_pasos_guardar_estado == 0:
            # Si se ha creado la matriz de temperaturas fijas es porque se ha creado el muro y entonces tiene sentido guardarlo.
            if locals().get("matriz_temperaturas_fijas") is not None:
                matriz_para_plot_muro = np.copy(matriz_temperaturas_fijas)
                for centro, perfil_filamento in zip(centros_calculados, mis_perfiles_extraidos):  # type: ignore
                    if centro is not None and perfil_filamento is not None:
                        matriz_para_plot_muro[centro, :] = perfil_filamento

            utils.guardar_estado_intermedio(
                ruta_destino=rutas["data_simulation_path"],
                etapa="sp_reset",
                num_simulation=num_simulation,
                k=k,
                actual_state=actual_state,
                cf_clean_matrix=locals().get("cf_clean_matrix"),  # Si no existe, devuelve None
                temperatura=locals().get("temperatura"),
            )

    # Guardo los datos de la simulación
    save_path_data = rutas["simulation_path"] / f"Data_sp_reset_{num_simulation}.txt"
    data_encabezados = {"datos_simulacion": "Tiempo [s],Voltaje [V],Intensidad [A]"}

    # Guardo los datos de la simulacion
    utils.guardar_datos(
        save_path_data=save_path_data,
        headers=data_encabezados,
        datos_sim=data_sp_reset,
    )

    np.save(rutas["simulation_path"] / f"Final_state_sp_reset_{num_simulation}.npz", actual_state)

    print(f"\nSimulación {num_simulation} finalizada correctamente.\n")

    # Guardo todas las variables del estado final del PP set para usarlas en el PS set
    final_state_sp_reset = {
        "actual_state": actual_state,
        "oxygen_state": oxygen_state,
        "sistema_percola": percola,
        "sim_ctes": sim_ctes,
        "params": params,
        "Temperatura_final": temperatura,
        "tiempo_sp_reset": simulation_time,
        "voltage_CF_destruido": voltage_CF_destruido,
        "CF_destruido": CF_destruido,
        "roturas_dict": roturas_dict,
    }

    return final_state_sp_reset
