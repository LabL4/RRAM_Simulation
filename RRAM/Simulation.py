from ast import And
from . import (
    CurrentSolver,
    Representate,
    exceptions,
    Percolation,
    Temperature,
    ElectricField,
    Generation,
    Recombination,
    utils,
)

from dataclasses import dataclass, replace, asdict, field
from typing import get_type_hints
from dataclasses import fields
from typing import List, Dict
from functools import wraps
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import time
import csv
import re


def medir_tiempo(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        inicio = time.time()
        resultado = func(*args, **kwargs)
        fin = time.time()
        print(f"Función '{func.__name__}' ejecutada en {fin - inicio:.6f} segundos\n")
        return resultado

    return wrapper


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

    num_max_vacantes: int = field(init=False)
    paso_temporal: float = field(init=False)
    paso_potencial: float = field(init=False)

    def __post_init__(self):
        self.num_max_vacantes = int(0.9 * (self.device_size / self.atom_size) ** 2)
        self.paso_temporal = self.total_simulation_time / self.num_pasos
        self.paso_potencial = self.voltaje_final_reset / self.num_pasos

    def __repr__(self):
        # Crear lista de líneas con "nombre=valor" para cada atributo
        atributos = []
        # Usar vars(self) para incluir también campos calculados en __post_init__
        for nombre, valor in vars(self).items():
            atributos.append(f"   {nombre}={valor}")
        # Formatear en varias líneas
        return "Los parámetros de la simulación son:\n" + ",\n".join(atributos) + "\n"

    @staticmethod
    def from_dict(d: dict):
        # Usar get_type_hints para asegurar que obtienes tipos reales

        field_types = get_type_hints(SimulationParameters)
        init_fields = {f.name for f in fields(SimulationParameters) if f.init}

        mapping = {
            "voltaje_final_reset": "voltaje_final",
            "T_0": "init_temp",
            "initial_temperatura": "init_temp",
        }

        kwargs = {}
        for k in init_fields:
            src = mapping.get(k, k)

            # Verificar que la clave existe en el diccionario
            if src not in d:
                raise KeyError(f"La clave '{src}' no existe en el diccionario")

            # Debug: ver qué tipo y valor estamos procesando
            # print(f"Campo: {k}, Tipo: {field_types[k]}, Fuente: {src}, Valor: {d[src]}")

            try:
                kwargs[k] = field_types[k](d[src])
            except Exception as e:
                print(f"Error al convertir {k}: {e}")
                print(
                    f"Tipo esperado: {field_types[k]}, tipo real: {type(field_types[k])}"
                )
                raise

        return SimulationParameters(**kwargs)


@dataclass(frozen=True)
class SimulationConstants:
    vibration_frequency: float
    migration_energy: float
    drift_coefficient: float
    cte_red: float
    recom_enchancement_factor: float
    decaimiento_concentracion: float
    activation_energy: float
    gamma: float
    ohm_resistence: float
    pb_metal_insul: float
    pb_metal_insul_reset: float
    permitividad_relativa: float
    permitividad_relativa_reset: float
    I_0: float
    I_0_reset: float
    r_termica_percola: float
    r_termica_no_percola: float
    factor_generacion: float
    recombination_energy: float
    num_oxigenos_pp_reset_1: int
    num_oxigenos_pp_reset_2: int
    num_oxigenos_sp_reset: int

    @staticmethod
    def from_dict(d: dict):
        # Obtiene los tipos reales de SimulationConstants, no de SimulationParameters
        field_types = get_type_hints(SimulationConstants)

        # Alias si es necesario; si no, mapa vacío sirve
        mapping = {}

        kwargs = {}
        for k in field_types:
            src = mapping.get(k, k)

            if src not in d:
                raise KeyError(f"La clave '{src}' no existe en el diccionario")

            try:
                kwargs[k] = field_types[k](d[src])
            except Exception as e:
                print(f"Error al convertir {k} con valor {d[src]}: {e}")
                raise

        return SimulationConstants(**kwargs)

    def update_gamma(self, nuevo_valor_gamma: float):
        return replace(self, gamma=nuevo_valor_gamma)

    def update_ohm_resistence(self, nuevo_valor_ohm_resistence: float):
        return replace(self, ohm_resistence=nuevo_valor_ohm_resistence)

    def update_I_0(self, nuevo_I_0: float):
        return replace(self, I_0=nuevo_I_0)

    def update_pb_metal_insul(self, nuevo_pb_metal_insul: float):
        return replace(self, pb_metal_insul=nuevo_pb_metal_insul)

    def update_permitividad_relativa(self, permitividad_relativa_nuevo: float):
        return replace(self, permitividad_relativa=permitividad_relativa_nuevo)

    def __repr__(self):
        # Crear lista de líneas con "nombre=valor" para cada atributo
        atributos = []
        # Usar vars(self) para incluir también campos calculados en __post_init__
        for nombre, valor in vars(self).items():
            atributos.append(f"   {nombre}={valor}")
        # Formatear en varias líneas
        return "Las constantes de la simulación son:\n" + ",\n".join(atributos) + "\n"


def procesar_filamentos_creados(
    imagen_path,
    pkl_path,
    existentes,
    CF_creado,
    voltage,
    voltage_CF_creado,
    actual_state,
    num_simulation,
):
    """
    Detecta filamentos nuevos, actualiza su estado y guarda imágenes e
    imágenes correspondientes, además guarda el estado actual en PKL.

    Args:
        filamentos (list): Lista de filamentos en el estado actual.
        CF_creado (np.ndarray): Vector booleano que indica si cada filamento fue creado.
        voltage (float): Voltaje actual.
        voltage_CF_creado (np.ndarray): Array para registrar voltajes de creación.
        actual_state (np.ndarray): Estado actual del sistema.
        num_simulation (int): Número de simulación.

    Returns:
        int: Índice actualizado para el filamento.
    """

    filamentos_nuevos = [i for i, v in enumerate(existentes) if v and not CF_creado[i]]

    # imagen_path = (Path.cwd()
    # / "Results copy"
    # / f"simulation_{num_simulation}"
    # / "Figures")

    # pkl_path = Path.cwd() / "Results copy" / f"simulation_{num_simulation}" / "set"

    for i in filamentos_nuevos:
        CF_creado[i] = True
        voltage_CF_creado[i] = voltage
        print(
            f"El filamento {i + 1} se ha creado en el voltaje {round(voltage, 4)} (V)"
        )

        nombre_img = imagen_path / f"Filamento_{i + 1}_creado_set_{num_simulation}.png"

        Representate.RepresentateState(actual_state, round(voltage, 3), str(nombre_img))

        # Guardar estado actual en archivo pkl
        nombre_pkl = pkl_path / f"filamento_{i + 1}_creado_set_{num_simulation}.pkl"

        with open(nombre_pkl, "wb") as f:
            pickle.dump(actual_state, f)

    return None


def procesar_filamentos_destruidos(
    imagen_path,
    pkl_path,
    existentes,
    CF_destruido,
    voltage,
    voltage_CF_destruido,
    actual_state,
    num_simulation,
    roturas_dict,
    etapa,
):
    """
    Detecta filamentos rotos, actualiza su estado y guarda imágenes e
    imágenes correspondientes, además guarda el estado actual en PKL.

    Args:
        existentes (list[bool]): Lista booleana indicando existencia actual de filamentos.
        CF_destruido (np.ndarray): Vector booleano que indica si cada filamento fue destruido.
        voltage (float): Voltaje actual.
        voltage_CF_destruido (np.ndarray): Array para registrar voltajes de destrucción.
        actual_state (np.ndarray): Estado actual del sistema.
        num_simulation (int): Número de simulación.
        imagen_path (Path): Ruta donde guardar imágenes.
        pkl_path (Path): Ruta donde guardar archivos PKL.

    Returns:
        None
    """
    filamentos_rotos = [
        i for i, v in enumerate(existentes) if not v and not CF_destruido[i]
    ]

    for i in filamentos_rotos:
        CF_destruido[i] = True
        if voltage_CF_destruido[i] == 0:
            voltage_CF_destruido[i] = voltage
            j = len(roturas_dict)  # obtiene el siguiente índice disponible
            roturas_dict[j] = {
                "filamento": i + 1,
                "voltaje": voltage,
                "etapa": etapa,
            }
            print(
                f"\nEl filamento {i + 1} se ha roto en el voltaje {round(voltage, 4)} (V)"
            )

        nombre_img = imagen_path / f"Filamento_{i + 1}_roto_reset_{num_simulation}.png"
        Representate.RepresentateState(actual_state, round(voltage, 3), str(nombre_img))

        nombre_pkl = pkl_path / f"filamento_{i + 1}_roto_reset_{num_simulation}.pkl"
        with open(nombre_pkl, "wb") as f:
            pickle.dump(actual_state, f)

    return None


def update_state_generate(
    state: np.ndarray,
    params: SimulationParameters,
    sim_ctes_cte: dict,
    E_field_vector: np.ndarray,
    temperatura: float,
    factor_vecinos: float,
    factor_sin_vecinos: float,
    neighbor_mode: str = "both",  # Opciones: 'horizontal', 'vertical', 'both'
) -> np.ndarray:
    """
    Updates the state matrix by generating new vacancies based on the electric field,
    temperature, and neighboring conditions.
    Args:
        state (np.ndarray): Current state matrix where 0 represents free positions
            and 1 represents occupied positions.
        params (SimulationParameters): Object containing simulation parameters such as
            `paso_temporal`, `x_size`, and `y_size`.
        sim_ctes_cte (dict): Dictionary containing simulation constants required for
            the generation process.
        E_field_vector (np.ndarray): Vector representing the electric field for each
            row of the state matrix.
        temperatura (float): Temperature value used in the generation probability calculation.
        factor_vecinos (float): Multiplicative factor to increase the probability of
            generation for free positions with neighbors.
        factor_sin_vecinos (float): Multiplicative factor to decrease the probability of
            generation for free positions without neighbors.
        neighbor_mode (str, optional): Mode to determine neighbor consideration.
            Options are "horizontal", "vertical", or "both". Defaults to "horizontal".
    Returns:
        np.ndarray: Updated state matrix with new vacancies generated based on the
        calculated probabilities.
    Raises:
        AssertionError: If the number of rows in `E_field_vector` does not match the
            number of rows in the `state` matrix.
    Notes:
        - The function calculates the probability of vacancy generation for each free
          position in the state matrix based on the electric field, temperature, and
          neighboring conditions.
        - Neighboring conditions are determined based on the `neighbor_mode` parameter.
        - The generation process is stochastic, using random numbers to determine
          whether a vacancy is generated at each position.
    """

    assert E_field_vector.shape[0] == state.shape[0], (
        "El vector de campo eléctrico debe tener igual número de filas que la matriz de estado"
    )

    act_state = state.copy()

    # Máscara posiciones libres (True donde actual_state == 0)
    free_mask = state == 0

    # Inicializamos la máscara de vecinos en Falso
    vecino_mask = np.zeros_like(state, dtype=bool)

    # --- Lógica Horizontal (Izquierda / Derecha) ---
    if neighbor_mode in ["horizontal", "both"]:
        # Vecinos a la izquierda: desplazamos matriz a la derecha (axis 1)
        left_neighbor = np.zeros_like(state, dtype=bool)
        left_neighbor[:, 1:] = state[:, :-1] == 1

        # Vecinos a la derecha: desplazamos matriz a la izquierda (axis 1)
        right_neighbor = np.zeros_like(state, dtype=bool)
        right_neighbor[:, :-1] = state[:, 1:] == 1

        # Acumulamos en la máscara total
        vecino_mask |= left_neighbor | right_neighbor

    # --- Lógica Vertical (Arriba / Abajo) ---
    if neighbor_mode in ["vertical", "both"]:
        # Vecino Arriba: desplazamos matriz hacia abajo (axis 0)
        up_neighbor = np.zeros_like(state, dtype=bool)
        up_neighbor[1:, :] = state[:-1, :] == 1

        # Vecino Abajo: desplazamos matriz hacia arriba (axis 0)
        down_neighbor = np.zeros_like(state, dtype=bool)
        down_neighbor[:-1, :] = state[1:, :] == 1

        # Acumulamos en la máscara total usando OR lógico
        vecino_mask |= up_neighbor | down_neighbor

    # Definir quiénes están libres CON vecinos y quiénes SIN vecinos
    free_with_vecino = free_mask & vecino_mask
    free_without_vecino = free_mask & (~vecino_mask)

    # --- Cálculo de probabilidades ---

    # Calcular las probabilidades por fila
    prob_generacion_fila = np.minimum(
        [
            Generation.Generate(
                params.paso_temporal,
                E_field_vector[i],
                temperatura,
                **sim_ctes_cte,
            )
            for i in range(params.x_size)
        ],
        1,
    )

    # Probabilidad base expandida al tamaño de la matriz
    prob_base_matrix = np.tile(prob_generacion_fila.reshape(-1, 1), (1, params.y_size))

    # Crear matriz de probabilidad final
    prob_final = np.zeros_like(prob_base_matrix)

    # Aplicar factores condicionales
    # 1. Si tiene vecinos (según el modo seleccionado), aumentamos prob
    prob_final[free_with_vecino] = prob_base_matrix[free_with_vecino] * factor_vecinos

    # 2. Si está aislado (según el modo seleccionado), reducimos prob
    prob_final[free_without_vecino] = (
        prob_base_matrix[free_without_vecino] * factor_sin_vecinos
    )

    # Generación estocástica
    aleatorios = np.random.rand(params.x_size, params.y_size)
    nueva_vacante = aleatorios < prob_final

    act_state[nueva_vacante] = 1

    return act_state


def update_state_recombinate(
    voltage: float,
    E_field: float,
    oxygen_config: dict,
    sim_ctes_dict: dict,
    params: SimulationParameters,
    actual_state: np.ndarray,
    oxygen_state: np.ndarray,
    temperatura: float,
) -> tuple[np.ndarray, np.ndarray]:  # type: ignore
    """
    Updates the state of the system by generating oxygen, moving oxygen atoms,
    and recombining states based on the provided parameters.
    Args:
        voltage (float): The applied voltage in the system.
        E_field (float): The electric field in the system.
        oxygen_config (dict): A dictionary mapping threshold voltages to the
            number of oxygen atoms to generate when the threshold is exceeded.
        sim_ctes_dict (dict): A dictionary containing simulation constants.
        params (SimulationParameters): An object containing simulation parameters
            such as time step and atom size.
        actual_state (np.ndarray): The current state of the system.
        oxygen_state (np.ndarray): The current state of oxygen atoms in the system.
        temperatura (float): The temperature of the system.
    Returns:
        tuple: A tuple containing:
            - actual_state_update (np.ndarray): The updated state of the system.
            - oxygen_state_update (np.ndarray): The updated state of oxygen atoms.
    """

    # Genera oxígenos según voltaje
    for threshold_voltage, num_oxigenos in oxygen_config.items():
        if abs(voltage) > threshold_voltage:
            # print("Se van a generar", num_oxigenos, "oxígenos por voltaje de", voltage)
            oxygen_state = Generation.generate_oxigen_old(oxygen_state, num_oxigenos)
            break  # Solo generar una vez por paso temporal

    # Muevo los oxígenos
    oxygen_state, velocidad = Recombination.update_oxygen_state_old(
        paso_temp=params.paso_temporal,
        oxygen_state=oxygen_state,
        temperature=temperatura,
        E_field=E_field,
        grid_size=params.atom_size,
        **sim_ctes_dict,
    )

    # Obtengo la nueva configuración
    actual_state_update, oxygen_state_update = Recombination.Recombine_opt(
        vacancy_state=actual_state,
        oxygen_state=oxygen_state,
        paso_temp=params.paso_temporal,
        velocidad=velocidad,
        temp=temperatura,
        **sim_ctes_dict,
    )

    return actual_state_update, oxygen_state_update


def PP_set(
    num_simulation: int,
    params: SimulationParameters,
    sim_ctes: SimulationConstants,
    CF_ranges: List[tuple],
    CF_creado: np.ndarray,
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
    Raises:
        exceptions.MaxVacantesException: Raised if the maximum number of vacancies is exceeded.
        exceptions.NoPercolationException: Raised if the system does not percolate.
        exceptions.HighPercolationVoltageException: Raised if the percolation voltage is too high.
        exceptions.NullResistanceException: Raised if the resistance becomes null during the simulation.
    Notes:
        - The function creates directories for saving simulation results and figures.
        - It uses various helper functions and classes for tasks such as representing the state,
          solving currents, and calculating probabilities.
        - The simulation stops early if certain conditions are met, such as exceeding the maximum
          number of vacancies or reaching the final set voltage.
    Returns:
        None
    """

    # Declaro todas las variables que voy a usar exclusivamente en la primera parte (PP) del set.
    rutas = utils.crear_rutas_simulacion(num_simulation=num_simulation, state="set")

    rutas["simulation_path"].mkdir(parents=True, exist_ok=True)
    rutas["figures_path"].mkdir(parents=True, exist_ok=True)
    rutas["data_simulation_path"].mkdir(parents=True, exist_ok=True)

    # Cargo y represento el estado inicial de configuración
    actual_state = utils.cargar_y_representar_estado(
        Path.cwd() / f"Init_data/init_state_{num_simulation - 1}",
        rutas["figures_path"] / f"Initial_state_{num_simulation}.png",
        params.initial_voltaje,
    )

    sistema_percola = False
    total_vacantes_pp_set = False

    ocupacion_max_pp_set = 0.35  # 35% de ocupación máxima en PP set
    factor_vecinos = 1  # Factor de aumento de la probabilidad si tiene vecino
    factor_libre = 0.7  # Factor de disminución de la probabilidad si no tiene vecino
    lim_voltage_percolacion = 0.5  # Si el voltaje de percolación es mayor que este valor la simulación no vale la pena seguirla
    temperatura = params.T_0
    current = 0.0
    compliance_voltage = 1.2
    compliance = False
    cambiado_ohm_resistence = False

    max_vancantes_pp_set = int(ocupacion_max_pp_set * params.num_max_vacantes)
    voltage_CF_creado = np.full(len(CF_ranges), 0.0)

    # Inicializo vectores donde almaceno datos
    E_field_vector = np.zeros((actual_state.shape[0]), dtype=np.float64)
    vector_ddp = np.arange(
        0.000, params.voltaje_final_reset + params.paso_potencial, params.paso_potencial
    )

    num_columnas = 3  # Tiempo, Voltaje, Intensidad
    # Defino la matriz para almacenar los datos
    data_pp_set = np.zeros((params.num_pasos, num_columnas), dtype=np.float64)
    # config_matrix_pp_set = np.zeros((int((params.num_pasos / params.paso_guardar)), params.x_size, params.y_size))

    params_dict = asdict(params)
    sim_ctes_dict = asdict(sim_ctes)

    print("El valor de gamma es:", sim_ctes_dict["gamma"], "\n")

    indice_gamma = 1

    print(f"Simulacion {num_simulation} - Primera parte del set\n")

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
                voltage, i, actual_state, **params_dict
            )

        # Verifica si el sistema ha percolado
        if voltage >= params.voltaje_final_set:
            if not sistema_percola:
                raise exceptions.NoPercolationException()

            voltaje_max_set = vector_ddp[k]
            tiempo_pp_set = params.paso_temporal * (
                k - 1
            )  # Le quitamos un paso porque se ha superado el voltaje de ruptura

            print(
                "Voltaje final set",
                voltaje_max_set,
                "en el tiempo",
                tiempo_pp_set,
                "\n",
            )
            # Elimino las filas sobrantes del array de datos y las lleno de nans para eliminarlas luego
            data_pp_set[k - 1 :] = np.nan  # Añadir valores nulos a partir de la fila k
            data_pp_set = data_pp_set[
                ~np.isnan(data_pp_set).any(axis=1)
            ]  # Eliminar filas con valores nulos
            break

        # Obtengo la corrriente, antes decido cual usar comprobando si ha percolado o no
        if Percolation.is_path(actual_state):
            # Si es la primera vez que percola, siste_percola será falso y entra aquí
            if sistema_percola is False:
                voltaje_percolacion = voltage  # Guardo el voltaje de percolación
                print(
                    "\nEl sistema ha percolado en la iteración:",
                    k,
                    " que corresponde con el voltaje:",
                    round(voltaje_percolacion, 4),
                    " con una ocupación del:",
                    round((np.sum(actual_state) / (params.num_max_vacantes)), 4) * 100,
                    "%\n",
                )

                if voltaje_percolacion >= lim_voltage_percolacion:
                    # Si el voltaje de percolación es demasiado alto no va a coincidir con los datos experimentales, y no merece la pena seguir con la simulación
                    raise exceptions.HighPercolationVoltageException(
                        voltage_percola=voltaje_percolacion
                    )

                Representate.RepresentateState(
                    actual_state,
                    round(voltaje_percolacion, 3),
                    str(rutas["figures_path"]) + f"/Percola_state_{num_simulation}.png",
                )

                # nueva_gamma = sim_ctes.gamma - 1  # / sim_ctes.factor_generacion
                # sim_ctes = sim_ctes.update_gamma(nueva_gamma)
                # sim_ctes_dict = asdict(sim_ctes)

                # indice_gamma = indice_gamma + 1

                # print("El nuevo valor de gamma es:", sim_ctes_dict["gamma"], "\n")

            sistema_percola = True

            _, CF_graph = CurrentSolver.Clean_state_matrix(actual_state)

            filamentos = CurrentSolver.Clasificar_CF(
                CF_graph, params.x_size, params.y_size, CF_ranges
            )

            exist_cf = CurrentSolver.Existe_filamentos(filamentos, len(CF_ranges))

            # Compruebo si hay filamentos nuevos
            if any(~CF_creado):
                procesar_filamentos_creados(
                    imagen_path=rutas["figures_path"],
                    pkl_path=rutas["data_simulation_path"],
                    existentes=exist_cf,
                    CF_creado=CF_creado,
                    voltage=voltage,
                    voltage_CF_creado=voltage_CF_creado,
                    actual_state=actual_state,
                    num_simulation=num_simulation,
                )

            if sum(CF_creado) == indice_gamma:
                if len(CF_ranges) == 1:
                    print("Todos los filamentos creados.")
                    nueva_gamma = sim_ctes.gamma - 4
                if len(CF_ranges) == 2:
                    if sum(CF_creado) == 2 and not cambiado_ohm_resistence:
                        print("Todos los filamentos creados.")
                        actual_resistance = sim_ctes.ohm_resistence
                        sim_ctes = sim_ctes.update_ohm_resistence(
                            actual_resistance - 15
                        )
                        print("El nuevo valor de R_ohm es:", sim_ctes.ohm_resistence)
                        sim_ctes_dict = asdict(sim_ctes)
                        cambiado_ohm_resistence = True
                    else:
                        nueva_gamma = sim_ctes.gamma / 2
                        sim_ctes = sim_ctes.update_gamma(nueva_gamma)
                        sim_ctes_dict = asdict(sim_ctes)
                        indice_gamma = indice_gamma + 1
                        print("\n El nuevo valor de gamma es:", sim_ctes_dict["gamma"])
                else:
                    nueva_gamma = sim_ctes.gamma - 1

            cf_clean_matrix = CurrentSolver.Eliminar_filamentos_incompletos(
                CF_graph, CF_ranges, exist_cf
            )

            # Si ha percolado uso la corriente de Ohm
            try:
                if voltage <= compliance_voltage:
                    current, _ = CurrentSolver.OmhCurrent(
                        voltage, cf_clean_matrix, **sim_ctes_dict
                    )
            except ZeroDivisionError:
                raise exceptions.NullResistanceException(
                    simulation_path=rutas["simulation_path"],
                    figures_path=rutas["figures_path"],
                    voltage=voltage,
                    num_simulation=num_simulation,
                    actual_state=actual_state,
                )

            if voltage <= compliance_voltage and compliance:
                compliance = True
                print(
                    f"\nSe ha alcanzado el voltaje de compliance de {compliance_voltage} V en el paso {k} con una ocupación de {actual_state.sum}.\n"
                )

        else:
            sistema_percola = False
            mean_field = np.mean(E_field_vector).item()
            # Si no ha percolado uso la corriente de Poole-Frenkel
            current = CurrentSolver.Poole_Frenkel(
                temperatura, mean_field, **sim_ctes_dict
            ) * (params.device_size)

        # Obtengo los valores del campo eléctrico y la temperatura
        temperatura = Temperature.Temperature_Joule(
            voltage, current, sistema_percola, params.T_0, **sim_ctes_dict
        )

        if (total_vacantes < max_vancantes_pp_set) and (voltage <= compliance_voltage):
            # Actualizo el estado del sistema
            actual_state = update_state_generate(
                actual_state,
                params,
                sim_ctes_dict,
                E_field_vector,
                temperatura,
                factor_vecinos,
                factor_libre,
            )
        elif not total_vacantes_pp_set:
            print(
                f"\nSe ha alcanzado la ocupación máxima del {ocupacion_max_pp_set * 100}% en la primera parte del set en el paso {k}.\n"
            )
            total_vacantes_pp_set = True

        # Guardo los datos de la simulación
        data_pp_set[k] = np.array([simulation_time, voltage, current])

    # Guardo los datos de la simulación
    save_path_pkl = rutas["data_simulation_path"] / f"Data_pp_set_{num_simulation}.pkl"
    save_path_data = rutas["simulation_path"] / f"Data_pp_set_{num_simulation}.txt"
    save_path_figures = (
        rutas["figures_path"] / f"Final_state_pp_set_{num_simulation}.png"
    )

    utils.guardar_datos(
        voltaje_final=params.voltaje_final_set,
        config_state=actual_state,
        datos_save=data_pp_set,
        header_files="Tiempo simulacion [s],Voltaje [V],Intensidad [A]",
        save_path_data=save_path_data,
        save_path_pkl=save_path_pkl,
        save_path_figures=save_path_figures,
    )

    # Guardo todas las variables del estado final del PP set para usarlas en el PS set
    final_state_pp_set = {
        "actual_state": actual_state,
        "sistema_percola": sistema_percola,
        "k_maxima": k,
        "sim_ctes": sim_ctes,
        "params": params,
        "Temperatura_final": temperatura,
        "voltaje_max_set": voltaje_max_set,
        "voltaje_percolacion": voltaje_percolacion,
        "tiempo_pp_set": simulation_time,
        "current_final": current,
    }
    with open(
        rutas["simulation_path"] / f"final_state_pp_set_{num_simulation}.pkl", "wb"
    ) as f:
        pickle.dump(actual_state, f)

    Representate.RepresentateState(
        actual_state,
        round(voltage, 3),
        str(rutas["figures_path"]) + f"/final_state_pp_set_{num_simulation}.png",
    )

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
    print("El número inicial de vacantes es:", np.sum(actual_state))

    k_max = final_state_pp_set["k_maxima"] - 1
    sistema_percola = final_state_pp_set["sistema_percola"]
    sim_ctes = final_state_pp_set["sim_ctes"]
    params = final_state_pp_set["params"]
    voltaje_max_set = final_state_pp_set["voltaje_max_set"]
    tiempo_pp_set = final_state_pp_set["tiempo_pp_set"]
    current = final_state_pp_set["current_final"]

    temperatura = final_state_pp_set["Temperatura_final"]

    ocupacion_max_sp_set = 0.37
    max_vancantes_sp_set = int(ocupacion_max_sp_set * params.num_max_vacantes)
    factor_vecinos = 1.0  # Factor de aumento de la probabilidad si tiene vecino
    compliance_voltage = 1.2
    factor_libre = 0.9  # Factor de disminución de la probabilidad si no tiene vecino
    total_vacantes_sp_set = False
    num_columnas = 3  # Tiempo, Voltaje, Intensidad

    rutas = utils.crear_rutas_simulacion(num_simulation=num_simulation, state="set")

    vector_ddp = np.arange(voltaje_max_set, 0.000, -params.paso_potencial)
    E_field_vector = np.zeros((actual_state.shape[0]), dtype=np.float64)

    # Defino la matriz para almacenar los datos
    data_sp_set = np.zeros((k_max, num_columnas), dtype=np.float64)

    # nueva_gamma = sim_ctes.gamma / sim_ctes.factor_generacion
    # sim_ctes = sim_ctes.update_gamma(nueva_gamma)
    sim_ctes_dict = asdict(sim_ctes)

    print(f"Simulacion {num_simulation} - Segunda parte del set\n")
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
                voltage, i, actual_state, **asdict(params)
            )

        # Obtengo la corrriente, antes decido cual usar comprobando si ha percolado o no
        if Percolation.is_path(actual_state):
            # TODO: Si el sistema llega al maximo de vacante, como no genera mas, no hace falta recalcular los filamentos, ya que no van a cambiarif total_vacantes < max_vancantes_sp_set: estaba dando error lo he quitado

            _, CF_graph = CurrentSolver.Clean_state_matrix(actual_state)
            filamentos = CurrentSolver.Clasificar_CF(
                CF_graph, params.x_size, params.y_size, CF_ranges
            )
            exist_cf = CurrentSolver.Existe_filamentos(filamentos, len(CF_ranges))
            cf_clean_matrix = CurrentSolver.Eliminar_filamentos_incompletos(
                CF_graph, CF_ranges, exist_cf
            )
            # Si ha percolado uso la corriente de Ohm
            try:
                if voltage <= compliance_voltage:
                    current, _ = CurrentSolver.OmhCurrent(
                        voltage, cf_clean_matrix, **sim_ctes_dict
                    )
            except ZeroDivisionError:
                raise exceptions.NullResistanceException(
                    simulation_path=rutas["simulation_path"],
                    figures_path=rutas["figures_path"],
                    voltage=voltage,
                    num_simulation=num_simulation,
                    actual_state=actual_state,
                )

        else:
            sistema_percola = False

            mean_field = np.mean(E_field_vector).item()
            # Si no ha percolado uso la corriente de Poole-Frenkel
            current = CurrentSolver.Poole_Frenkel(
                temperatura, mean_field, **sim_ctes_dict
            ) * (params.device_size)

        # Obtengo los valores del campo eléctrico y la temperatura
        temperatura = Temperature.Temperature_Joule(
            voltage, current, sistema_percola, params.T_0, **sim_ctes_dict
        )
        if total_vacantes < max_vancantes_sp_set:
            # Actualizo el estado del sistema
            actual_state = update_state_generate(
                actual_state,
                params,
                sim_ctes_dict,
                E_field_vector,
                temperatura,
                factor_vecinos,
                factor_libre,
            )
        elif not total_vacantes_sp_set:
            print(
                f"Se ha alcanzado la ocupación máxima del {ocupacion_max_sp_set * 100}% en la primera segunda del set en el paso {k}.\n"
            )
            total_vacantes_sp_set = True

        # Guardo los datos de la simulación
        data_sp_set[k] = np.array([simulation_time + tiempo_pp_set, voltage, current])

    tiempo_sp_set = simulation_time + tiempo_pp_set

    # Guardo los datos de la simulación
    save_path_pkl = rutas["data_simulation_path"] / f"Data_sp_set_{num_simulation}.pkl"
    save_path_data = rutas["simulation_path"] / f"Data_sp_set_{num_simulation}.txt"
    save_path_figures = (
        rutas["figures_path"] / f"Final_state_sp_set_{num_simulation}.png"
    )

    utils.guardar_datos(
        voltaje_final=params.voltaje_final_set,
        config_state=actual_state,
        datos_save=data_sp_set,
        header_files="Tiempo simulacion [s],Voltaje [V],Intensidad [A]",
        save_path_data=save_path_data,
        save_path_pkl=save_path_pkl,
        save_path_figures=save_path_figures,
    )

    # Guardo todas las variables del estado final del PP set para usarlas en el PS set
    final_state_sp_set = {
        "actual_state": actual_state,
        "sim_ctes": sim_ctes,
        "params": params,
        "Temperatura_final": temperatura,
        "tiempo_sp_set": tiempo_sp_set,
        "percola": sistema_percola,
    }
    with open(
        rutas["simulation_path"] / f"final_state_sp_set_{num_simulation}.pkl", "wb"
    ) as f:
        pickle.dump(actual_state, f)

    Representate.RepresentateState(
        actual_state,
        round(voltage, 3),
        str(rutas["figures_path"]) + f"/final_state_sp_set_{num_simulation}.png",
    )

    print("\nSimulación del set finalizada correctamente.\n")

    return final_state_sp_set


def PP_reset(
    final_state_sp_set: dict,
    num_simulation: int,
    CF_ranges: List[tuple],
    num_pasos_guardar_estado: int = 3000,
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
    percola = final_state_sp_set[
        "percola"
    ]  # un poco inútil pero bueno, por si acaso (no ha tenido oportunidad de cambir el estado)
    actual_state = final_state_sp_set["actual_state"]

    rutas = utils.crear_rutas_simulacion(num_simulation=num_simulation, state="reset")

    rutas["data_simulation_path"].mkdir(parents=True, exist_ok=True)

    oxygen_state = np.zeros_like(actual_state, dtype=np.int8)

    num_columnas = 3  # Tiempo, Voltaje, Intensidad
    data_pp_reset = np.zeros((params.num_pasos + 1, num_columnas), dtype=np.float64)

    params_dict = asdict(params)
    sim_ctes_dict = asdict(sim_ctes)

    sim_ctes_dict["voltaje_generar_oxigeno_1"] = 0.6
    sim_ctes_dict["voltaje_generar_oxigeno_2"] = 0.8

    # CUIDADO Configuración de umbrales, tiene q estar ordenado de mayor a menor!!
    oxygen_config = {
        float(sim_ctes_dict["voltaje_generar_oxigeno_2"]): int(
            sim_ctes_dict["num_oxigenos_pp_reset_2"]
        ),
        float(sim_ctes_dict["voltaje_generar_oxigeno_1"]): int(
            sim_ctes_dict["num_oxigenos_pp_reset_1"]
        ),
    }

    print("La configuración de generación de oxígeno es:")
    for key, value in oxygen_config.items():
        print(f"  - {key} V: {value} oxígenos")

    CF_destruido = np.full(len(CF_ranges), False, dtype=bool)
    all_df_destruidos = False
    voltage_CF_destruido = np.full(len(CF_ranges), 0.0)

    E_field_vector = np.zeros((actual_state.shape[0]), dtype=np.float64)
    vector_ddp = np.arange(
        -0,
        -(params.voltaje_final_reset + params.paso_potencial),
        -params.paso_potencial,
    )

    CF_destruido_index = 1
    roturas_dict = dict()

    ohm_resistence_nuevo = 250
    sim_ctes = sim_ctes.update_ohm_resistence(ohm_resistence_nuevo)
    sim_ctes_dict = asdict(sim_ctes)
    all_df_destruidos = False
    print(
        "\n El nuevo valor de resistencia de cada celda es:",
        sim_ctes_dict["ohm_resistence"],
        "\n",
    )

    print(f"Simulacion {num_simulation} - primera parte del reset")

    # Ciclo para la primera parte del reset
    for k in range(0, params.num_pasos + 1):
        simulation_time = params.paso_temporal * k
        voltage = vector_ddp[k]

        # Obtengo los valores del campo eléctrico
        E_field = abs(ElectricField.SimpleElectricField(voltage, params.device_size))

        # Genero el vector campo eléctrico
        for i in range(0, actual_state.shape[0]):
            E_field_vector[i] = abs(
                ElectricField.GapElectricField(voltage, i, actual_state, **params_dict)
            )

        _, CF_graph = CurrentSolver.Clean_state_matrix(actual_state)

        max_x, max_y = actual_state.shape
        filamentos = CurrentSolver.Clasificar_CF(CF_graph, max_x, max_y, CF_ranges)
        exist_cf = CurrentSolver.Existe_filamentos(filamentos, len(CF_ranges))

        if any(~CF_destruido):  # mientras haya alguno sin romper
            procesar_filamentos_destruidos(
                imagen_path=rutas["figures_path"],
                pkl_path=rutas["data_simulation_path"],
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
            cf_clean_matrix = CurrentSolver.Eliminar_filamentos_incompletos(
                CF_graph, CF_ranges, exist_cf
            )
            percola = True

            # Si ha percolado uso la corriente de Ohm
            try:
                current, _ = CurrentSolver.OmhCurrent(
                    voltage, cf_clean_matrix, **sim_ctes_dict
                )
            except ZeroDivisionError:
                raise exceptions.NullResistanceException(
                    simulation_path=rutas["simulation_path"],
                    figures_path=rutas["figures_path"],
                    voltage=voltage,
                    num_simulation=num_simulation,
                    actual_state=actual_state,
                )

            # Calculo la temperatura cuando hay percolación
            temperatura = Temperature.Temperature_Joule(
                voltage, current, percola, params.T_0, **sim_ctes_dict
            )

        else:
            percola = False

            # Cambio el valor I_0 cuando el sistema ha roto todos los filamentos
            if np.all(CF_destruido) and not all_df_destruidos:
                I_0_nuevo = sim_ctes.I_0_reset
                permitividad_relativa_nuevo = sim_ctes.permitividad_relativa_reset
                pb_metal_insul_nuevo = sim_ctes.pb_metal_insul_reset
                print(
                    f"Se han destruido todos los filamentos en el paso",
                    k,
                    "con un voltaje de",
                    round(voltage, 4),
                    "los valores de la nueva corriente Poole-Frenkel son:",
                    f"\n I_0: {I_0_nuevo}, permitividad_relativa: {permitividad_relativa_nuevo}, pb_metal_insul: {pb_metal_insul_nuevo}\n",
                )
                sim_ctes = sim_ctes.update_I_0(I_0_nuevo)
                sim_ctes = sim_ctes.update_permitividad_relativa(
                    permitividad_relativa_nuevo
                )
                sim_ctes = sim_ctes.update_pb_metal_insul(pb_metal_insul_nuevo)

                sim_ctes_dict = asdict(sim_ctes)
                all_df_destruidos = True

            # Si no ha percolado uso la corriente de Poole-Frenkel
            current = abs(
                CurrentSolver.Poole_Frenkel(
                    temperatura,
                    float(np.mean(E_field_vector)),
                    **sim_ctes_dict,
                )
                * (params.device_size)
            )

            # Calculo la temperatura cuando no hay percolación

            temperatura = Temperature.Temperature_Joule(
                voltage, current, percola, params.T_0, **sim_ctes_dict
            )

        # Actualizo el estado del sistema con la recombinación
        actual_state, oxygen_state = update_state_recombinate(
            voltage=voltage,
            E_field=E_field,
            oxygen_config=oxygen_config,
            sim_ctes_dict=sim_ctes_dict,
            params=params,
            actual_state=actual_state,
            oxygen_state=oxygen_state,
            temperatura=temperatura,
        )

        # Tiempo total de la simulacion
        tiempo_total = simulation_time + tiempo_sp_set
        data_pp_reset[k] = np.array([tiempo_total, voltage, current])

        # # Represento el estado cada 3000 pasos
        # if k % num_pasos_guardar_estado == 0:
        #     fig_voltage = round(vector_ddp[k], 3)
        #     utils.guardar_representar_estado(
        #         voltaje=fig_voltage,
        #         config_state=actual_state,
        #         save_path_pkl=rutas["data_simulation_path"]
        #         / f"pp_reset_state_V={fig_voltage}_{num_simulation}.pkl",
        #         save_path_figures=rutas["figures_path"]
        #         / f"pp_reset_state_V={fig_voltage}_{num_simulation}.png",
        #     )

        #     Representate.RepresentateTwoStates(
        #         matriz1=actual_state,
        #         matriz2=oxygen_state,
        #         voltage=fig_voltage,
        #         filename=str(
        #             rutas["figures_path"]
        #             / f"pp_reset_full_state_V={fig_voltage}_{num_simulation}.png"
        #         ),
        #     )

        #     print(
        #         "Representando el estado de la simulación en el voltaje ",
        #         fig_voltage,
        #         " (V)",
        #     )

    # Guardo los datos de la simulación
    save_path_pkl = (
        rutas["data_simulation_path"] / f"Data_pp_reset_{num_simulation}.pkl"
    )
    save_path_data = rutas["simulation_path"] / f"Data_pp_reset_{num_simulation}.txt"
    save_path_figures = (
        rutas["figures_path"] / f"Final_state_pp_reset_{num_simulation}.png"
    )

    utils.guardar_datos(
        voltaje_final=voltage,
        config_state=actual_state,
        datos_save=data_pp_reset,
        header_files="Tiempo simulacion [s],Voltaje [V],Intensidad [A]",
        save_path_data=save_path_data,
        save_path_pkl=save_path_pkl,
        save_path_figures=save_path_figures,
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
    }
    with open(
        rutas["simulation_path"] / f"final_state_pp_reset_{num_simulation}.pkl", "wb"
    ) as f:
        pickle.dump(actual_state, f)

    Representate.RepresentateState(
        actual_state,
        round(voltage, 3),
        str(rutas["figures_path"]) + f"/final_state_pp_reset_{num_simulation}.png",
    )
    print("La temperatura final del pp reset es:", temperatura, "\n")
    print("La intensidad al final de la primera parte del reset es:", current, "\n")

    return final_state_pp_reset


def SP_reset(
    final_state_pp_reset: dict,
    num_simulation: int,
    CF_ranges: List[tuple],
    num_pasos_guardar_estado: int = 3000,
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

    print("Lol voltaje de rotura de pp reset son: ", voltage_CF_destruido)

    rutas = utils.crear_rutas_simulacion(num_simulation=num_simulation, state="reset")

    num_columnas = 3  # Tiempo, Voltaje, Intensidad
    data_sp_reset = np.zeros((params.num_pasos, num_columnas), dtype=np.float64)

    params_dict = asdict(params)
    sim_ctes_dict = asdict(sim_ctes)

    sim_ctes_dict["voltaje_generar_oxigeno"] = -0.2

    # Configuración de umbrales
    oxygen_config = {
        float(sim_ctes_dict["voltaje_generar_oxigeno"]): int(
            sim_ctes_dict["num_oxigenos_sp_reset"]
        )
    }

    print("Los filamentos destruidos al inicio del SP reset son: ", CF_destruido)

    E_field_vector = np.zeros((actual_state.shape[0]), dtype=np.float64)
    vector_ddp = np.arange(
        -params.voltaje_final_reset,
        0,
        params.paso_potencial,
    )

    print(f"\nSimulacion {num_simulation} - segunda parte del reset\n")

    # Ciclo para la primera parte del reset
    for k in range(0, params.num_pasos):
        simulation_time = params.paso_temporal * k
        voltage = vector_ddp[k]

        # Obtengo los valores del campo eléctrico y la temperatura
        E_field = abs(ElectricField.SimpleElectricField(voltage, params.device_size))

        # Genero el vector campo eléctrico
        for i in range(0, actual_state.shape[0]):
            E_field_vector[i] = abs(
                ElectricField.GapElectricField(voltage, i, actual_state, **params_dict)
            )

        _, CF_graph = CurrentSolver.Clean_state_matrix(actual_state)

        max_x, max_y = actual_state.shape
        filamentos = CurrentSolver.Clasificar_CF(CF_graph, max_x, max_y, CF_ranges)
        exist_cf = CurrentSolver.Existe_filamentos(filamentos, len(CF_ranges))

        anterior_voltage_CF = voltage_CF_destruido.copy()

        if any(~CF_destruido):  # mientras haya alguno sin romper
            procesar_filamentos_destruidos(
                imagen_path=rutas["figures_path"],
                pkl_path=rutas["data_simulation_path"],
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
            cf_clean_matrix = CurrentSolver.Eliminar_filamentos_incompletos(
                CF_graph, CF_ranges, exist_cf
            )
            percola = True

            # Si ha percolado uso la corriente de Ohm
            try:
                current, _ = CurrentSolver.OmhCurrent(
                    voltage, cf_clean_matrix, **sim_ctes_dict
                )
            except ZeroDivisionError:
                raise exceptions.NullResistanceException(
                    simulation_path=rutas["simulation_path"],
                    figures_path=rutas["figures_path"],
                    voltage=voltage,
                    num_simulation=num_simulation,
                    actual_state=actual_state,
                )
            print("Temperatura percolado antes: ", temperatura)
            temperatura = Temperature.Temperature_Joule(
                voltage, current, percola, params.T_0, **sim_ctes_dict
            )
            print("Temperatura percolado antes: ", temperatura)

        else:
            # Si no ha percolado uso la corriente de Poole-Frenkel
            current = abs(
                CurrentSolver.Poole_Frenkel(
                    temperatura,
                    ElectricField.SimpleElectricField(voltage, params.device_size),
                    # float(np.mean(E_field_vector)),
                    **sim_ctes_dict,
                )
                * (params.device_size)
            )

            percola = False
            temperatura = Temperature.Temperature_Joule(
                voltage, current, percola, params.T_0, **sim_ctes_dict
            )

        # Actualizo el estado del sistema con la recombinación
        actual_state, oxygen_state = update_state_recombinate(
            voltage=voltage,
            E_field=E_field,
            oxygen_config=oxygen_config,
            sim_ctes_dict=sim_ctes_dict,
            params=params,
            actual_state=actual_state,
            oxygen_state=oxygen_state,
            temperatura=temperatura,
        )

        # Tiempo total de la simulacion
        tiempo_total = simulation_time + tiempo_pp_reset
        data_sp_reset[k] = np.array([tiempo_total, voltage, current])

        # # Represento el estado cada 3000 pasos
        # if k % num_pasos_guardar_estado == 0:
        #     fig_voltage = round(vector_ddp[k], 3)
        #     utils.guardar_representar_estado(
        #         voltaje=fig_voltage,
        #         config_state=actual_state,
        #         save_path_pkl=rutas["data_simulation_path"]
        #         / f"sp_reset_state_V={fig_voltage}_{num_simulation}.pkl",
        #         save_path_figures=rutas["figures_path"]
        #         / f"sp_reset_state_V={fig_voltage}_{num_simulation}.png",
        #     )

        #     Representate.RepresentateTwoStates(
        #         matriz1=actual_state,
        #         matriz2=oxygen_state,
        #         voltage=fig_voltage,
        #         filename=str(
        #             rutas["figures_path"]
        #             / f"sp_reset_full_state_V={fig_voltage}_{num_simulation}.png"
        #         ),
        #     )

        #     print(
        #         "Representando el estado de la simulación en el voltaje: ",
        #         fig_voltage,
        #         " (V)",
        #     )

    # Guardo los datos de la simulación
    save_path_pkl = (
        rutas["data_simulation_path"] / f"Data_sp_reset_{num_simulation}.pkl"
    )
    save_path_data = rutas["simulation_path"] / f"Data_sp_reset_{num_simulation}.txt"
    save_path_figures = (
        rutas["figures_path"] / f"Final_state_sp_reset_{num_simulation}.png"
    )

    utils.guardar_datos(
        voltaje_final=voltage,
        config_state=actual_state,
        datos_save=data_sp_reset,
        header_files="Tiempo simulacion [s],Voltaje [V],Intensidad [A]",
        save_path_data=save_path_data,
        save_path_pkl=save_path_pkl,
        save_path_figures=save_path_figures,
    )

    with open(
        rutas["simulation_path"] / f"final_state_sp_reset_{num_simulation}.pkl", "wb"
    ) as f:
        pickle.dump(actual_state, f)

    Representate.RepresentateState(
        actual_state,
        round(voltage, 3),
        str(rutas["figures_path"]) + f"/final_state_sp_reset_{num_simulation}.png",
    )

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


def simulation_IV(
    num_simulation: int,
    figures_path: Path,
    simulation_path: Path,
    desplazamiento: dict,
    voltaje_percolacion: float,
    roturas_dict: dict,
):
    # region Representar datos
    save_path = figures_path / f"I-V_{num_simulation}"
    save_path_marcado = figures_path / f"I-V_{num_simulation}_marcado"

    # Definir nombres base y tipos
    prefixes = ["pp", "sp"]
    stages = ["set", "reset"]
    # Diccionario para guardar los datos cargados
    data = {}
    # Cargar archivos de forma automatizada
    for prefix in prefixes:
        for stage in stages:
            name = f"data_{prefix}_{stage}_{num_simulation}.npz"
            key = f"{prefix}_{stage}"
            data[key] = np.load(simulation_path / name)
    # Extraer y concatenar columnas de interés
    i_set = np.concatenate(
        [abs(data["pp_set"]["datos"][:, 2]), abs(data["sp_set"]["datos"][:, 2])]
    )
    v_set = np.concatenate(
        [data["pp_set"]["datos"][:, 1], data["sp_set"]["datos"][:, 1]]
    )
    v_reset = np.concatenate(
        [data["pp_reset"]["datos"][:, 1], data["sp_reset"]["datos"][:, 1]]
    )
    i_reset = np.concatenate(
        [abs(data["pp_reset"]["datos"][:, 2]), abs(data["sp_reset"]["datos"][:, 2])]
    )

    # # Diccionario de puntos que quieres ubicar
    # letras_ruptura = [
    #     "e",
    #     "f",
    #     "g",
    #     "h",
    #     "i",
    #     "j",
    #     "k",
    #     "l",
    #     "m",
    #     "n",
    #     "o",
    #     "p",
    #     "q",
    #     "r",
    #     "s",
    # ]

    # puntos_x_set = {"a": 1e-7, "b": voltaje_percolacion, "c": 1.1}
    # puntos_x_pp_reset = {"d": -0.42, "e": roturas_dict[0]["voltaje"], "f": -1.4}
    # puntos_x_sp_reset = {"g": -0.001}

    # contador_pp = sum(1 for v in roturas_dict.values() if v["etapa"] == "pp")

    # puntos_x_sp_reset = {}

    # Contador para seguir la última letra usada
    # ultima_letra_usada = 0

    # # Añado puntos de ruptura
    # for clave, datos in roturas_dict.items():
    #     voltaje = datos["voltaje"]
    #     etapa = datos["etapa"]
    #     if etapa == "pp" and ultima_letra_usada < len(letras_ruptura):
    #         puntos_x_pp_reset[letras_ruptura[ultima_letra_usada]] = voltaje
    #         # print("Los puntos de la pp reset son:", puntos_x_pp_reset, "\n")

    #     elif etapa == "sp" and ultima_letra_usada < len(letras_ruptura):
    #         puntos_x_sp_reset[letras_ruptura[ultima_letra_usada + contador_pp]] = (
    #             voltaje
    #         )
    #         # print("Los puntos de la sp reset son:", puntos_x_sp_reset)

    #     ultima_letra_usada += 1

    # # Añadir el punto final -1.4 usando la siguiente letra disponible
    # puntos_x_pp_reset[letras_ruptura[contador_pp]] = -1.4
    # puntos_x_sp_reset[letras_ruptura[contador_pp + ultima_letra_usada]] = -0.001

    # Obtener puntos en cada curva
    # puntos_set = utils.obtener_puntos_en_curva(
    #     data["pp_set"]["datos"][:, 1], abs(data["pp_set"]["datos"][:, 2]), puntos_x_set
    # )

    # puntos_x_pp_reset = utils.obtener_puntos_en_curva(
    #     data["pp_reset"]["datos"][:, 1],
    #     abs(data["pp_reset"]["datos"][:, 2]),
    #     puntos_x_pp_reset,
    # )

    # puntos_x_sp_reset = utils.obtener_puntos_en_curva(
    #     data["sp_reset"]["datos"][:, 1],
    #     abs(data["sp_reset"]["datos"][:, 2]),
    #     puntos_x_sp_reset,
    # )

    # print("Puntos en la curva I-V:\n")
    # for label, (v, i) in {
    #     **puntos_set,
    #     **puntos_x_pp_reset,
    #     **puntos_x_sp_reset,
    # }.items():
    #     print(f"  Punto {label}: V = {v:.6f} V, I = {i:.6e} A")

    # # Crear un único diccionario combinando ambos
    # puntos_totales = {}
    # puntos_totales.update(puntos_set)
    # puntos_totales.update(puntos_x_pp_reset)
    # puntos_totales.update(puntos_x_sp_reset)

    Representate.plot_IV(
        v_set,
        i_set,
        v_reset,
        i_reset,
        num_simulation - 1,
        titulo_figura="",
        figures_path=str(save_path),
    )
    # Representate.plot_IV_marcado(
    #     v_set,
    #     i_set,
    #     v_reset,
    #     i_reset,
    #     num_simulation - 1,
    #     puntos_totales,
    #     desplazamiento,
    #     titulo_figura="",
    #     figures_path=str(save_path_marcado),
    # )
    return None
