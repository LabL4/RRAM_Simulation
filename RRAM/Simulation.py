from matplotlib.pyplot import plot
from encodings.punycode import T
from scipy import constants
from ctypes import util
from hmac import new
import sys
from . import (
    CurrentSolver,
    ElectricField,
    Recombination,
    Representate,
    Percolation,
    Temperature,
    exceptions,
    Generation,
    utils,
)

from dataclasses import dataclass, replace, asdict, field
from typing import get_type_hints
from dataclasses import fields
from typing import List, Dict
from functools import wraps
from pathlib import Path
import numpy as np
import time


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
    device_size_x: float
    device_size_y: float
    atom_size: float
    num_trampas: int
    total_simulation_time: float
    num_pasos: int
    voltaje_final_reset: float
    voltaje_final_set: float
    init_temp: float

    x_size: int = field(init=False)
    y_size: int = field(init=False)
    num_max_vacantes: int = field(init=False)
    paso_temporal: float = field(init=False)
    paso_potencial_set: float = field(init=False)
    paso_potencial_reset: float = field(init=False)

    def __post_init__(self):
        self.x_size = int(np.ceil(self.device_size_x / self.atom_size))  # Número de "casillas" en la dimensión x
        self.y_size = int(np.ceil(self.device_size_y / self.atom_size))  # Número de "casillas" en la dimensión y
        self.num_max_vacantes = int(0.95 * (self.x_size * self.y_size))  # 95% de la matriz puede llenarse de vacantes
        self.paso_temporal = self.total_simulation_time / self.num_pasos  # Paso temporal en segundos
        self.paso_potencial_set = self.voltaje_final_set / self.num_pasos  # Paso de voltaje para la parte de set
        self.paso_potencial_reset = self.voltaje_final_reset / self.num_pasos  # Paso de voltaje para la parte de reset

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
        field_types = get_type_hints(SimulationParameters)
        init_fields = {f.name for f in fields(SimulationParameters) if f.init}
        kwargs = {}
        for k in init_fields:
            if k not in d:
                raise KeyError(f"La clave '{k}' no existe en el diccionario")
            kwargs[k] = field_types[k](d[k])
        return SimulationParameters(**kwargs)


@dataclass(frozen=True)
class SimulationConstants:
    vibration_frequency: float
    cte_red: float
    permitividad_relativa_set: float
    permitividad_relativa_reset: float
    generation_energy: float
    recombination_energy: float
    pb_metal_insul_set: float
    pb_metal_insul_reset: float
    recom_enchancement_factor: float
    long_decaimiento_concentracion: float
    ohm_resistence_set: float
    ohm_resistence_reset: float
    num_filamentos: int
    grosor_filamento: int
    gamma: float
    gamma_drift: float
    E_m: float
    I_0_set: float
    I_0_reset: float
    r_termica_no_percola: float
    conductividad_termica_dielectrico: float
    conductividad_termica_CF: float
    conductividad_termica_aislante: float
    conductividad_termica_electrodo: float
    Temperatura_electrodo: float
    factor_generar_calor: float
    pendiente_temperatura: float
    ocupacion_max_pp_set: float
    ocupacion_max_sp_set: float
    factor_vecinos_pp_set: float
    factor_libre_pp_set: float
    factor_vecinos_sp_set: float
    factor_libre_sp_set: float
    lim_voltage_percolacion: float
    compliance_voltage: float
    voltaje_gen_oxigeno_pp_1: float
    num_oxigenos_pp_reset_1: int
    voltaje_gen_oxigeno_pp_2: float
    num_oxigenos_pp_reset_2: int
    voltaje_gen_oxigeno_sp: float
    num_oxigenos_sp_reset: int

    @staticmethod
    def from_dict(d: dict):
        field_types = get_type_hints(SimulationConstants)
        kwargs = {}
        for k in field_types:
            if k not in d:
                raise KeyError(f"La clave '{k}' no existe en el diccionario")
            kwargs[k] = field_types[k](d[k])
        return SimulationConstants(**kwargs)

    @property
    def propiedades_termicas(self) -> dict:
        return {
            0: {"k": self.conductividad_termica_dielectrico},
            1: {"k": self.conductividad_termica_CF},
            2: {"k": self.conductividad_termica_aislante},
            3: {"k": self.conductividad_termica_electrodo},
        }

    def update_gamma(self, nuevo_valor_gamma: float):
        # gamma no tiene distinción de set/reset en los atributos originales,
        # por lo que este se queda igual, pero asegurándonos de que la variable exista.
        return replace(self, gamma=nuevo_valor_gamma)

    def update_ohm_resistence(self, nuevo_valor: float, fase: str = "set"):
        """Actualiza la resistencia óhmica dependiendo de la fase ('set' o 'reset')."""
        if fase not in ["set", "reset"]:
            raise ValueError("El parámetro 'fase' debe ser 'set' o 'reset'.")

        # Construimos el nombre exacto del atributo: "ohm_resistence_set" o "ohm_resistence_reset"
        atributo = f"ohm_resistence_{fase}"

        # Usamos un diccionario para pasar el argumento dinámicamente a replace()
        return replace(self, **{atributo: nuevo_valor})

    def update_I_0(self, nuevo_I_0: float, fase: str = "set"):
        """Actualiza la corriente de referencia I_0 dependiendo de la fase ('set' o 'reset')."""
        if fase not in ["set", "reset"]:
            raise ValueError("El parámetro 'fase' debe ser 'set' o 'reset'.")

        atributo = f"I_0_{fase}"
        return replace(self, **{atributo: nuevo_I_0})

    def update_pb_metal_insul(self, nuevo_pb_metal_insul: float, fase: str = "set"):
        """Actualiza la barrera de potencial dependiendo de la fase ('set' o 'reset')."""
        if fase not in ["set", "reset"]:
            raise ValueError("El parámetro 'fase' debe ser 'set' o 'reset'.")

        atributo = f"pb_metal_insul_{fase}"
        return replace(self, **{atributo: nuevo_pb_metal_insul})

    def update_permitividad_relativa(self, permitividad_relativa_nuevo: float, fase: str = "set"):
        """Actualiza la permitividad relativa dependiendo de la fase ('set' o 'reset')."""
        if fase not in ["set", "reset"]:
            raise ValueError("El parámetro 'fase' debe ser 'set' o 'reset'.")
        atributo = f"permitividad_relativa_{fase}"
        return replace(self, **{atributo: permitividad_relativa_nuevo})

    def update_generation_energy(self, nueva_energia: float):
        """Actualiza la energía de generación."""
        return replace(self, generation_energy=nueva_energia)

    def update_recombination_energy(self, nueva_energia: float):
        """Actualiza la energía de generación."""
        return replace(self, recombination_energy=nueva_energia)

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
    data_save_path,
    existentes,
    CF_creado,
    voltage,
    voltage_CF_creado,
    actual_state,
    num_simulation,
    plot_filamento: bool = False,
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

    for i in filamentos_nuevos:
        CF_creado[i] = True
        voltage_CF_creado[i] = voltage
        print(f"El filamento {i + 1} se ha creado en el voltaje {round(voltage, 4)} (V)")

        if plot_filamento:
            nombre_img = imagen_path / f"Filamento_{i + 1}_creado_set_{num_simulation}.png"
            Representate.RepresentateState(actual_state, round(voltage, 5), str(nombre_img))

        # Guardar estado actual en archivo pkl
        data_name = data_save_path / f"filamento_{i + 1}_creado_set_{num_simulation}.npz"
        np.savez_compressed(data_name, actual_state=actual_state)

    return None


def procesar_filamentos_destruidos(
    imagen_path,
    data_save_path,
    existentes,
    CF_destruido,
    voltage,
    voltage_CF_destruido,
    actual_state,
    num_simulation,
    roturas_dict,
    etapa,
    plot_filamento: bool = False,
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
    filamentos_rotos = [i for i, v in enumerate(existentes) if not v and not CF_destruido[i]]

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
            print(f"\nEl filamento {i + 1} se ha roto en el voltaje {round(voltage, 4)} (V)")

        if plot_filamento:
            nombre_img = imagen_path / f"Filamento_{i + 1}_roto_reset_{num_simulation}.png"
            Representate.RepresentateState(actual_state, round(voltage, 5), str(nombre_img))

        data_name = data_save_path / f"filamento_{i + 1}_roto_reset_{num_simulation}.npz"
        np.savez_compressed(data_name, actual_state=actual_state)

    return None


def update_state_generation(
    state: np.ndarray,
    params: SimulationParameters,
    sim_ctes: SimulationConstants,
    E_field: np.ndarray | float,
    temperatura: np.ndarray | float,
    factor_vecinos: float,
    factor_sin_vecinos: float,
    max_vacantes_permitidas: int,  # <-- NUEVO PARÁMETRO
    neighbor_mode: str = "both",
    CF_clean_matrix: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Orquesta el proceso de generación de vacantes asegurando que no se supere
    el límite máximo establecido. Prioriza aquellas con mayor probabilidad.
    """
    act_state = state.copy()
    x_size, y_size = state.shape

    # 1. Ajuste del Campo Eléctrico
    if isinstance(E_field, np.ndarray) and E_field.ndim == 1:
        E_field_matrix = np.repeat(E_field[:, np.newaxis], y_size, axis=1)
    else:
        E_field_matrix = E_field

    # 2. Ajuste de la matriz de Temperatura
    if isinstance(temperatura, np.ndarray) and temperatura.shape != state.shape:
        temp_matrix = temperatura[:, 1:-1]
    else:
        temp_matrix = temperatura

    # # 3. Cálculo de la matriz de probabilidades
    # prob_final = Generation.get_generation_probabilities_matrix(
    #     state=state,
    #     paso_temporal=params.paso_temporal,
    #     Electric_field=E_field_matrix,
    #     temperatura=temp_matrix,
    #     factor_vecinos=factor_vecinos,
    #     factor_sin_vecinos=factor_sin_vecinos,
    #     vibration_frequency=sim_ctes.vibration_frequency,
    #     generation_energy=sim_ctes.generation_energy,
    #     cte_red=sim_ctes.cte_red,
    #     gamma=sim_ctes.gamma,
    #     neighbor_mode=neighbor_mode,
    # )

    # 3. Cálculo de la matriz de probabilidades
    prob_final = Generation.get_generation_probabilities_matrix_CF(
        state=state,
        paso_temporal=params.paso_temporal,
        Electric_field=E_field_matrix,
        temperatura=temp_matrix,
        factor_vecinos=factor_vecinos,
        factor_sin_vecinos=factor_sin_vecinos,
        vibration_frequency=sim_ctes.vibration_frequency,
        generation_energy=sim_ctes.generation_energy,
        cte_red=sim_ctes.cte_red,
        gamma=sim_ctes.gamma,
        neighbor_mode=neighbor_mode,
        cf_matrix=CF_clean_matrix,
    )

    # 4. Evaluación de la estocástica (cuáles "intentan" generarse)
    aleatorios = np.random.rand(x_size, y_size)
    nueva_vacante = aleatorios < prob_final

    # 5. LIMITACIÓN INTELIGENTE DE VACANTES
    vacantes_actuales = np.sum(act_state)
    num_nuevas = np.sum(nueva_vacante)
    vacantes_disponibles = int(max_vacantes_permitidas - vacantes_actuales)

    if vacantes_disponibles <= 0:
        # Ya estamos en el límite, no se permite generar ninguna más
        return act_state, prob_final

    if num_nuevas > vacantes_disponibles:
        # Se ha superado el límite. Toca priorizar.

        # Obtenemos las coordenadas (i,j) de las candidatas
        coords = np.argwhere(nueva_vacante)

        # Extraemos las probabilidades exactas que tenían estas candidatas
        probs_candidatas = prob_final[nueva_vacante]

        # Ordenamos los índices de las candidatas de MAYOR a MENOR probabilidad
        indices_ordenados = np.argsort(probs_candidatas)[::-1]

        # Nos quedamos estrictamente con las mejores según el cupo disponible
        mejores_indices = indices_ordenados[:vacantes_disponibles]
        mejores_coords = coords[mejores_indices]

        # Actualizamos la matriz solo con las ganadoras
        act_state[mejores_coords[:, 0], mejores_coords[:, 1]] = 1
    else:
        # Si no se ha superado el límite, se aprueban todas
        act_state[nueva_vacante] = 1

    return act_state, prob_final


def update_state_recombinate(
    voltage: float,
    E_field: float,
    oxygen_config: dict,
    sim_ctes: SimulationConstants,
    params: SimulationParameters,
    actual_state: np.ndarray,
    oxygen_state: np.ndarray,
    temperatura: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Orquesta el proceso completo de RESET en un paso de tiempo:
    1. Generación de iones oxígenos por voltaje.
    2. Movimiento de iones.
    3. Recombinación.
    """

    # 1. Se generan oxígenos según el voltaje
    for threshold_voltage, max_oxigenos in oxygen_config.items():
        if abs(voltage) > threshold_voltage:
            oxygen_state = Recombination.generate_oxygen(oxygen_state, max_oxigenos)
            break  # Solo usar el umbral más alto superado

    # 2. Movemos los iones
    oxygen_state, velocidad = Recombination.move_oxygen_ions(
        paso_temp=params.paso_temporal,
        oxygen_state=oxygen_state,
        temperature=temperatura,
        E_field=E_field,
        grid_size=params.atom_size,
        vibration_frequency=sim_ctes.vibration_frequency,
        gamma_drift=sim_ctes.gamma_drift,
        migration_energy=sim_ctes.E_m,
        cte_red=sim_ctes.cte_red,
        potencial=voltage,
    )

    # 3.Recombinación de iones con vacantes
    actual_state_update, oxygen_state_update = Recombination.Recombine_opt(
        vacancy_state=actual_state,
        oxygen_state=oxygen_state,
        paso_temp=params.paso_temporal,
        velocidad=velocidad,
        temperatura=temperatura,
        vibration_frequency=sim_ctes.vibration_frequency,
        recom_enchancement_factor=sim_ctes.recom_enchancement_factor,
        recombination_energy=sim_ctes.recombination_energy,
        long_decaimiento_concentracion=sim_ctes.long_decaimiento_concentracion,
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
    np.set_printoptions(threshold=sys.maxsize)
    # Declaro todas las variables que voy a usar exclusivamente en la primera parte (PP) del set.
    rutas = utils.crear_rutas_simulacion(num_simulation=num_simulation, state="set")

    rutas["simulation_path"].mkdir(parents=True, exist_ok=True)
    rutas["figures_path"].mkdir(parents=True, exist_ok=True)
    rutas["data_simulation_path"].mkdir(parents=True, exist_ok=True)

    # Cargo el estado inicial de configuración
    actual_state = utils.cargar_estado(Path.cwd() / f"Init_data/init_state_{num_simulation - 1}")

    sistema_percola = False
    total_vacantes_pp_set = False
    num_pasos_guardar_estado = 250
    cf_clean_matrix = None
    voltaje_percolacion = params.voltaje_final_set

    # AL inicio como la corriente es de tipo poole frenkel, la resitencia ohmica se considera nula
    resistencia = 0.0
    temperatura_anterior = params.init_temp
    pendiente_temperatura = sim_ctes.pendiente_temperatura

    print(
        "Los valores de factor vecinos y factor libre son:",
        sim_ctes.factor_vecinos_pp_set,
        "y",
        sim_ctes.factor_libre_pp_set,
        "\n",
    )

    max_vancantes_pp_set = int(sim_ctes.ocupacion_max_pp_set * params.num_max_vacantes)
    voltage_CF_creado = np.full(len(CF_ranges), 0.0)
    filamentos_previos = 0

    # Inicializo vectores donde almaceno datos y condiciones iniciales
    temperatura = params.init_temp
    current = 0.0
    E_field_vector = np.zeros((actual_state.shape[0]), dtype=np.float64)
    vector_ddp = np.arange(0.000, params.voltaje_final_reset + params.paso_potencial_set, params.paso_potencial_set)
    print("El paso de potencial para la parte de set es:", params.paso_potencial_set, "\n")

    num_columnas = 3  # Tiempo, Voltaje, Intensidad

    # Defino la matriz para almacenar los datos
    data_pp_set = np.zeros((params.num_pasos, num_columnas), dtype=np.float64)
    resistencia_vector = np.zeros((params.num_pasos, num_columnas), dtype=np.float64)
    num_vacantes_total = np.zeros((params.num_pasos, num_columnas), dtype=np.float64)

    # Inicializamos las matrices variables a None para que existan desde el principio
    cf_clean_matrix = None
    probabilidad_matrix = None
    matriz_para_plot_muro = None

    print("El valor de gamma es:", sim_ctes.gamma, "\n")

    print(f"Simulacion {num_simulation} - Primera parte del set\n")
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

            print(
                "Voltaje final set",
                voltaje_max_set,
                "en el tiempo",
                tiempo_pp_set,
                "\n",
            )
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
                print(
                    "\nEl sistema ha percolado en la iteración:",
                    k,
                    " que corresponde con el voltaje:",
                    round(voltaje_percolacion, 5),
                    " con una ocupación del:",
                    round((np.sum(actual_state) / (params.num_max_vacantes)), 4) * 100,
                    "que corresponde con un numero de vacantes de:",
                    int(np.sum(actual_state)),
                    "\n",
                )

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
                )

            filamentos_actuales = sum(CF_creado)

            # Solo entramos si el número de filamentos ha AUMENTADO en este paso
            if filamentos_actuales > filamentos_previos:
                # 1. Caso para un solo filamento
                if len(CF_ranges) == 1:
                    if filamentos_actuales == 1:
                        print("Todos los filamentos creados.")
                        sim_ctes = sim_ctes.update_gamma(sim_ctes.gamma / 5)
                        sim_ctes = sim_ctes.update_generation_energy(1.75)
                        print(f"El nuevo valor de gamma es: {sim_ctes.gamma}")
                        print("El nuevo valor de la energía de generación es:", sim_ctes.generation_energy, "\n")
                        all_CFs_created = True
                        print("Todos los filamentos esperados se han creado:", all_CFs_created, "\n")

                # 2. Caso para dos filamentos
                elif len(CF_ranges) == 2:
                    if filamentos_actuales == 1:
                        # Se acaba de formar el PRIMERO
                        print("Se ha formado el primer filamento de dos.")
                        sim_ctes = sim_ctes.update_gamma(sim_ctes.gamma / 4)
                        sim_ctes = sim_ctes.update_generation_energy(sim_ctes.generation_energy + 0.1)
                        print(f"El nuevo valor de gamma es: {sim_ctes.gamma}")
                        print("El nuevo valor de la energía de generación es:", sim_ctes.generation_energy, "\n")
                        # obtengo los centros de los CF
                        centros_calculados = Temperature.obtener_centro_CF(actual_state_clean_CF, cf_ranges=CF_ranges)
                        print("Los centros calculados de los filamentos son:", centros_calculados, "\n")

                    elif filamentos_actuales == 2:
                        # Se acaba de formar el SEGUNDO
                        print("Se ha formado el segundo filamento de dos.")
                        sim_ctes = sim_ctes.update_gamma(sim_ctes.gamma / 2)
                        sim_ctes = sim_ctes.update_generation_energy(sim_ctes.generation_energy + 0.25)
                        print(f"El nuevo valor de gamma es: {sim_ctes.gamma}")
                        print("El nuevo valor de la energía de generación es:", sim_ctes.generation_energy, "\n")
                        all_CFs_created = True

                        # obtengo los centros de los CF
                        centros_calculados = Temperature.obtener_centro_CF(actual_state_clean_CF, cf_ranges=CF_ranges)
                        filas_intermedias, dist_casillas = Temperature.calcular_filas_intermedias(centros_calculados)

                        print("Los centros calculados de los filamentos son:", centros_calculados, "\n")
                        print("Las filas intermedias calculadas son:", filas_intermedias, "\n")

                # 3. Caso general para 3 o más filamentos
                else:
                    nuevos_formados = filamentos_actuales - filamentos_previos
                    for _ in range(nuevos_formados):
                        sim_ctes = sim_ctes.update_gamma(sim_ctes.gamma - 1)
                    print(
                        f"\nNuevos filamentos detectados {filamentos_actuales}. El nuevo valor de gamma es: {sim_ctes.gamma} \n"
                    )

                    if filamentos_actuales == len(CF_ranges):
                        print("Todos los filamentos creados.")

                # Actualizamos el historial para que no vuelva a entrar en iteraciones futuras
                filamentos_previos = filamentos_actuales

            cf_clean_matrix = CurrentSolver.Eliminar_filamentos_incompletos(CF_graph, CF_ranges, exist_cf, actual_state)

            # Limito el grosor de los filamentos a un máximo de 3 celdas
            _, new_cf_clean_matrix = CurrentSolver.limitar_grosor_filamentos(
                actual_state,
                cf_clean_matrix,
                centros_calculados,
                sim_ctes.grosor_filamento,
                CF_ranges,
            )

            cf_clean_matrix = new_cf_clean_matrix.copy()

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
                        matriz_temperaturas=temperatura_anterior, filas_centros=centros_calculados
                    )
                # print("\n--- Resultados de la Extracción ---")
                # for i, (centro, perfil) in enumerate(zip(centros_calculados, mis_perfiles_extraidos)):
                #     if perfil is not None:
                #         print(f"Filamento {i + 1} (Fila {centro}): Extraído un perfil de {len(perfil)} columnas.")
                #         print(f"   -> Primeros 5 valores térmicos: {np.round(perfil[:40], 1)} K")
                #     else:
                #         print(f"Filamento {i + 1} (Fila {centro}): No se formó (None).")

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
            if all_CFs_created:
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
                )
            else:
                actual_state, probabilidad_matrix = update_state_generation(
                    actual_state,
                    params,
                    sim_ctes,
                    E_field_vector,
                    temperatura,
                    sim_ctes.factor_vecinos_pp_set,
                    sim_ctes.factor_libre_pp_set,
                    max_vancantes_pp_set,
                    CF_clean_matrix=cf_clean_matrix,
                )

        elif not total_vacantes_pp_set:
            print(
                f"\nSe ha alcanzado la ocupación máxima del {sim_ctes.ocupacion_max_pp_set * 100}% en la primera parte del set en el paso {k}.\n"
            )
            total_vacantes_pp_set = True

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
                for centro, perfil_filamento in zip(centros_calculados, mis_perfiles_extraidos):
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

    # Muestro el valor de temperatura más alto alcanzado en la simulación
    print(f"\nLa temperatura máxima alcanzada en la simulación ha sido de: {round(np.max(temperatura), 4)} K\n")

    # SI no se fomran los flamentos esperados se decarta la simulación
    if filamentos_actuales < len(CF_ranges):
        raise exceptions.FilamentosNoFormadosException(
            simulation_path=rutas["simulation_path"],
            num_simulation=num_simulation,
            actual_state=actual_state,
            CF_formados=filamentos_actuales,
            CF_esperados=len(CF_ranges),
        )

    # Guardo los datos de la simulación
    save_path_data = rutas["simulation_path"] / f"ALL_data_pp_set_{num_simulation}"

    data_encabezados = {
        "datos_simulacion": "Tiempo [s],Voltaje [V],Intensidad [A]",
        "vacantes": "paso, Voltaje [V], Total Vacantes",  # Correcto, porque solo hay 1 columna
        "resistencia": "paso, Voltaje [V], Resistencia [Ohm]",  # Correcto, porque solo hay 1 columna
    }

    # Guardo los datos de la simulacion
    utils.guardar_datos(
        save_path_data=save_path_data,
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
        "centros_calculados": centros_calculados,
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
    print("El número inicial de vacantes es:", np.sum(actual_state))

    k_max = final_state_pp_set["k_maxima"] - 1
    sistema_percola = final_state_pp_set["sistema_percola"]
    sim_ctes = final_state_pp_set["sim_ctes"]
    params = final_state_pp_set["params"]
    voltaje_max_set = final_state_pp_set["voltaje_max_set"]
    tiempo_pp_set = final_state_pp_set["tiempo_pp_set"]
    current = final_state_pp_set["current_final"]
    max_vancantes_pp_set = final_state_pp_set["ocupacion_percola"]
    current = final_state_pp_set["intensidad_final"]
    centros_calculados = final_state_pp_set["centros_calculados"]
    cf_clean_matrix = final_state_pp_set["cf_clean_matrix"]

    print("El numero de vacantes al inicio del SP set es:", np.sum(actual_state))
    print("El numero de vacantes al percolar fue :", max_vancantes_pp_set)

    ocupacion_max_sp_set = 0 + 0  # 0.35
    max_vancantes_sp_set = max_vancantes_pp_set + int(ocupacion_max_sp_set * params.num_max_vacantes)
    compliance_voltage = 2
    total_vacantes_sp_set = False
    num_columnas = 3  # Tiempo, Voltaje, Intensidad
    num_pasos_guardar_estado = 250
    rutas = utils.crear_rutas_simulacion(num_simulation=num_simulation, state="set")

    temperatura_anterior = final_state_pp_set["Temperatura_final"]

    # Elimino las columnas 0 y ultima de la matriz de temperatura porque corresponden a los electrodos
    temperatura_anterior = final_state_pp_set["Temperatura_final"][:, 1:-1]

    pendiente_temperatura = sim_ctes.pendiente_temperatura

    print("El paso de potencial para sp set es:", params.paso_potencial_set, "\n")
    vector_ddp = np.arange(voltaje_max_set, 0.000, -params.paso_potencial_set)

    E_field_vector = np.zeros((actual_state.shape[0]), dtype=np.float64)

    # Defino la matriz para almacenar los datos
    data_sp_set = np.zeros((k_max, num_columnas), dtype=np.float64)

    filas_intermedias, dist_casillas = Temperature.calcular_filas_intermedias(centros_calculados)

    print("Los centros calculados de los filamentos son:", centros_calculados)
    print("Las filas intermedias calculadas son:", filas_intermedias)
    print("La distancia entre casillas es:", dist_casillas, "\n")

    actual_state_clean_CF, CF_graph = CurrentSolver.Clean_state_matrix(actual_state)
    filamentos = CurrentSolver.Clasificar_CF(CF_graph, actual_state, CF_ranges)
    exist_cf = CurrentSolver.Existe_filamentos(filamentos, len(CF_ranges))

    # cf_clean_matrix = CurrentSolver.Eliminar_filamentos_incompletos(CF_graph, CF_ranges, exist_cf)
    # _, cf_clean_matrix = CurrentSolver.limitar_grosor_filamentos(
    #     actual_state,
    #     cf_clean_matrix,
    #     centros_calculados,
    #     sim_ctes.grosor_filamento,
    #     CF_ranges,
    # )

    print(f"Simulacion {num_simulation} - Segunda parte del set\n")
    for k in range(0, k_max):
        total_vacantes = np.sum(actual_state)
        fig_voltage = round(vector_ddp[k], 5)
        # print("Número de vacantes en el paso", k, ":", total_vacantes)

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

        # Obtengo la corrriente, antes decido cual usar comprobando si ha percolado o no
        # print(f"Comprobando si el sistema percola para calcular la corriente: {Percolation.is_path(actual_state)} \n ")
        if Percolation.is_path(actual_state):
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
                    matriz_temperaturas=temperatura_anterior, filas_centros=centros_calculados
                )

            # CÁLCULO DE LOS PERFILES PARA LOS MUROS Y COLOCACIÓN
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

            # Importante para que se actualice el perfil termico de los filamentos
            temperatura_anterior = temperatura[:, 1:-1]

            if k % num_pasos_guardar_estado == 0:
                if locals().get("matriz_temperaturas_fijas"):
                    matriz_para_plot_muro = np.copy(matriz_temperaturas_fijas)
                    for centro, perfil_filamento in zip(centros_calculados, mis_perfiles_extraidos):
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
            if voltage <= compliance_voltage:
                current = CurrentSolver.Poole_Frenkel(
                    temperatura,
                    mean_field,
                    pb_metal_insul=sim_ctes.pb_metal_insul_set,
                    permitividad_relativa=sim_ctes.permitividad_relativa_set,
                    I_0=sim_ctes.I_0_set,
                ) * (params.device_size)

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
            print(
                f"Se ha alcanzado la ocupación máxima del {ocupacion_max_sp_set * 100}% en la primera segunda del set en el paso {k}.\n"
            )
            total_vacantes_sp_set = True

        # Guardo los datos de la simulación
        data_sp_set[k] = np.array([simulation_time + tiempo_pp_set, voltage, current])

    tiempo_sp_set = simulation_time + tiempo_pp_set

    # Guardo los datos de la simulación
    save_path_data = rutas["simulation_path"] / f"Data_sp_set_{num_simulation}"
    data_encabezados = {"datos_simulacion": "Tiempo [s],Voltaje [V],Intensidad [A]"}

    # Guardo los datos de la simulacion
    utils.guardar_datos(
        save_path_data=save_path_data,
        headers=data_encabezados,
        datos_sim=data_sp_set,
    )

    # Guardo todas las variables del estado final del SP set para usarlas en el PP reset
    final_state_sp_set = {
        "actual_state": actual_state,
        "sim_ctes": sim_ctes,
        "params": params,
        "Temperatura_final": temperatura,
        "tiempo_sp_set": tiempo_sp_set,
        "percola": sistema_percola,
    }

    np.save(rutas["simulation_path"] / f"final_state_pp_set_{num_simulation}.npz", actual_state)

    print("\nSimulación del set finalizada correctamente.\n")

    return final_state_sp_set


def PP_reset(
    final_state_sp_set: dict,
    num_simulation: int,
    CF_ranges: List[tuple],
    num_pasos_guardar_estado: int = 250,  # Antes era cada 2000
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

    # CUIDADO Configuración de umbrales, tiene q estar ordenado de mayor a menor!!
    oxygen_config = {
        float(sim_ctes.voltaje_gen_oxigeno_pp_2): int(sim_ctes.num_oxigenos_pp_reset_2),
        float(sim_ctes.voltaje_gen_oxigeno_pp_1): int(sim_ctes.num_oxigenos_pp_reset_1),
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
        E_field = abs(ElectricField.SimpleElectricField(voltage, params.device_size))

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

            # Obtengo la matriz de temperatura
            temperatura = Temperature.solve_thermal_state(
                types_map=materials_map,
                Q_map=Q_source_map,
                thermal_props=sim_ctes.propiedades_termicas,
                atom_size=params.atom_size,
                T_ambient=sim_ctes.Temperatura_electrodo,
            )

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
                * (params.device_size)
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
        )

        # Tiempo total de la simulacion
        tiempo_total = simulation_time + tiempo_sp_set
        data_pp_reset[k] = np.array([tiempo_total, voltage, current])

        # Represento el estado cada 3000 pasos
        if k % num_pasos_guardar_estado == 0:
            # matriz_para_plot_muro = np.copy(matriz_temperaturas_fijas)
            # for centro, perfil_filamento in zip(centros_calculados, mis_perfiles_extraidos):
            # if centro is not None and perfil_filamento is not None:
            # matriz_para_plot_muro[centro, :] = perfil_filamento

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

    # Guardo los datos de la simulación
    save_path_data = rutas["simulation_path"] / f"Data_pp_reset_{num_simulation}"

    data_encabezados = {
        "datos_simulacion": "Tiempo [s],Voltaje [V],Intensidad [A]",
    }

    # Guardo los datos de la simulacion
    utils.guardar_datos(
        save_path_data=save_path_data,
        headers=data_encabezados,
        datos_sim=data_pp_reset,
    )

    np.save(rutas["simulation_path"] / f"final_state_pp_reset_{num_simulation}.npz", actual_state)

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

    print("Lol voltaje de rotura de pp reset son: ", voltage_CF_destruido)

    rutas = utils.crear_rutas_simulacion(num_simulation=num_simulation, state="reset")

    num_columnas = 3  # Tiempo, Voltaje, Intensidad
    data_sp_reset = np.zeros((params.num_pasos, num_columnas), dtype=np.float64)

    # configuración de generación de oxígeno, si el voltaje supera el umbral se generan el número de oxígenos indicados, si hay varios umbrales se comprueba de mayor a menor y se asigna el número de oxígenos correspondiente al primer umbral que se supere
    oxygen_config = {float(sim_ctes.voltaje_gen_oxigeno_sp): int(sim_ctes.num_oxigenos_sp_reset)}

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
        E_field = abs(ElectricField.SimpleElectricField(voltage, params.device_size))

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

            # Obtengo la matriz de temperatura
            temperatura = Temperature.solve_thermal_state(
                types_map=materials_map,
                Q_map=Q_source_map,
                thermal_props=sim_ctes.propiedades_termicas,
                atom_size=params.atom_size,
                T_ambient=sim_ctes.Temperatura_electrodo,
            )

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
                * (params.device_size)
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
        )

        # Tiempo total de la simulacion
        tiempo_total = simulation_time + tiempo_pp_reset
        data_sp_reset[k] = np.array([tiempo_total, voltage, current])

        # Represento el estado cada x pasos
        if k % num_pasos_guardar_estado == 0:
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
    save_path_pkl = rutas["data_simulation_path"] / f"Data_sp_reset_{num_simulation}.pkl"
    save_path_data = rutas["simulation_path"] / f"Data_sp_reset_{num_simulation}.txt"
    save_path_figures = rutas["figures_path"] / f"Final_state_sp_reset_{num_simulation}.png"

    utils.guardar_datos(
        voltaje_final=voltage,
        config_state=actual_state,
        datos_save=data_sp_reset,
        header_files="Tiempo simulacion [s],Voltaje [V],Intensidad [A]",
        save_path_data=save_path_data,
        save_path_pkl=save_path_pkl,
        save_path_figures=save_path_figures,
    )

    np.save(rutas["simulation_path"] / f"final_state_sp_reset_{num_simulation}.npz", actual_state)

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
    i_set = np.concatenate([abs(data["pp_set"]["datos"][:, 2]), abs(data["sp_set"]["datos"][:, 2])])
    v_set = np.concatenate([data["pp_set"]["datos"][:, 1], data["sp_set"]["datos"][:, 1]])
    v_reset = np.concatenate([data["pp_reset"]["datos"][:, 1], data["sp_reset"]["datos"][:, 1]])
    i_reset = np.concatenate([abs(data["pp_reset"]["datos"][:, 2]), abs(data["sp_reset"]["datos"][:, 2])])

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

    puntos_x_set = {"a": 1e-9, "b": voltaje_percolacion, "c": 1.1}
    puntos_x_pp_reset = {"d": -0.44, "e": roturas_dict[0]["voltaje"], "f": -1.1}
    puntos_x_sp_reset = {"g": -2e-8}

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
    puntos_set = utils.obtener_puntos_en_curva(
        data["pp_set"]["datos"][:, 1], abs(data["pp_set"]["datos"][:, 2]), puntos_x_set
    )

    puntos_x_pp_reset = utils.obtener_puntos_en_curva(
        data["pp_reset"]["datos"][:, 1],
        abs(data["pp_reset"]["datos"][:, 2]),
        puntos_x_pp_reset,
    )

    puntos_x_sp_reset = utils.obtener_puntos_en_curva(
        data["sp_reset"]["datos"][:, 1],
        abs(data["sp_reset"]["datos"][:, 2]),
        puntos_x_sp_reset,
    )

    print("Puntos en la curva I-V:\n")
    for label, (v, i) in {
        **puntos_set,
        **puntos_x_pp_reset,
        **puntos_x_sp_reset,
    }.items():
        print(f"  Punto {label}: V = {v:.6f} V, I = {i:.6e} A")

    # Crear un único diccionario combinando ambos
    puntos_totales = {}
    puntos_totales.update(puntos_set)
    puntos_totales.update(puntos_x_pp_reset)
    puntos_totales.update(puntos_x_sp_reset)

    Representate.plot_IV(
        v_set,
        i_set,
        v_reset,
        i_reset,
        num_simulation - 1,
        titulo_figura="",
        figures_path=str(save_path),
    )
    Representate.plot_IV_marcado(
        v_set,
        i_set,
        v_reset,
        i_reset,
        num_simulation - 1,
        puntos_totales,
        desplazamiento,
    )

    return None
