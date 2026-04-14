from RRAM import Representate, Temperature
from . import Representate, utils
from typing import List, Dict
from pathlib import Path

import numpy as np
import pickle
import csv


def generar_configuracion_filamentos(eje_x, eje_y, num_filamentos, peso_central=70):
    """
    Calcula los rangos de cada filamento y define las regiones de peso iniciales.

    Argumentos:
        eje_x (int): Número de celdas en el eje X (filas).
        eje_y (int): Número de celdas en el eje Y (columnas).
        num_filamentos (int): Número de filamentos a distribuir.
        peso_central (int): Probabilidad asignada a la zona central (por defecto 70).

    Retorna:
        tuple: (filamentos_ranges, regiones_pesos, filas_centrales)
    """
    filamentos_ranges = []
    regiones_pesos = []
    filas_centrales = []

    # Calculamos el ancho de cada zona dividiendo el espacio total entre el número de filamentos
    ancho_zona = eje_x // num_filamentos

    for n in range(num_filamentos):
        # 1. Calcular el rango (zona) de este filamento
        inicio_rango = n * ancho_zona
        # El último filamento se extiende hasta el final para cubrir el resto por división entera
        fin_rango = (n + 1) * ancho_zona - 1 if n < num_filamentos - 1 else eje_x - 1
        filamentos_ranges.append((inicio_rango, fin_rango))

        # 2. Identificar la fila central de dicha zona
        fila_central = (inicio_rango + fin_rango) // 2
        filas_centrales.append(fila_central)

        # 3. Definir la región de alta probabilidad (Fila central + 2 arriba + 2 abajo)
        # x_start: fila_central - 2
        # x_end: fila_central + 3 (se usa +3 porque en el slicing el límite superior no se incluye)
        x_start = max(0, fila_central - 2)
        x_end = min(eje_x, fila_central + 3)

        # La región cubre todo el ancho del dispositivo (de 0 a eje_y)
        region = (x_start, x_end, 0, eje_y)
        regiones_pesos.append((region, peso_central))

    return filamentos_ranges, regiones_pesos, filas_centrales


def obtener_puntos_en_curva(v_array, i_array, puntos_x):
    """
    Devuelve un diccionario con las coordenadas (x, y) de los puntos
    más cercanos en la curva definida por v_array e i_array.

    v_array: array de voltajes de la curva
    i_array: array de corrientes de la curva
    puntos_x: diccionario {'label': x}

    Retorna: diccionario {'label': (x, y)}
    """
    puntos = {}
    for label, xp in puntos_x.items():
        idx = np.argmin(np.abs(v_array - xp))  # índice del x más cercano
        yp = i_array[idx]  # valor de y correspondiente
        puntos[label] = (xp, yp)
    return puntos


def crear_rutas_simulacion(num_simulation: int, state: str) -> dict:
    """
    Crea las rutas necesarias para guardar resultados de la simulación.

    Args:
        num_simulation (int): Número índice de la simulación.
        state (str): Estado de la simulación, puede ser 'set' o 'reset'.

    Returns:
        dict: Diccionario con las rutas Path para 'simulation', 'figures' y 'set'.

    Side-effects:
        Crea las carpetas en disco si no existen.
    """
    simulation_path = Path.cwd() / "Results" / f"simulation_{num_simulation}"
    figures_path = simulation_path / "Figures"
    data_simulation_path = simulation_path / state

    return {
        "simulation_path": simulation_path,
        "figures_path": figures_path,
        "data_simulation_path": data_simulation_path,
    }


def cargar_y_representar_estado(
    pkl_path: Path, figures_path: Path, voltage: float, plot_state: bool = True, device_size: float = 10e-9
) -> np.ndarray:
    """
    Carga el estado de configuración desde archivo pkl y genera una imagen de ese estado.

    Args:
        pkl_path (Path): Ruta del pkl con el estado.
        figures_path (Path): Ruta donde se guardará la imagen generada, debe incluir el nombre del archivo CON extensión.
        plot_state (bool): Indica si se debe generar la imagen del estado, por defecto es True.

    Returns:
        np.ndarray: Matriz con el estado inicial cargado.
    """
    with open(f"{pkl_path}.pkl", "rb") as f:
        actual_state = pickle.load(f)

    if plot_state:
        Representate.RepresentateState(actual_state, voltage, str(figures_path), device_size=device_size)

    return actual_state


def guardar_datos(
    voltaje_final: float,
    config_state: np.ndarray,
    datos_save: np.ndarray,
    header_files: str,
    save_path_data: Path,
    save_path_pkl: Path,
    save_path_figures: Path,
) -> None:
    """
    Saves simulation data, configuration state, and generates a representation of the final state.
    Args:
        voltaje_final_set (float): The final voltage value used in the simulation.
        config_state (np.ndarray): The final configuration state of the simulation.
        datos_save (np.ndarray): The data to be saved as a CSV file.
        header_files (str): The header string for the CSV file.
        save_path_data (Path): The file path where the configuration state will be saved as a binary file.
        save_path_pkl (Path): The file path where the simulation data will be saved as a CSV file.
        save_path_figures (Path): The directory path where the final state representation will be saved.
    Returns:
        None
    """

    # Cuando acaba la simulacion guardo el estado final de configuracion
    with open(str(save_path_pkl), "wb") as f:
        pickle.dump(config_state, f)

    np.savez(
        save_path_data.with_suffix(".npz"),
        datos=datos_save,
        header=np.array([header_files]),
    )
    Representate.RepresentateState(config_state, voltaje_final, str(save_path_figures))

    return None


def guardar_representar_estado(
    voltaje: float,
    config_state: np.ndarray,
    save_path_pkl: Path,
    save_path_figures: Path,
    representar_estado: bool = False,
) -> None:
    """
    Saves simulation data, configuration state, and generates a representation of the final state.
    Args:
        voltaje_ (float): The voltage value in the state.
        config_state (np.ndarray): The final configuration state of the simulation.
        save_path_pkl (Path): The file path where the simulation data will be saved as a CSV file.
        save_path_figures (Path): The directory path where the final state representation will be saved.
    Returns:
        None
    """

    # Cuando acaba la simulacion guardo el estado final de configuracion
    with open(str(save_path_pkl), "wb") as f:
        pickle.dump(config_state, f)

    if representar_estado:
        Representate.RepresentateState(config_state, voltaje, str(save_path_figures))

    return None


def read_csv_to_dic(cvs_path: str) -> List[Dict[str, str]]:
    """
    Reads a CSV file and returns its contents as a list of dictionaries.
    Args:
        cvs_path (str): The name of the CSV file to read.

    Returns:
        list: A list of dictionaries, where each dictionary represents a row in the CSV file.
    """
    with open(cvs_path, mode="r") as archivo:
        reader = csv.DictReader(archivo)
        data = [row for row in reader]
    return data


def simulation_IV(
    num_simulation: int,
    figures_path: Path,
    simulation_path: Path,
    desplazamiento: dict,
    voltaje_percolacion: float,
    voltage_CF_destruido: list,
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
    i_set = np.concatenate([data["pp_set"]["datos"][:, 2], data["sp_set"]["datos"][:, 2]])
    v_set = np.concatenate([data["pp_set"]["datos"][:, 1], data["sp_set"]["datos"][:, 1]])
    v_reset = np.concatenate([data["pp_reset"]["datos"][:, 1], data["sp_reset"]["datos"][:, 1]])
    i_reset = np.concatenate([data["pp_reset"]["datos"][:, 2], data["sp_reset"]["datos"][:, 2]])

    # Diccionario de puntos que quieres ubicar
    puntos_x_set = {"a": 1e-6, "b": voltaje_percolacion, "c": 1.1}
    puntos_x_pp_reset = {"d": -0.42, "e": voltage_CF_destruido[0], "f": -1.4}
    puntos_x_sp_reset = {"g": -0.01}

    # Obtener puntos en cada curva
    puntos_set = utils.obtener_puntos_en_curva(
        data["pp_set"]["datos"][:, 1], data["pp_set"]["datos"][:, 2], puntos_x_set
    )
    puntos_x_pp_reset = utils.obtener_puntos_en_curva(
        data["pp_reset"]["datos"][:, 1],
        data["pp_reset"]["datos"][:, 2],
        puntos_x_pp_reset,
    )
    puntos_x_sp_reset = utils.obtener_puntos_en_curva(
        data["sp_reset"]["datos"][:, 1],
        data["sp_reset"]["datos"][:, 2],
        puntos_x_sp_reset,
    )

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
        num_simulation,
        titulo_figura="",
        figures_path=str(save_path),
    )

    Representate.plot_IV_marcado(
        v_set,
        i_set,
        v_reset,
        i_reset,
        num_simulation,
        puntos_totales,
        desplazamiento,
        titulo_figura="",
        figures_path=str(save_path_marcado),
    )

    return None


def resumen_plots(
    k,
    fig_voltage,
    filas_intermedias,
    temperatura,
    materials_map,
    rutas,
    num_simulation,
    voltage,
    params,
    actual_state,
    actual_state_clean_CF,
    matriz_temperaturas_fijas,
    centros_calculados,
    mis_perfiles_extraidos,
    CF_ranges,
    etapa="pp_set",
    columna_perfil=21,
):
    """
    Genera y guarda todas las gráficas de estado, temperatura y muros térmicos
    para un paso específico de la simulación.
    """

    print(f"Representando para el paso {k} con voltaje {fig_voltage} V las filas intermedias son {filas_intermedias}\n")

    # 1. Mapa de temperatura y muros
    Representate.plot_thermal_state_muro(
        temperatura,
        materials_map,
        fig_voltage,
        10,
        save_path=rutas["figures_path"] / f"Mapa_temperatura_{num_simulation}_{round(voltage, 4)}_{etapa}.png",
        device_size=params.device_size,
        filas_intermedias=filas_intermedias,
    )

    # 2. Estados de la matriz
    Representate.RepresentateState(
        matriz=actual_state,
        voltaje=fig_voltage,
        filename=str(rutas["figures_path"]) + f"/State_{num_simulation}_{fig_voltage}_{etapa}.png",
        device_size=params.device_size,
    )

    Representate.RepresentateState(
        matriz=actual_state_clean_CF,
        voltaje=fig_voltage,
        filename=str(rutas["figures_path"]) + f"/State_Clean_{num_simulation}_{fig_voltage}_{etapa}.png",
        device_size=params.device_size,
    )

    # 3. Preparación y plot del muro térmico
    matriz_para_plot_muro = np.copy(matriz_temperaturas_fijas)
    for centro, perfil_filamento in zip(centros_calculados, mis_perfiles_extraidos):
        if centro is not None and perfil_filamento is not None:
            matriz_para_plot_muro[centro, :] = perfil_filamento

    Representate.plot_muro_termico(
        matriz_muros=matriz_para_plot_muro,
        matriz_molde=actual_state_clean_CF,
        filename=rutas["figures_path"] / f"Muro_termico_{num_simulation}_{fig_voltage}_{etapa}.png",
        device_size=params.device_size,
    )

    # 4. Centros de filamentos
    Representate.plot_centros_filamento_det(
        matriz_state=actual_state,
        rangos_CF=CF_ranges,
        filas_intermedias=filas_intermedias,
        centros_calculados=centros_calculados,
        filename=rutas["figures_path"] / f"Centros_filamentos_{num_simulation}_{fig_voltage}_{etapa}.png",
        device_size=params.device_size,
    )

    # # 5. Perfil térmico
    # distancias, perfiles = Temperature.extraer_perfiles_temperatura(
    #     lista_matrices=[temperatura], etiquetas=[f"{fig_voltage}"], columna_x=columna_perfil, atom_size=params.atom_size
    # )

    # Representate.plot_perfil_temperatura(
    #     distancias=distancias,
    #     perfiles=perfiles,
    #     save_path=rutas["figures_path"] / f"Perfil_termico_{num_simulation}_{fig_voltage}_{etapa}.png",
    # )
