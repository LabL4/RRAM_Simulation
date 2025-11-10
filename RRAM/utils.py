from typing import List, Dict
from . import Representate, utils
from pathlib import Path

import numpy as np
import pickle
import csv


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
    pkl_path: Path, figures_path: Path, voltage: float, plot_state: bool = True
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
        Representate.RepresentateState(actual_state, voltage, str(figures_path))

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
    i_set = np.concatenate(
        [data["pp_set"]["datos"][:, 2], data["sp_set"]["datos"][:, 2]]
    )
    v_set = np.concatenate(
        [data["pp_set"]["datos"][:, 1], data["sp_set"]["datos"][:, 1]]
    )
    v_reset = np.concatenate(
        [data["pp_reset"]["datos"][:, 1], data["sp_reset"]["datos"][:, 1]]
    )
    i_reset = np.concatenate(
        [data["pp_reset"]["datos"][:, 2], data["sp_reset"]["datos"][:, 2]]
    )

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
