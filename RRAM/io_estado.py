"""Lectura y escritura de estados, datos y CSV."""

from typing import List, Dict
from typing import Optional
from pathlib import Path
import numpy as np
import csv


def cargar_estado(load_data_path: Path) -> np.ndarray:
    """
    Carga el estado de configuración desde archivo pkl y genera una imagen de ese estado.

    Args:
        pkl_path (Path): Ruta del pkl con el estado.
        figures_path (Path): Ruta donde se guardará la imagen generada, debe incluir el nombre del archivo CON extensión.
        plot_state (bool): Indica si se debe generar la imagen del estado, por defecto es True.

    Returns:
        np.ndarray: Matriz con el estado inicial cargado.
    """

    # Cargamos directamente el archivo .npz y extraemos la matriz 'actual_state'
    datos = np.load(f"{load_data_path}.npz")
    actual_state = datos["actual_state"]
    datos.close()  # Siempre es buena práctica cerrar el archivo

    return actual_state


def guardar_datos(save_path_data: Path, headers: Optional[dict] = None, **vectores) -> None:
    """
    Guarda múltiples vectores, matrices o datos escalares en un archivo .npz comprimido.
    Asocia dinámicamente encabezados a los datos si se proporcionan.

    Args:
        ruta_destino (Path): Ruta del archivo donde se guardarán los datos.
        headers (dict, opcional): Diccionario con los encabezados. Las claves deben
                                  coincidir con los nombres dados en **vectores.
        **vectores: Datos a guardar pasados como argumentos con nombre (ej: datos_iv=matriz).
    """
    if headers is None:
        headers = {}

    datos_seguros = {}

    for nombre, vector in vectores.items():
        if vector is not None:
            datos_seguros[nombre] = vector

            # Si el usuario ha proporcionado un encabezado para este vector específico,
            # lo guardamos en el .npz añadiendo el prefijo 'header_' al nombre original.
            if nombre in headers:
                # Lo convertimos a array de NumPy de 1 elemento para guardarlo de forma nativa
                datos_seguros[f"header_{nombre}"] = np.array([headers[nombre]])
        else:
            # Si se pasa un None, guardamos un array vacío para mantener la estructura de claves
            datos_seguros[nombre] = np.array([])

    # Guardamos todo el paquete de forma consolidada y comprimida
    np.savez_compressed(save_path_data.with_suffix(".npz"), **datos_seguros)


def guardar_estado_intermedio(ruta_destino: Path, etapa: str, num_simulation: int, k: int, **matrices) -> None:
    """
    Guarda todas las matrices relevantes de un paso de simulación en un único archivo .npz comprimido.
    Garantiza que todas las claves pasadas existan en el archivo final.
    """

    # 1. Reemplazamos los valores None por un array vacío de NumPy.
    # Esto es mejor que guardar 'None' a secas, porque NumPy prefiere trabajar con arrays.
    matrices_seguras = {}
    for nombre, matriz in matrices.items():
        if matriz is None:
            matrices_seguras[nombre] = np.array([])  # Array vacío indicando que no hay datos
        else:
            matrices_seguras[nombre] = matriz

    # 2. Construimos el nombre del archivo estandarizado
    nombre_archivo = ruta_destino / f"Estado_{etapa}_sim_{num_simulation}_paso_{k}.npz"

    # 3. Guardamos de forma comprimida asegurando que todas las claves se escriben
    np.savez_compressed(nombre_archivo, **matrices_seguras)


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
