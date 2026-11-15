import pandas as pd
import shutil
import csv
import os
import logging

logger = logging.getLogger(__name__)


def leer_txt(ruta_txt):
    """
    Lee un archivo txt y devuelve sus líneas como lista de strings sin saltos de línea.
    """
    try:
        with open(ruta_txt, "r", encoding="utf-8") as f:
            lineas = [linea.strip() for linea in f.readlines()]
        return lineas
    except FileNotFoundError:
        logger.info(f"[❌] Archivo {ruta_txt} no encontrado.")
        return []
    except Exception as e:
        logger.info(f"[❌] Error al leer {ruta_txt}: {e}")
        return []


def leer_csv(ruta_csv, encoding="utf-8"):
    """
    Lee un archivo CSV y devuelve su contenido como un DataFrame de pandas,
    convirtiendo todas las columnas a float si es posible.
    """
    try:
        with open(ruta_csv, "r", encoding=encoding) as f:
            lector = csv.DictReader(f)
            datos = [fila for fila in lector]

        df = pd.DataFrame(datos)

        # 🔧 Conversión a float robusta
        df = df.apply(
            lambda col: pd.to_numeric(
                col.astype(str).str.strip().replace("\ufeff", ""), errors="coerce"
            )
        )

        return df

    except FileNotFoundError:
        logger.info(f"[❌] Archivo {ruta_csv} no encontrado.")
        return pd.DataFrame()
    except Exception as e:
        logger.info(f"[❌] Error al leer {ruta_csv}: {e}")
        return pd.DataFrame()


def leer_txt_as_csv(ruta_txt, separador=",", encoding="utf-8", header="infer"):
    """
    Lee un archivo .txt con estructura CSV y lo carga en un DataFrame de pandas.

    Parámetros:
        ruta_txt (str): Ruta al archivo .txt
        separador (str): Separador de campos (por defecto ',')
        encoding (str): Encoding del archivo (por defecto 'utf-8')
        header (int, list of int, 'infer' or None): Fila de encabezado para pandas.read_csv

    Retorna:
        pd.DataFrame: DataFrame con el contenido del archivo.
    """
    try:
        df = pd.read_csv(ruta_txt, sep=separador, encoding=encoding, header=header)  # pyright: ignore[reportArgumentType]
        return df
    except FileNotFoundError:
        logger.info(f"[❌] Archivo {ruta_txt} no encontrado.")
        return pd.DataFrame()
    except pd.errors.ParserError as e:
        logger.info(f"[❌] Error de parsing en {ruta_txt}: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.info(f"[❌] Error inesperado al leer {ruta_txt}: {e}")
        return pd.DataFrame()


def safe_reset_folder(folder_path):
    # Evita borrar carpetas peligrosas como la raíz del usuario
    if folder_path.strip().lower() in [
        "c:/users/usuario",
        "c:\\users\\usuario",
        "c:/users",
        "c:\\users",
    ]:
        logger.info(f"ADVERTENCIA: No se permite borrar la carpeta protegida: {folder_path}")
        return
    try:
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
        os.makedirs(folder_path)
    except PermissionError as e:
        logger.info(f"Error de permisos al intentar borrar o crear la carpeta: {folder_path}\n{e}")
    except Exception as e:
        logger.info(f"Error inesperado con la carpeta: {folder_path}\n{e}")
