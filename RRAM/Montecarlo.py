import csv
import pickle
import numpy as np

# from tqdm import tqdm


def read_csv_to_dic(cvs_path: str):
    """
    Reads a CSV file and returns its contents as a list of dictionaries.
    Args:
        cvs_path (str): The name of the CSV file to read.

    Returns:
        list: A list of dictionaries, where each dictionary represents a row in the CSV file.
    """
    with open(cvs_path, mode='r') as archivo:
        reader = csv.DictReader(archivo)
        data = [row for row in reader]
    return data


# Función para unir dos archivos pickle y guardar el resultado en un nuevo archivo
def merge_pickles_to_array(file1, file2, output_file):
    try:
        # Cargar datos del primer archivo
        with open(file1, 'rb') as f1:
            data1 = pickle.load(f1)
    except (FileNotFoundError, EOFError):
        data1 = np.array([])  # Si no existe, inicializamos como un array vacío

    try:
        # Cargar datos del segundo archivo
        with open(file2, 'rb') as f2:
            data2 = pickle.load(f2)
    except (FileNotFoundError, EOFError):
        data2 = np.array([])  # Si no existe, inicializamos como un array vacío

    # Convertir ambos datos en arrays de numpy (si no lo son ya)
    data1 = np.array(data1)
    data2 = np.array(data2)

    # Unir ambos arrays en uno solo
    combined_data = np.concatenate((data1, data2))

    # Guardar los datos combinados en un archivo pickle
    with open(output_file, 'wb') as f_out:
        pickle.dump(combined_data, f_out)
