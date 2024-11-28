import sys
import glob
import time
import pickle
import imageio
import numpy as np
from RRAM import *
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from tqdm.contrib.concurrent import process_map

# Definimos las funciones necesarias para modularizar el código


def cargar_configuraciones(archivo_pickle):
    """
    Carga las configuraciones desde un archivo pickle.

    Args:
        archivo_pickle (str): La ruta del archivo pickle.

    Returns:
        list: Lista de matrices cargadas del archivo.
    """
    with open(archivo_pickle, 'rb') as f:
        Oxigeno = pickle.load(f)
    return Oxigeno


def procesar_matriz(args):
    """
    Procesa una matriz y genera un gráfico.

    Args:
        args (tuple): Tupla que contiene la matriz y el índice.

    Returns:
        BytesIO: Objeto BytesIO que contiene la imagen del gráfico generado.
    """
    global im
    matrix, idx = args

    if len(plt.get_fignums()) == 0:
        fig, ax = plt.subplots()
        im = None
    else:
        fig, ax = plt.gcf(), plt.gca()

    im = RepresentateState_parall(matrix, fig, ax, im, color=(0.878, 0.227, 0.370),
                                  filename="Figuras/grafica_" + str(idx+1) + ".png")

    plt.savefig((buffer := BytesIO()), format='png')
    plt.clf()

    return buffer


def crear_imagenes(Oxigeno, num_procesos=8, chunk_size=25):
    """
    Crea imágenes a partir de matrices utilizando procesamiento paralelo.

    Args:
        Oxigeno (list): Lista de matrices a procesar.
        num_procesos (int): Número de procesos paralelos. Default: 8.
        chunk_size (int): Tamaño de los chunks para el procesamiento. Default: 25.

    Returns:
        list: Lista de objetos PIL Image.
    """
    start = time.time()
    args = [(Oxigeno[i], i) for i in range(len(Oxigeno))]
    buffers = process_map(procesar_matriz, args, max_workers=num_procesos, chunksize=chunk_size)
    images = [Image.open(buffer) for buffer in buffers]
    end = time.time()

    print(f"Tiempo de creación de las imágenes: {end - start:.2f} segundos")
    return images


def crear_video_con_imagenes(images, video_path, fps=12):
    """
    Crea un video a partir de una lista de imágenes.

    Args:
        images (list): Lista de objetos PIL Image.
        video_path (str): La ruta de salida del video.
        fps (int): Cuadros por segundo para el video. Default: 12.
    """
    # Definir el tamaño del video (usando la primera imagen)
    sample_image = images[0]
    height, width = sample_image.size[1], sample_image.size[0]

    # Crear el video
    start = time.time()
    writer = imageio.get_writer(video_path, fps=fps)

    for img in images:
        img_array = np.array(img)
        writer.append_data(img_array)

    writer.close()
    end = time.time()

    print(f"Tiempo de creación del video: {end - start:.2f} segundos")


# Función principal que puede ser llamada externamente
def procesar_y_generar_video(archivo_configuraciones, video_output, num_procesos=8, chunk_size=25, fps=12):
    """
    Procesa el archivo de configuraciones y genera un video con las imágenes resultantes.

    Args:
        archivo_configuraciones (str): Ruta del archivo pickle con las configuraciones.
        video_output (str): Ruta del archivo de salida del video.
        num_procesos (int): Número de procesos paralelos para la creación de imágenes. Default: 8.
        chunk_size (int): Tamaño de los chunks para el procesamiento paralelo. Default: 25.
        fps (int): Cuadros por segundo del video generado. Default: 12.
    """
    # Cargar configuraciones
    Oxigeno = cargar_configuraciones(archivo_configuraciones)

    # Crear imágenes
    images = crear_imagenes(Oxigeno, num_procesos, chunk_size)

    # Crear video a partir de las imágenes
    crear_video_con_imagenes(images, video_output, fps)


# Solo ejecuta el proceso si se llama directamente este script
if __name__ == '__main__':
    if len(sys.argv) > 2:
        data_path = sys.argv[1]
        save_path = sys.argv[1]
        print(f"Procesando archivo: {data_path}")
        procesar_y_generar_video(data_path, save_path)
    else:
        print("Por favor, proporciona el archivo de configuraciones como parámetro.")
