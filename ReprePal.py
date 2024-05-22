import glob
import pickle
from RRAM import *
import time as time
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from multiprocessing import Process, Lock
from tqdm.contrib.concurrent import process_map

# Cargo el fichero con las configuraciones
with open('Configuraciones.pkl', 'rb') as f:
    configuraciones_matriz = pickle.load(f)

# Supongamos que las imágenes están en el subdirectorio "Figuras" y tienen nombres de archivo que siguen el patrón "image*.png"
filenames = glob.glob('Figuras/grafica*.png')

# Crear una lista para almacenar las imágenes
images = []

global im


def process_matrix(args):
    global im

    matrix, idx = args
    if len(plt.get_fignums()) == 0:
        print("No hay figuras")
        fig, ax = plt.subplots()
        im = None
    else:
        fig, ax = plt.gcf(), plt.gca()
    # fig, ax = plt.subplots()

    im = RepresentateStateOptAnto(matrix, fig, ax, im, filename="Figuras/grafica_" + str(idx) + ".png")

    plt.savefig((buffer := BytesIO()), format='png')
    # plt.close()
    plt.clf()

    # images[idx] = (Image.open(buffer))
    return buffer


if __name__ == '__main__':

    NUM_PARALLEL_PROCESSES = 5
    start = time.time()
    args = [(configuraciones_matriz[i], i) for i in range(len(configuraciones_matriz))]
    buffers = process_map(process_matrix, args, max_workers=NUM_PARALLEL_PROCESSES)
    images = [Image.open(buffer) for buffer in buffers]
    end = time.time()

    print(f"Tiempo de creación de las imágenes: {end - start:.2f} segundos")

    start = time.time()
    # Guardar las imágenes como un GIF
    images[0].save('animated_matrix.gif', save_all=True, append_images=images[1:], optimize=False, duration=40, loop=1)
    end = time.time()

    # Imprimir un mensaje de éxito
    print("Las matrices se han guardado correctamente como un GIF en 'animated_matrix.gif'")
    # Imprimir el tiempo que tardó en crearse el GIF en segundos
    print(f"Tiempo de creación del GIF: {end - start:.2f} segundos")
