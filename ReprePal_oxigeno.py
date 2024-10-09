import sys
import glob
import pickle
import imageio
import time as time
import matplotlib.pyplot as plt

from RRAM import *
from PIL import Image
from io import BytesIO
from tqdm.contrib.concurrent import process_map

global im

# Asegúrate de que se ha pasado un parámetro
if len(sys.argv) > 1:
    parametro_recibido = sys.argv[1]
    print(f"Parámetro recibido: {parametro_recibido}")
else:
    print("No se ha pasado ningún parámetro.")


# Cargo el fichero con las configuraciones
with open(parametro_recibido, 'rb') as f:
    Oxigeno = pickle.load(f)

# Supongamos que las imágenes están en el subdirectorio "Figuras" y tienen nombres de archivo que siguen el patrón "image*.png"
filenames = glob.glob('Figuras/grafica*.png')

# Crear una lista para almacenar las imágenes
images = []


def process_matrix(args):
    """
    Process a matrix and generate a plot.
    Args:
        args (tuple): A tuple containing the matrix and idx.

    Returns:
        BytesIO: A BytesIO object containing the generated plot image.
    """
    global im

    matrix, idx = args
    if len(plt.get_fignums()) == 0:
        fig, ax = plt.subplots()
        im = None
    else:
        fig, ax = plt.gcf(), plt.gca()

    im = RepresentateState_parall(matrix, fig, ax,  im, color=(0.878, 0.227, 0.370),
                                  filename="Figuras/grafica_" + str(idx+1) + ".png")

    plt.savefig((buffer := BytesIO()), format='png')
    plt.clf()

    return buffer


if __name__ == '__main__':

    NUM_PARALLEL_PROCESSES = 8
    start = time.time()
    args = [(Oxigeno[i], i) for i in range(len(Oxigeno))]
    buffers = process_map(process_matrix, args, max_workers=NUM_PARALLEL_PROCESSES, chunksize=25)
    images = [Image.open(buffer) for buffer in buffers]
    end = time.time()

    print(f"Tiempo de creación de las imágenes: {end - start:.2f} segundos")

    # Definir el tamaño del video (usando la primera imagen)
    sample_image = images[0]
    height, width = sample_image.size[1], sample_image.size[0]

    start = time.time()
    # Crear un escritor de video
    writer = imageio.get_writer('Videos/Oxygen.mp4', fps=12)

    # Cargar y procesar las imágenes una por una
    for img in images:
        img_array = np.array(img)
        writer.append_data(img_array)

    # Cerrar el escritor de video
    writer.close()

    end = time.time()

    print(f"Tiempo de creación del video: {end - start:.2f} segundos")
