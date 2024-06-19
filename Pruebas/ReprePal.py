import glob
import pickle
import imageio
from RRAM import *
import time as time
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from tqdm.contrib.concurrent import process_map

global im

# Cargo el fichero con las configuraciones
with open('Configuraciones.pkl', 'rb') as f:
    configuraciones_matriz = pickle.load(f)

# Supongamos que las imágenes están en el subdirectorio "Figuras" y tienen nombres de archivo que siguen el patrón "image*.png"
filenames = glob.glob('Figuras/grafica*.png')

# Crear una lista para almacenar las imágenes
images = []


def process_matrix(args):
    global im

    matrix, idx = args
    if len(plt.get_fignums()) == 0:
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

    NUM_PARALLEL_PROCESSES = 7
    start = time.time()
    args = [(configuraciones_matriz[i], i) for i in range(len(configuraciones_matriz))]
    buffers = process_map(process_matrix, args, max_workers=NUM_PARALLEL_PROCESSES, chunksize=25)
    images = [Image.open(buffer) for buffer in buffers]
    end = time.time()

    print(f"Tiempo de creación de las imágenes: {end - start:.2f} segundos")

    # Definir el tamaño del video (usando la primera imagen)
    sample_image = images[0]
    height, width = sample_image.size[1], sample_image.size[0]

    start = time.time()
    # Crear un escritor de video
    writer = imageio.get_writer('animated_matrix.mp4', fps=25)

    # Cargar y procesar las imágenes una por una
    for img in images:
        img_array = np.array(img)
        writer.append_data(img_array)

    # Cerrar el escritor de video
    writer.close()

    end = time.time()

    print(f"Tiempo de creación del video: {end - start:.2f} segundos")
