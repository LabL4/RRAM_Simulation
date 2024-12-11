import os
import sys
import glob
import pickle
import imageio
import numpy as np
import time as time
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from tqdm.contrib.concurrent import process_map

import time
import numpy as np
from PIL import Image
from moviepy import ImageSequenceClip

global im
# ruta_raiz = 'C:/Users/Usuario/Documents/GitHub/RRAM_Simulation/'  # Ruta en el PC
ruta_raiz = '/Users/antonio_lopez_torres/Documents/GitHub/RRAM_Simulation/' # Ruta en el mac

# Add the directory containing the RRAM module to the Python path
module_path = os.path.abspath(os.path.join('..', ruta_raiz))
if module_path not in sys.path:
    sys.path.append(module_path)

from RRAM import Representate as rp

sys.path.append(ruta_raiz)

data_path = ruta_raiz + 'Results/reset/Oxygen_pp_reset_0.pkl'
save_path = ruta_raiz + 'Videos/Oxygen_pp_reset.mp4'

# Asegúrate de que se ha pasado un parámetro
if len(sys.argv) > 1:
    data_path = sys.argv[1]
    save_path = sys.argv[2]
    print(f"Ruta del archivo: {data_path}")
    print(f"Ruta de guardado: {save_path}")
else:
    print("No se ha pasado ningún parámetro.")


# Cargo el fichero con las configuraciones
with open(data_path, 'rb') as f:
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

    im = rp.RepresentateState_parall(matrix, fig, ax,  im, color=(0.878, 0.227, 0.370),
                                  filename="Figuras/grafica_" + str(idx+1) + ".png")

    plt.savefig((buffer := BytesIO()), format='png')
    plt.clf()

    return buffer


if __name__ == '__main__':

    NUM_PARALLEL_PROCESSES = 10
    start = time.time()
    args = [(Oxigeno[i], i) for i in range(len(Oxigeno))]
    buffers = process_map(process_matrix, args, max_workers=NUM_PARALLEL_PROCESSES, chunksize=50)
    images = [Image.open(buffer) for buffer in buffers]
    end = time.time()

    print(f"Tiempo de creación de las imágenes: {end - start:.2f} segundos")

    # Convertir las imágenes a una secuencia de numpy arrays
    image_sequence = [np.array(img) for img in images]

    start = time.time()
    # Crear el video usando moviepy
    clip = ImageSequenceClip(image_sequence, fps=12)
    clip.write_videofile(save_path, codec='libx264')

    end = time.time()

    print(f"Tiempo de creación del video: {end - start:.2f} segundos")
