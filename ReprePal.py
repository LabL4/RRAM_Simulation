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


def process_matrix(args):
    matrix, idx = args

    RepresentateStateOpt(matrix, filename="Figuras/grafica_" + str(idx) + ".png")

    plt.savefig((buffer := BytesIO()), format='png')
    plt.close()

    # images[idx] = (Image.open(buffer))
    return Image.open(buffer)


NUM_PARALLEL_PROCESSES = 1

args = [(configuraciones_matriz[i], i) for i in range(len(configuraciones_matriz))]
images = process_map(process_matrix, args, max_workers=NUM_PARALLEL_PROCESSES)
