import glob
import matplotlib.pyplot as plt

from RRAM import *
from PIL import Image
from tqdm import tqdm
from io import BytesIO
from icecream import ic


# comienzo la simulación montecaarlo

espesor_dispositivo = 10        # nm
Atom_size = 0.25                # nm

eje_x = round(espesor_dispositivo / Atom_size)
eje_y = round(espesor_dispositivo / Atom_size)

num_trampas = 40

# FIXME: Hay una zona donde nunca se ponen trampas
actual_state = Generation.initial_state(eje_x, eje_y, num_trampas)


# Guardo la gráfica en el esritorio con el nombre grafica 2
RepresentateState(actual_state, "Configuracion_inicial.png")

time_simulation = 1
num_pasos = 10
voltaje_final = 3

paso_guardar = 10

configuraciones_matriz = np.zeros((int((num_pasos / paso_guardar)), eje_x, eje_y))

Temperatura = 300
Campo_Electrico = 0

paso_temporal = time_simulation / num_pasos

voltaje = 0

for k in tqdm(range(0, num_pasos)):
    # Guardo el estado anterior
    last_state = actual_state

    # Calculo la corriente
    voltaje += voltaje_final * paso_temporal
    ic(voltaje)

    # Obtengo la corrriente, antes decido cual usar comprobando si ha percolado o no
    if Percolation.is_path(actual_state):
        # Si ha percolado uso la corriente de percolación
        Corriente = CurentSolver.OmhCurrent(Temperatura, Campo_Electrico)
    else:
        # Si no ha percolado uso la corriente de campo
        Corriente = CurentSolver.poole_frenkel(Temperatura, Campo_Electrico)
        ic(Temperatura, Campo_Electrico)
        ic(CurentSolver.poole_frenkel(Temperatura, Campo_Electrico))

    # Obtengo los valores del campo eléctrico y la temperatura
    Campo_Electrico = SimpleElectricField(voltaje, espesor_dispositivo)
    ic(Campo_Electrico)
    Temperatura = Temperature_Joule(voltaje, Corriente)

    # Calculo la probabilidad de generación o recombinación para ello recorro toda la matriz
    for i in range(eje_x):
        for j in range(eje_y):
            if actual_state[i, j] == 0:
                # TODO: REVISAR PROBABILIDAD QUE A VECES SALE MAYOR DE 1
                # TODO: HACER UN REESCALADO DE LOS VALORES PARA EVITAR TENER QUE TRABAJAR CON NUMEROS TAN GRANDES
                prob_generacion = Generation.generation(paso_temporal, Campo_Electrico, Temperatura)

                # genero un número aleatorio entre 0 y 1
                random_number = np.random.rand()
                if random_number < prob_generacion:
                    actual_state[i, j] = 1  # Generación
            else:
                # TODO: REVISAR PROBABILIDAD QUE A VECES SALE MAYOR DE 1
                # TODO: HACER UN REESCALADO DE LOS VALORES PARA EVITAR TENER QUE TRABAJAR CON NUMEROS TAN GRANDES
                prob_recombinacion = Recombination.recombination(paso_temporal, i, Campo_Electrico, Temperatura)

                # genero un número aleatorio entre 0 y 1
                random_number = np.random.rand()
                if random_number < prob_recombinacion:
                    actual_state[i, j] = 0  # Recombinación

    # Guardo el estado actual CADA paso_guardar PASOS MONTECARLO
    if k % paso_guardar == 0:
        configuraciones_matriz[int(k / paso_guardar) - 1] = actual_state

# Supongamos que las imágenes están en el subdirectorio "Figuras" y tienen nombres de archivo que siguen el patrón "image*.png"
filenames = glob.glob('Figuras/grafica*.png')

# Crear una lista para almacenar las imágenes
images = []
k = 1
for matrix in tqdm(configuraciones_matriz):
    # Llamar a tu función para representar la matriz y guardar la imagen
    RepresentateStateOpt(matrix, filename="Figuras/grafica" + str(k) + ".png")

    plt.savefig((buffer := BytesIO()), format='png')
    plt.close()

    # Leer la imagen guardada y agregarla a la lista de imágenes
    images.append(Image.open(buffer))

    k += 1

# Guardar las imágenes como un GIF
images[0].save('animated_matrix.gif', save_all=True, append_images=images[1:], optimize=False, duration=500, loop=0)

# Imprimir un mensaje de éxito
print("Las matrices se han guardado correctamente como un GIF en 'animated_matrix.gif'")
