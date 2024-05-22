import glob
import pandas as pd
import matplotlib.pyplot as plt

from re import T
from RRAM import *
from PIL import Image
from tqdm import tqdm
from io import BytesIO
from icecream import ic


# Creo el excel donde voy a sacar todos los datos
df = pd.DataFrame(columns=['Tiempo simulacion', 'Voltaje', 'Campo Eléctrico', 'Corriente', 'Temperatura',
                  'Probabilidad de Generación', 'Probabilidad de Recombinación', 'Percolacion'])

# comienzo la simulación montecaarlo

espesor_dispositivo = 10        # nm
Atom_size = 0.25                 # nm

eje_x = round(espesor_dispositivo / Atom_size)
eje_y = round(espesor_dispositivo / Atom_size)

num_trampas = 20

# FIXME: Hay una zona donde nunca se ponen trampas
actual_state = Generation.initial_state_priv(eje_x, eje_y, num_trampas)


# Guardo la gráfica en el esritorio con el nombre grafica 2
RepresentateState(actual_state, "Configuracion_inicial.png")

total_simulation_time = 1
num_pasos = 1000
paso_temporal = total_simulation_time / num_pasos

voltaje_final = 1

paso_guardar = 1

configuraciones_matriz = np.zeros((int((num_pasos / paso_guardar)), eje_x, eje_y))

# Configuraciones iniciales:
Temperatura = 350
Campo_Electrico = 0
voltaje = 0
simulation_time = 0
Corriente = 0

# ic.disable()

for k in tqdm(range(0, num_pasos)):
    # Guardo el estado anterior
    last_state = actual_state

    simulation_time = paso_temporal*k

    # Calculo la corriente
    voltaje += voltaje_final * paso_temporal

    # Obtengo la corrriente, antes decido cual usar comprobando si ha percolado o no
    # TODO: Revisar por qué me dice que ha percolado si no lo ha hecho
    if Percolation.is_path(actual_state):
        # Si ha percolado uso la corriente de percolación
        # Corriente = CurentSolver.OmhCurrent(Temperatura, Campo_Electrico)
        # print("Ha percolado")
        pass
    else:
        # Si no ha percolado uso la corriente de campo
        # TODO: REVISAR QUE LA CORRIENTE TIENE LAS UNIDADES CORRECTAS PORQUE NO CUADRAN VALORES.
        Corriente = 1000*CurentSolver.poole_frenkel(Temperatura, Campo_Electrico)

    # Obtengo los valores del campo eléctrico y la temperatura
    Campo_Electrico = SimpleElectricField(voltaje, espesor_dispositivo*1e-9)
    Temperatura = Temperature_Joule(voltaje, Corriente, T_0=350)

    # Calculo la probabilidad de generación o recombinación para ello recorro toda la matriz
    for i in range(eje_x):
        for j in range(eje_y):
            if actual_state[i, j] == 0:
                # TODO: REVISAR PROBABILIDAD QUE A VECES SALE MAYOR DE 1
                # TODO: HACER UN REESCALADO DE LOS VALORES PARA EVITAR TENER QUE TRABAJAR CON NUMEROS TAN GRANDES
                prob_generacion = Generation.generation(paso_temporal, Campo_Electrico, Temperatura)
                if (i == 0 and j == 0):
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

    # Guardo los datos en el excel
    df.loc[k] = [simulation_time, voltaje, Campo_Electrico, Corriente, Temperatura,
                 prob_generacion, prob_recombinacion, Percolation.is_path(actual_state)]

    # Guardo el estado actual CADA paso_guardar PASOS MONTECARLO
    if k % paso_guardar == 0:
        configuraciones_matriz[int(k / paso_guardar) - 1] = actual_state

# Guardo los datos en un excel
df.to_excel('resultados.xlsx', index=False)

# Supongamos que las imágenes están en el subdirectorio "Figuras" y tienen nombres de archivo que siguen el patrón "image*.png"
filenames = glob.glob('Figuras/grafica*.png')

# Crear una lista para almacenar las imágenes
images = []
k = 1
for matrix in tqdm(configuraciones_matriz):
    # Llamar a tu función para representar la matriz y guardar la imagen
    RepresentateStateOpt(matrix, filename="Figuras/grafica_" + str(k) + ".png")

    plt.savefig((buffer := BytesIO()), format='png')
    plt.close()

    # Leer la imagen guardada y agregarla a la lista de imágenes
    images.append(Image.open(buffer))
    k += 1

# Guardar las imágenes como un GIF
images[0].save('animated_matrix.gif', save_all=True, append_images=images[1:], optimize=False, duration=40, loop=1)

# Imprimir un mensaje de éxito
print("Las matrices se han guardado correctamente como un GIF en 'animated_matrix.gif'")
