
from RRAM import *

# comienzo la simulación montecaarlo

long_dispositivo = 10
Atom_size = 0.5

eje_x = round(long_dispositivo / Atom_size)
eje_y = round(long_dispositivo / Atom_size)

num_trampas = 10

actual_state = Generation.initial_state(eje_x, eje_y, num_trampas)


# Guardo la gráfica en el esritorio con el nombre grafica 2
RepresentateState(actual_state, "grafica2.png")

time_simulation = 1
num_pasos = 10
final_volt = 1

paso_temporal = time_simulation / num_pasos

voltaje = 0

for k in range(1, num_pasos + 1):
    # Guardo el estado anterior
    last_state = actual_state

    # Calculo el campo eléctrico
    voltaje = voltaje + final_volt * paso_temporal
    Campo_Electrico = SimpleElectricField(voltaje, long_dispositivo)

    # Calculo la temperatura
    Temperatura = SimpleTemperature()

    # Calculo la probabilidad de generación o recombinación para ello recorro toda kla matriz
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

    print("Paso: ", k)
    # TODO: REVISAR POR QUÉ NO SE GUARDA LA IMAGEN
    RepresentateState(actual_state, "grafica" + str(k) + ".png")
    # Guardo una imagen de la matriz con el nombre de grafica + el numero de paso
