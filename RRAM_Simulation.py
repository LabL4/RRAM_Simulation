from RRAM import *

# comienzo la simulación montecaarlo

espesor_dispositivo = 10
Atom_size = 0.5

eje_x = round(espesor_dispositivo / Atom_size)
eje_y = round(espesor_dispositivo / Atom_size)

num_trampas = 20

# FIXME: Hay una zona donde nunca se ponen trampas
actual_state = Generation.initial_state(eje_x, eje_y, num_trampas)


# Guardo la gráfica en el esritorio con el nombre grafica 2
RepresentateState(actual_state, "Configuracion_inicial.png")

time_simulation = 1
num_pasos = 100
voltaje_final = 1

Temperatura = 300
Campo_Electrico = 0

paso_temporal = time_simulation / num_pasos

voltaje = 0

for k in range(1, num_pasos + 1):
    # Guardo el estado anterior
    last_state = actual_state

    # Calculo la corriente
    voltaje += voltaje_final * paso_temporal

    # Obtengo la corrriente, antes decido cual usar comprobando si ha percolado o no
    if Percolation.is_path(actual_state):
        # Si ha percolado uso la corriente de percolación
        Corriente = CurentSolver.poole_frenkel(Temperatura, Campo_Electrico)
    else:
        # Si no ha percolado uso la corriente de campo
        Corriente = CurentSolver.poole_frenkel(Temperatura, Campo_Electrico)

    # Obtengo los valores del campo eléctrico y la temperatura
    Campo_Electrico = SimpleElectricField(voltaje, espesor_dispositivo)
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

    # Guardo el estado actual CADA 10 PASOS
    if k % 10 == 0:
        print("Paso: ", k)
        RepresentateState(actual_state, "grafica" + str(k) + ".png")
