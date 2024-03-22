
from RRAM import *

# comienzo la simulación montecaarlo

Longitud_Dispositivo = 10
Atom_size = 0.5

Eje_x = round(Longitud_Dispositivo / Atom_size)
Eje_y = round(Longitud_Dispositivo / Atom_size)

num_trampas = 10

Matrix_Configuation = Generation.initial_state(Eje_x, Eje_y, num_trampas)
# Guardo la gráfica en el esritorio con el nombre grafica 2
RepresentateState(Matrix_Configuation, "C:/Users/anton/Desktop/grafica2.png")
