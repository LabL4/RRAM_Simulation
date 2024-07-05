import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Varias funciones para representar los datos obtenidos de la simulación

def Plot_panel(data_path: str):
    """
    Función que representa los datos obtenidos de la simulación en un panel con 4 subplots, acepta un archivo csv con los datos con la siguiente estructura:
        - La primera columna contiene la variable independiente
        - Las siguientes columnas contienen las variables dependientes

    Args:
    data_path: contiene la ruta del archivo de datos, se encuentra en la primera columna la variable independiente 
               y en las siguientes columnas las variables dependientes

    Returns:
        La figura con los 4 subplots representando los datos
    """

    # leo los datos desde el csv
    data = pd.read_csv(data_path)

    # Elimino la primera fila que son los nombres de las columnas
    data = data.values[1:, :]

    # Extraigo la variable independiente
    x = data[:, 0]

    # Extraigo las variables dependientes
    y = data[:, 1:]

    # Creo la figura que será un panel con 4 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    # Creo el primer subplot
    ax1.plot(x, y[:, 0])
    ax1.set_title('Velocidad')

    # Creo el segundo subplot
    ax2.plot(x, y[:, 1])
    ax2.set_title('desplazamiento')

    # Creo el tercer subplot
    ax3.plot(x, y[:, 2])
    ax3.set_title('Probabilidad generacion')

    # Ajustamos el espacio entre los plots
    fig.tight_layout()

    # Guardo la figura
    plt.savefig('Results/Panel_' + data_path + '.png')

    return None
