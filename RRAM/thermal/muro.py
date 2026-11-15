"""Muro térmico entre filamentos y extracción de perfiles de temperatura."""

import numpy as np
import logging

logger = logging.getLogger(__name__)


def filas_filamento(Ny: int, centros, grosor) -> np.ndarray:
    """
    Máscara 1D (Ny,) de las filas de los filamentos: centro ± grosor de cada uno.

    `grosor` es un entero común o una lista con un valor por filamento (misma banda
    que Generation.create_custom_mask, pero en 1D).
    """
    grosores = [int(grosor)] * len(centros) if isinstance(grosor, (int, np.integer)) else list(grosor)
    if len(grosores) != len(centros):
        raise ValueError("Debe haber un grosor asignado para cada centro de filamento.")
    filas = np.zeros(Ny, dtype=bool)
    for centro, g in zip(centros, grosores):
        filas[max(0, centro - g) : min(Ny, centro + g + 1)] = True
    return filas


def calcular_filas_intermedias(centros: list) -> tuple[list, list]:
    """
    Calcula la fila central (punto medio) entre los centros de filamentos consecutivos
    y las distancias físicas desde los centros hasta dicha fila media.

    Argumentos:
    - centros: Lista de enteros con los centros de cada filamento (Ej: [20, 50, 80]).
    - atom_size: Tamaño físico de cada celda del grid (Ej: 0.25).

    Retorna:
    - filas_medias: Lista de enteros con las filas intermedias (Ej: [35, 65]).
    - distancias: Lista de tuplas con las distancias desde cada filamento a la fila media (Ej: [(15, 15), (15, 15)]).
                  Formato: [(dist_centro1_a_medio, dist_centro2_a_medio), ...]
    """
    filas_medias = []
    distancias = []

    # Si hay menos de 2 filamentos, no hay distancia intermedia que calcular
    if not centros or len(centros) < 2:
        return filas_medias, distancias

    # Recorremos la lista emparejando el elemento actual con el siguiente
    for i in range(len(centros) - 1):
        centro_actual = centros[i]
        centro_siguiente = centros[i + 1]

        # En caso de que un filamento no se haya formado (None), no podemos calcular el medio
        if centro_actual is None or centro_siguiente is None:
            continue

        # Calculamos la media aritmética entre ambos centros
        medio = (centro_actual + centro_siguiente) / 2.0

        # Redondeamos al entero más cercano (fila de la matriz)
        fila_media = int(np.round(medio))
        filas_medias.append(fila_media)

        # Calculamos cuántas casillas hay de diferencia usando valor absoluto (abs)
        # y lo multiplicamos por el tamaño físico de la celda (atom_size)
        # NOTA: el muro superior se coloca en fila_media, pero el muro inferior se
        # coloca en fila_media + 1 (ver colocar_muro_termico), por lo que su
        # distancia al centro del filamento inferior es una casilla menor.
        distancia_actual = abs(fila_media - centro_actual)
        distancia_siguiente = abs(centro_siguiente - (fila_media + 1))

        # Guardamos ambas distancias en la lista
        distancias.append((distancia_actual, distancia_siguiente))

    return filas_medias, distancias


def calcular_perfiles_muro(
    perfiles_filamentos: list,
    distancias_casillas: list,
    pendiente_temperatura: float,
    atom_size: float,
    T_ambient: float,
) -> list:
    """
    Calcula el perfil de temperatura 1D para cada muro térmico basándose en
    la temperatura de su filamento correspondiente columna a columna.
    """

    # BLOQUE 1: Preparación del contenedor de resultados
    # -----------------------------------------------------------------
    # Crearemos una lista que almacenará tuplas. Cada tupla contendrá
    # (perfil_muro_arriba, perfil_muro_abajo) en formato de array 1D.
    perfiles_muros = []

    # Verificación de seguridad: debe haber un perfil de filamento más que distancias
    if len(perfiles_filamentos) < 2 or not distancias_casillas:
        return perfiles_muros

    # BLOQUE 2: Iteración sobre las interfaces (muros)
    # -----------------------------------------------------------------
    # Recorremos la lista de distancias. El índice 'i' representa el espacio
    # INTERMEDIO entre el filamento 'i' (arriba) y el filamento 'i-1' (abajo).
    for i in range(len(distancias_casillas)):
        casillas_arriba, casillas_abajo = distancias_casillas[i]

        # BLOQUE 3: Extracción de los perfiles de los filamentos origen
        # -----------------------------------------------------------------
        # El muro superior está influenciado por el filamento que tiene encima (i)
        T_filamento_arriba = perfiles_filamentos[i]

        # El muro inferior está influenciado por el filamento que tiene debajo (i+1)
        T_filamento_abajo = perfiles_filamentos[i + 1]

        # print("El perfil de temperatura del filamento superior en la funcion del muro es:", T_filamento_arriba[10:21])
        # print("El perfil de temperatura del filamento inferior en la funcion del muro es:", T_filamento_abajo[10:21])

        # BLOQUE 4: Cálculo vectorizado de la temperatura
        # -----------------------------------------------------------------
        # Como T_filamento es un array de numpy (ej: 100 valores, uno por columna),
        # al sumar el término matemático se realiza la operación celda por celda
        # instantáneamente sin necesidad de hacer un bucle for anidado.

        # Ecuación: T_muro = T_filamento + pendiente_temperatura * distancia
        # (Nota: 'pendiente_temperatura' debe ser negativo ya que la temperatura cae al alejarse)
        T_muro_arriba = T_filamento_arriba + pendiente_temperatura * (casillas_arriba * atom_size * 1e9)
        T_muro_abajo = T_filamento_abajo + pendiente_temperatura * (casillas_abajo * atom_size * 1e9)

        # Imprimir los primeros 10 números del perfil para depuración
        # print("Temperatura calculada para el muro superior antes de limitar:", T_muro_arriba[10:21])
        # print("Temperatura calculada para el muro inferior antes de limitar:", T_muro_abajo[10:21])

        # Si la temperatura calculada para el muro es mayor que la del filamento, la limitamos a la temperatura del filamento
        T_muro_arriba = np.minimum(T_muro_arriba, T_filamento_arriba)
        T_muro_abajo = np.minimum(T_muro_abajo, T_filamento_abajo)

        # Si la temperatura calculada para el muro es menor que la temperatura ambiente, la limitamos a la temperatura ambiente
        T_muro_arriba = np.maximum(T_muro_arriba, T_ambient)
        T_muro_abajo = np.maximum(T_muro_abajo, T_ambient)

        # print("Temperatura calculada para el muro superior después de limitar:", T_muro_arriba[10:21])
        # print("Temperatura calculada para el muro inferior después de limitar:", T_muro_abajo[10:21])

        # Guardamos el par de perfiles calculados para esta interfaz
        perfiles_muros.append((T_muro_arriba, T_muro_abajo))

    return perfiles_muros


def colocar_muro_termico(
    matriz_molde: np.ndarray, filas_intermedias: list, perfiles_muros_calculados: list
) -> np.ndarray:
    """
    Coloca los perfiles 1D de temperatura en las filas correspondientes de la matriz 2D.
    """
    matriz_muros = np.zeros_like(matriz_molde, dtype=float)
    Ny, Nx = matriz_muros.shape

    for i, fila_mid in enumerate(filas_intermedias):
        if fila_mid is None:
            continue

        # Extraemos los perfiles 1D ya calculados por tu nueva función
        perfil_muro_arriba, perfil_muro_abajo = perfiles_muros_calculados[i]

        # 1er Muro: Fila Intermedia
        mask_vacio_arriba = matriz_molde[fila_mid, :] == 0
        # Numpy asigna el valor del perfil SOLO en las columnas donde la máscara es True
        matriz_muros[fila_mid, mask_vacio_arriba] = perfil_muro_arriba[mask_vacio_arriba]

        # 2do Muro: Una fila por debajo
        fila_encima = fila_mid + 1
        if 0 <= fila_encima < Ny:
            mask_vacio_abajo = matriz_molde[fila_encima, :] == 0
            matriz_muros[fila_encima, mask_vacio_abajo] = perfil_muro_abajo[mask_vacio_abajo]

    return matriz_muros


def extraer_perfiles_filamentos(matriz_temperaturas: np.ndarray, filas_centros: list) -> list:
    """
    Extrae los perfiles de temperatura 1D (fila completa) de una matriz térmica 2D,
    basándose en una lista de filas específicas (centros de los filamentos).

    Argumentos:
    - matriz_temperaturas: Matriz 2D con los resultados del solver térmico.
    - filas_centros: Lista de enteros indicando las filas a extraer (Ej: [20, 80]).

    Retorna:
    - perfiles: Lista de arrays 1D correspondientes a la temperatura de cada filamento.
    """

    # BLOQUE 1: Preparación
    perfiles = []
    Ny, Nx = matriz_temperaturas.shape

    # BLOQUE 2: Iteración y Extracción
    for fila in filas_centros:
        # Caso A: El filamento no se formó (su centro es None)
        if fila is None:
            # Añadimos None a la lista para mantener la correspondencia de índices
            perfiles.append(None)
            logger.info("Advertencia: Se detectó un filamento no formado (None). Perfil omitido.")

        # Caso B: La fila es un índice válido dentro de la matriz
        elif 0 <= fila < Ny:
            # Extraemos la fila completa (todas las columnas de esa fila)
            # Usamos .copy() para evitar modificar la matriz original accidentalmente
            perfil_1d = matriz_temperaturas[fila, :].copy()
            perfiles.append(perfil_1d)

        # Caso C: Índice fuera de los límites de la matriz (Error de entrada)
        else:
            raise IndexError(
                f"La fila solicitada ({fila}) está fuera de los límites de la matriz térmica (0 a {Ny - 1})."
            )

    return perfiles


def componer_matriz_muro_plot(matriz_muros, centros, perfiles):
    """
    Matriz del muro térmico con el perfil de cada filamento en su fila central, para guardarla.

    Devuelve None si el paso no tuvo muro (`matriz_muros` o `perfiles` son None), de modo que
    el guardado nunca arrastra el muro de un paso anterior.
    """
    if matriz_muros is None or perfiles is None:
        return None
    resultado = np.copy(matriz_muros)
    for centro, perfil in zip(centros, perfiles):
        if centro is not None and perfil is not None:
            resultado[centro, :] = perfil
    return resultado


def extraer_perfiles_temperatura(lista_matrices: list, etiquetas: list, columna_x: int, atom_size: float):
    """
    Extrae el perfil vertical de temperatura para múltiples matrices.

    Argumentos:
    - lista_matrices: Lista de arrays 2D de temperatura.
    - etiquetas: Nombres para la leyenda (ej: ["1.0 V", "1.5 V", "2.0 V"]).
    - columna_x: Índice de la columna a evaluar.
    - atom_size: Tamaño físico de cada celda en nm.

    Retorna:
    - distancias: Array 1D con el eje X físico.
    - perfiles: Diccionario con formato { "etiqueta": array_temperaturas_1D }
    """
    if len(lista_matrices) != len(etiquetas):
        raise ValueError("Debe haber el mismo número de matrices que de etiquetas.")

    # Usamos la forma de la primera matriz para calcular las distancias
    distancias = np.arange(0, lista_matrices[0].shape[0]) * atom_size

    perfiles = {}
    for matriz, etiqueta in zip(lista_matrices, etiquetas):
        # Extraemos solo la columna deseada
        perfiles[etiqueta] = matriz[:, columna_x]

    return distancias, perfiles
