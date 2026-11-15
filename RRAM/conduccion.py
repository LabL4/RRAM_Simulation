import logging
import math

import numpy as np
from scipy.constants import Boltzmann, elementary_charge, epsilon_0

logger = logging.getLogger(__name__)


def mapa_resistencias(
    cf_matrix: np.ndarray,
    temperatura,
    sigma_0: float,
    alpha_T: float,
    T_0: float,
    delta_z_r: float,
) -> np.ndarray:
    """
    Construye el MAPA DE RESISTENCIAS por celda a partir de la geometría del
    filamento y del mapa de temperaturas.

    Es el análogo eléctrico de `fvm.crear_matriz_materiales`: combina
    geometría y estado local en una única matriz que el resto de funciones
    consume sin saber cómo se ha construido.

    Modelo de conductividad dependiente de la temperatura:

        sigma(T) = sigma_0 / (1 + alpha_T * (T - T_0))

    La celda es un prisma Delta_x_R x Delta_y_R x Delta_z_R, con Delta_x_R =
    Delta_y_R = atom_size (paso de red) siempre, así que su cociente vale 1 y
    desaparece de la fórmula; Delta_z_R = delta_z_r es un espesor INDEPENDIENTE
    del paso de red, no necesariamente igual a `atom_size`:

        R(T) = rho(T) * Delta_x_R / (Delta_y_R * Delta_z_R) = rho(T) / delta_z_r
             = (1 + alpha_T * (T - T_0)) / (sigma_0 * delta_z_r)

    En T = T_0 queda R = 1 / (sigma_0 * delta_z_r), la resistencia de celda de
    referencia. Con alpha_T = 0 el mapa es uniforme y reproduce el modelo de
    resistencia constante previo.

    Esta es la ÚNICA función del código que conoce el modelo sigma(T): cambiarlo
    (Arrhenius, dependencia con el campo, ...) es tocar solo aquí.

    Convenio de celdas sin filamento
    --------------------------------
    Las celdas de óxido reciben `np.inf`, es decir conductancia CERO. Así se
    excluyen SOLAS de la aritmética aguas abajo, sin necesidad de enmascarar:

      - Combinación en paralelo (columna):  1/inf = 0, no aporta conductancia.
      - Calor Joule:                        Q = delta_V^2 / (inf * atom_size**2 * delta_z_r) = 0.

    Además hace el mapa autodescriptivo: `np.isfinite(R_local)` devuelve la
    geometría del filamento.

    Args:
        cf_matrix (np.ndarray): Matriz (Ny, Nx) donde 1 indica celda de filamento.
            Marco INTERIOR, sin las columnas de electrodo.
        temperatura: Temperatura de cada celda [K]. Matriz de la misma forma que
            `cf_matrix` (típicamente `temperatura_anterior`, ya recortada) o escalar
            (primer paso percolante, cuando aún no hay mapa FVM).
        sigma_0 (float): Conductividad de referencia a T_0 [S/m].
        alpha_T (float): Coeficiente térmico de resistencia [1/K]. Con alpha_T > 0
            la resistencia sube al calentar (realimentación negativa).
        T_0 (float): Temperatura de referencia [K] (`params.init_temp`).
        delta_z_r (float): Espesor Delta_z_R de la celda en la dirección fuera de
            plano [m], factor geométrico sigma <-> R. Independiente del paso de
            red `atom_size` y del espesor que asuma el solver térmico.

    Returns:
        np.ndarray: Mapa de resistencias (Ny, Nx) [Ohm]. Finito en las celdas de
            filamento, `inf` en el resto.

    Nota de dominio: la expresión se indefine si alpha_T * (T - T_0) <= -1. En el
    régimen físico esperado (T >= T_0, alpha_T > 0) el denominador es siempre >= 1.
    """
    cf_matrix = np.asarray(cf_matrix)

    # R(T) = (1 + alpha_T * (T - T_0)) / (sigma_0 * delta_z_r). Se evalúa en toda
    # la matriz (coste despreciable) y np.where se queda solo con las celdas de filamento.
    R_celda = (1.0 + alpha_T * (np.asarray(temperatura, dtype=float) - T_0)) / (sigma_0 * delta_z_r)

    return np.where(cf_matrix == 1, R_celda, np.inf)


def resistencias_por_columna(R_local: np.ndarray) -> np.ndarray:
    """
    Calcula la resistencia equivalente de CADA columna a partir del mapa de
    resistencias por celda.

    Las celdas de una misma columna están en paralelo, así que se combinan sumando
    conductancias (media armónica):

        R_col_j = 1 / sum_i ( 1 / R_local[i, j] )

    Las celdas de óxido llevan `inf` (ver `mapa_resistencias`), y como 1/inf = 0 se
    excluyen SOLAS del sumatorio: no hace falta enmascarar la geometría.

    Esta es la FUENTE ÚNICA del modelo de resistencia. La consumen:
      - `calcular_resistencia`      (rama eléctrica): suma el vector -> R del bloque.
      - `calculate_heat_source`     (fuente de calor): necesita la caída de tensión
        por columna (delta_V_j = I_filamento * R_col_j) para repartir el calor.

    Devolver el vector en vez de solo la suma evita que ambas ramas implementen el
    mismo modelo por separado y se desincronicen.

    Args:
        R_local (np.ndarray): Mapa de resistencias por celda [Ohm] (o un bloque de
            filas del mismo), con `inf` donde no hay filamento.

    Returns:
        np.ndarray: Vector de longitud Nx con la resistencia de cada columna [Ohm].
            `inf` en las columnas sin ninguna celda de filamento (circuito abierto).
    """
    # Conductancia total de cada columna. 1/inf = 0 -> el óxido no aporta.
    conductancias = np.sum(1.0 / np.asarray(R_local, dtype=float), axis=0)

    # inf por defecto: solo se invierten las columnas con conductancia > 0.
    # Una columna sin filamento tiene conductancia 0 -> R_col = inf (circuito abierto).
    R_cols = np.full(conductancias.shape, np.inf, dtype=float)
    np.divide(1.0, conductancias, out=R_cols, where=conductancias > 0)

    return R_cols


def calcular_resistencia(R_local: np.ndarray, mostrar_calculo: bool = False) -> float:
    """
    Calcula la resistencia total de un bloque a partir del mapa de resistencias por
    celda: las columnas están en serie, así que se suman sus resistencias.

    Args:
        R_local (np.ndarray): Mapa de resistencias por celda [Ohm], `inf` fuera del filamento.
        mostrar_calculo (bool): Si True, registra el detalle columna a columna.

    Returns:
        float: Resistencia total del bloque [Ohm]. `inf` si alguna columna está vacía
            (un hueco en el filamento es un circuito abierto).
    """
    R_cols = resistencias_por_columna(R_local)

    if mostrar_calculo:
        acumulada = 0.0
        for j, R_col in enumerate(R_cols):
            if not np.isfinite(R_col):
                logger.info(f"La columna {j} no tiene filamento: circuito abierto")
                continue
            acumulada += R_col
            logger.info(
                f"La resistencia de la columna {j} es: {R_col:.4f} ohmios \n Resistencia total acumulada: {acumulada:.4f} ohmios"
            )

    # Columnas en serie. Si alguna es inf, el bloque entero es un circuito abierto.
    return float(np.sum(R_cols))


def Poole_Frenkel(
    temperature: float, E_field: float, pb_metal_insul: float, permitividad_relativa: float, I_0: float
) -> float:
    """
    Calcula la corriente mediante el mecanismo de conducción Poole-Frenkel.

    El efecto Poole-Frenkel describe la emisión de portadores de carga desde trampas
    en un material aislante bajo la influencia de un campo eléctrico alto. Este modelo
    es importante en la caracterización de materiales dieléctricos y semiconductores.

    Args:
        temperature (float): Temperatura absoluta en Kelvin [K]
        E_field (float): Campo eléctrico aplicado en V/m [V/m]
        pb_metal_insul (float): Barrera de potencial metal-aislante en eV [eV]
        permitividad_relativa (float): Permitividad relativa del material dieléctrico [adimensional]
        I_0 (float): Corriente de referencia o prefactor en Amperios [A]

    Returns:
        float: Corriente Poole-Frenkel calculada en Amperios [A]

    Fórmula:
        I_PF = I_0 * E * exp((β - φ_b) / (k_B * T))
        donde β = sqrt(e³ / (π * ε_0 * ε_r)) * sqrt(E)
    """

    k_b_ev = Boltzmann / elementary_charge
    beta = math.sqrt(abs(E_field)) * math.sqrt(elementary_charge / (epsilon_0 * permitividad_relativa * math.pi))

    exponencial = math.exp((beta - pb_metal_insul) / (k_b_ev * temperature))
    I_poole_frenkel = I_0 * E_field * exponencial

    return I_poole_frenkel


def calcular_resistencia_por_filamento(
    R_local: np.ndarray,
    CF_ranges: list[tuple[int, int]],
) -> list[float]:
    """
    Calcula la resistencia de cada filamento por separado, aplicando `calcular_resistencia`
    solo sobre el bloque de filas que le corresponde a cada uno según `CF_ranges`.

    Args:
        R_local (np.ndarray): Mapa de resistencias por celda [Ohm], `inf` fuera del filamento.
        CF_ranges (list): Bandas de filas (fila_min, fila_max) de cada filamento.

    Returns:
        list[float]: Resistencia de cada filamento [Ohm]. `inf` si no está formado.
    """
    resistencias = []
    for fila_min, fila_max in CF_ranges:
        sub = R_local[fila_min : fila_max + 1, :]
        # Guarda defensiva: una banda sin ninguna celda de filamento ya daría inf por
        # aritmética (todas las columnas a conductancia 0), pero se explicita.
        if not np.any(np.isfinite(sub)):
            resistencias.append(np.inf)
        else:
            resistencias.append(calcular_resistencia(sub))
    return resistencias


def calcular_resistencia_paralelo(resistencias_fils: list[float]) -> float:
    """
    Combina las resistencias de varios filamentos en paralelo (1/R_total = sum(1/R_i)).
    """
    conductancias = [1 / r for r in resistencias_fils if r != np.inf]
    if not conductancias:
        return np.inf
    return 1 / sum(conductancias)


def OhmCurrent_filamentos(
    potential: float,
    R_local: np.ndarray,
    CF_ranges: list[tuple[int, int]],
) -> tuple[float, float, list[float], list[float]]:
    """
    Calcula la corriente y resistencia total del dispositivo (filamentos en paralelo)
    junto con la corriente y resistencia de cada filamento individual.

    Args:
        potential (float): Tensión aplicada [V].
        R_local (np.ndarray): Mapa de resistencias por celda [Ohm] (ver `mapa_resistencias`).
        CF_ranges (list): Bandas de filas de cada filamento.

    Returns:
        (I_total, R_total, I_fils, R_fils). Un filamento no formado tiene R = inf e
        I = 0: no aporta conductancia, es una rama abierta.
    """
    R_fils = calcular_resistencia_por_filamento(R_local, CF_ranges)
    R_total = calcular_resistencia_paralelo(R_fils)
    I_fils = [potential / r if r != np.inf else 0.0 for r in R_fils]
    I_total = potential / R_total
    return I_total, R_total, I_fils, R_fils


def calculate_heat_source(
    atom_size: float,
    delta_z_r: float,
    R_local: np.ndarray,
    factor_generar_calor: float,
    CF_ranges: list,
    I_fils: list,
) -> np.ndarray:
    """
    Calcula el mapa de calor Q [W/m^3] resolviendo CADA FILAMENTO POR SEPARADO.

    Cada filamento ocupa una banda de filas (`CF_ranges`) y disipa el calor que le
    corresponde por SU propia corriente `I_fils[f]`, no por la corriente total del
    dispositivo. Para una columna 'j' del filamento 'f':

        R_col   = 1 / sum_i (1 / R_local[i, j])              <- celdas de la columna en paralelo
        delta_V = I_f * R_col                                <- caída de tensión en la columna
        Q[i, j] = delta_V^2 / (R_local[i, j] * atom_size**2 * delta_z_r)

    La última expresión es la potencia disipada por la celda (delta_V^2 / R) dividida
    por su volumen (atom_size**2 * delta_z_r), es decir densidad de potencia. Ese volumen
    es el de la MISMA celda Delta_x_R x Delta_y_R x Delta_z_R que usa `mapa_resistencias`
    para R_local (Delta_x_R = Delta_y_R = atom_size, Delta_z_R = delta_z_r): por eso esta
    función vive en la rama eléctrica y no en `thermal`, y `delta_z_r` es independiente
    del espesor que asuma `thermal.fvm.solve_thermal_state` (que no lo usa ni lo necesita:
    el balance del FVM es invariante frente a su propio espesor).
    Todas las celdas de una columna comparten `delta_V` (están en paralelo) pero pueden
    tener resistencias distintas, de modo que cada una disipa según la suya.

    La resistencia por columna se recalcula aquí con `resistencias_por_columna`, la misma
    función que usa la rama eléctrica para obtener `I_fils`. Ambas ramas comparten modelo
    de resistencia, por lo que no pueden desincronizarse.

    Las columnas de cada banda se procesan de una vez (vectorizado), con el mismo orden
    de operaciones que el cálculo columna a columna: el resultado es idéntico bit a bit.

    Args:
        atom_size (float): Tamaño de celda 'h' [m] (Delta_x_R = Delta_y_R).
        delta_z_r (float): Espesor Delta_z_R de la celda [m], el mismo que usa
            `mapa_resistencias` para calcular `R_local`.
        R_local (np.ndarray): Mapa de resistencias por celda (Ny, Nx_interior) [Ohm],
            con `inf` fuera del filamento (ver `mapa_resistencias`).
        factor_generar_calor (float): Factor de calibración del calor generado.
        CF_ranges (list): Bandas de filas (fila_min, fila_max) de cada filamento.
        I_fils (list): Corriente de cada filamento [A]. 0.0 / NaN si no está formado.

    Returns:
        np.ndarray: Mapa de calor Q [W/m^3] en marco EXTENDIDO (Ny, Nx_interior + 2),
            con las columnas de electrodo a cero, listo para `solve_thermal_state`.
    """
    R_local = np.asarray(R_local, dtype=float)
    Ny, Nx_int = R_local.shape

    # Marco extendido: se añaden las dos columnas de electrodo (a cero) para que la
    # forma coincida con types_map en solve_thermal_state.
    Q_map_global = np.zeros((Ny, Nx_int + 2))

    # Iteramos por filamento. CF_ranges particiona todas las filas sin huecos ni
    # solapes, así que cada celda de filamento recibe Q exactamente una vez.
    for f, (fila_min, fila_max) in enumerate(CF_ranges):
        I_f = I_fils[f]

        # Filamento no formado o sin corriente: no genera calor
        if not np.isfinite(I_f) or I_f == 0.0:
            continue

        # Bloque de resistencias de esta banda (marco interior)
        bloque_R = R_local[fila_min : fila_max + 1, :]
        R_cols = resistencias_por_columna(bloque_R)

        # Columnas sin filamento (circuito abierto, R_col = inf): no disipan. Se les
        # pone delta_V = 0 (y R_col = 0 antes de multiplicar, para no evaluar inf * I_f).
        con_filamento = np.isfinite(R_cols)

        # Caída de tensión en cada columna del filamento f
        delta_V_local = np.where(con_filamento, I_f * np.where(con_filamento, R_cols, 0.0), 0.0)

        # Q = delta_V^2 / (R_celda * atom_size**2 * delta_z_r). Las celdas de óxido
        # llevan R = inf, así que reciben Q = 0 automáticamente: no hace falta
        # enmascarar. El volumen atom_size**2 * delta_z_r es el de la celda de la
        # RAMA ELÉCTRICA (misma Delta_z_R que mapa_resistencias); no tiene relación
        # con el espesor que asuma solve_thermal_state, que es independiente.
        # np.power(x, 2.0) y no x**2: sobre un array, `**2` se atajó como x*x, que difiere
        # en el último bit de pow(x, 2.0), lo que hacía el cálculo escalar columna a columna.
        Q_bloque = np.power(delta_V_local, 2.0) / (bloque_R * atom_size**2 * delta_z_r) * factor_generar_calor

        # 1:-1 en columnas para saltar los electrodos del marco extendido
        Q_map_global[fila_min : fila_max + 1, 1:-1] = np.where(con_filamento, Q_bloque, 0.0)

    return Q_map_global


# Nombre antiguo (errata) de OhmCurrent_filamentos, aún usado por código externo
OmhCurrent_filamentos = OhmCurrent_filamentos
