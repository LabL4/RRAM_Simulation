"""Postprocesado y representación de curvas I-V de la simulación RRAM."""

from pathlib import Path

import numpy as np

from . import Representate, utils
import logging

logger = logging.getLogger(__name__)

#: Por debajo de este valor absoluto de intensidad, el punto se descarta antes
#: de representar/marcar: es ruido de fondo (offset del solver, filamento aún
#: sin percolar) y distorsiona la escala logarítmica del eje Y.
INTENSIDAD_MINIMA_DEFAULT = 1e-7


def simulation_IV(
    num_simulation: int,
    figures_path: Path,
    simulation_path: Path,
    desplazamiento: dict,
    voltaje_percolacion: float,
    roturas_dict: dict,
    marcado: bool = False,
    intensidad_minima: float = INTENSIDAD_MINIMA_DEFAULT,
):
    """
    Genera UNA figura: la curva I-V sin marcar (``marcado=False``, default) o
    la curva con los puntos a-g marcados (``marcado=True``). Antes esta
    función generaba ambas figuras en la misma llamada; ahora cada subcomando
    de la CLI (`plot` / `plot_marcado`) pide explícitamente la que necesita.

    Args:
        marcado: Si True, dibuja `I-V_marcado_{N}` (curva + puntos a-g). Si
            False, dibuja solo `I-V_{N}` (curva sin marcar).
        intensidad_minima: Umbral absoluto de intensidad (A). Cualquier punto
            con |I| por debajo de este valor se descarta de TODAS las fases
            antes de unir curvas y buscar los puntos marcados, para no
            representar ruido de fondo cerca de I=0 en la escala log.
    """
    # region Representar datos
    # Los nombres de fichero (I-V_{N}, I-V_marcado_{N}) los construye cada
    # función de plot a partir de la CARPETA figures_path que le pasamos.
    # Definir nombres base y tipos
    prefixes = ["pp", "sp"]
    stages = ["set", "reset"]

    # Diccionario para guardar los datos cargados en memoria
    data = {}

    # Cargar archivos de forma automatizada
    for prefix in prefixes:
        for stage in stages:
            # ACTUALIZACIÓN 1: Nombre de archivo ajustado a tu nuevo formato
            name = f"Data_{prefix}_{stage}_{num_simulation}.npz"
            key = f"{prefix}_{stage}"

            try:
                # Cargamos el archivo .npz
                archivo_npz = np.load(simulation_path / name)

                # ACTUALIZACIÓN 2: Extraemos solo la matriz "datos_sim" a la memoria
                # Si en el futuro guardas vectores sueltos (ej: voltaje=v), aquí usarías archivo_npz["voltaje"]
                data[key] = archivo_npz["datos_sim"]

                # Cerramos el archivo npz (buena práctica de manejo de I/O)
                archivo_npz.close()
            except FileNotFoundError:
                logger.info(f"Advertencia: No se encontró el archivo {name}")
                # Podrías inicializar un array vacío o manejar el error según convenga
                data[key] = np.zeros((0, 3))

    # Filtrado de ruido: descarta filas con |I| < intensidad_minima ANTES de
    # unir curvas y de buscar los puntos marcados, para que ni la curva ni los
    # marcadores puedan caer en esa zona de ruido.
    for key, arr in data.items():
        if arr.shape[0] == 0:
            continue
        mask = np.abs(arr[:, 2]) >= intensidad_minima
        n_descartados = arr.shape[0] - int(mask.sum())
        if n_descartados > 0:
            logger.info(
                f"{key}: {n_descartados}/{arr.shape[0]} puntos descartados por |I| < {intensidad_minima:.1e} A."
            )
        data[key] = arr[mask]

    # Unir las partes PP y SP para el SET
    # Nota: Ya que hemos extraído 'datos_sim', podemos acceder a las columnas directamente
    # Columna 1 = Voltaje, Columna 2 = Intensidad
    i_set = np.concatenate([abs(data["pp_set"][:, 2]), abs(data["sp_set"][:, 2])])
    v_set = np.concatenate([data["pp_set"][:, 1], data["sp_set"][:, 1]])

    # Unir las partes PP y SP para el RESET
    i_reset = np.concatenate([abs(data["pp_reset"][:, 2]), abs(data["sp_reset"][:, 2])])
    v_reset = np.concatenate([data["pp_reset"][:, 1], data["sp_reset"][:, 1]])

    if not marcado:
        # `plot_IV` acepta arrays vacíos para las fases que falten y
        # simplemente no dibuja esa rama.
        Representate.plot_IV(
            v_set,
            i_set,
            v_reset,
            i_reset,
            num_simulation - 1,
            titulo_figura="",
            figures_path=str(figures_path),
        )
        return None

    # ----- marcado=True: solo la curva con puntos a-g -----
    # Los puntos marcados solo se calculan sobre fases con datos reales: si la
    # simulación se quedó a medias (p.ej. no llegó a RESET), `data[fase]` es el
    # array vacío inicializado más arriba y no hay curva sobre la que buscar el
    # punto más cercano. Cada bloque se salta de forma independiente para que
    # el resto de puntos disponibles se sigan marcando.
    puntos_totales: dict = {}

    if data["pp_set"].shape[0] > 0:
        puntos_x_set = {"a": 1e-7, "b": voltaje_percolacion, "c": 1.1}
        puntos_totales.update(
            utils.obtener_puntos_en_curva(data["pp_set"][:, 1], abs(data["pp_set"][:, 2]), puntos_x_set)
        )
    else:
        logger.info("Sin datos de pp_set; se omiten los puntos marcados a,b,c del SET.")

    if data["pp_reset"].shape[0] > 0:
        puntos_x_pp_reset = {"d": -0.44, "f": -1.1}
        rotura_0 = roturas_dict.get(0)
        if rotura_0 is not None:
            puntos_x_pp_reset["e"] = rotura_0["voltaje"]
        else:
            logger.info("Sin rotura 0 registrada; se omite el punto marcado 'e' del RESET.")
        puntos_totales.update(
            utils.obtener_puntos_en_curva(data["pp_reset"][:, 1], abs(data["pp_reset"][:, 2]), puntos_x_pp_reset)
        )
    else:
        logger.info("Sin datos de pp_reset; se omiten los puntos marcados d,e,f del RESET.")

    if data["sp_reset"].shape[0] > 0:
        puntos_x_sp_reset = {"g": -2e-7}
        puntos_totales.update(
            utils.obtener_puntos_en_curva(data["sp_reset"][:, 1], abs(data["sp_reset"][:, 2]), puntos_x_sp_reset)
        )
    else:
        logger.info("Sin datos de sp_reset; se omite el punto marcado 'g'.")

    logger.info("Puntos en la curva I-V:")
    for label, (v, i) in puntos_totales.items():
        logger.info(f"  Punto {label}: V = {v:.6f} V, I = {i:.6e} A")

    if not puntos_totales:
        logger.warning(
            f"Sim {num_simulation}: no hay ningún punto marcado calculable (sin datos de "
            f"pp_set/pp_reset/sp_reset). Se omite I-V_marcado."
        )
        return None

    Representate.plot_IV_marcado(
        v_set,
        i_set,
        v_reset,
        i_reset,
        num_simulation - 1,
        puntos_totales,
        desplazamiento,
        figures_path=str(figures_path),
    )

    return None
