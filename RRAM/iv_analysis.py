"""Postprocesado y representación de curvas I-V de la simulación RRAM."""

from pathlib import Path

import numpy as np

from . import Representate, utils


def simulation_IV(
    num_simulation: int,
    figures_path: Path,
    simulation_path: Path,
    desplazamiento: dict,
    voltaje_percolacion: float,
    roturas_dict: dict,
):
    # region Representar datos
    save_path = figures_path / f"I-V_{num_simulation}"
    save_path_marcado = figures_path / f"I-V_{num_simulation}_marcado"

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
                print(f"Advertencia: No se encontró el archivo {name}")
                # Podrías inicializar un array vacío o manejar el error según convenga
                data[key] = np.zeros((0, 3))

    # Unir las partes PP y SP para el SET
    # Nota: Ya que hemos extraído 'datos_sim', podemos acceder a las columnas directamente
    # Columna 1 = Voltaje, Columna 2 = Intensidad
    i_set = np.concatenate([abs(data["pp_set"][:, 2]), abs(data["sp_set"][:, 2])])
    v_set = np.concatenate([data["pp_set"][:, 1], data["sp_set"][:, 1]])

    # Unir las partes PP y SP para el RESET
    i_reset = np.concatenate([abs(data["pp_reset"][:, 2]), abs(data["sp_reset"][:, 2])])
    v_reset = np.concatenate([data["pp_reset"][:, 1], data["sp_reset"][:, 1]])

    puntos_x_set = {"a": 1e-9, "b": voltaje_percolacion, "c": 1.1}
    puntos_x_pp_reset = {"d": -0.44, "e": roturas_dict[0]["voltaje"], "f": -1.1}
    puntos_x_sp_reset = {"g": -2e-8}

    # Obtener puntos en cada curva (se elimina el llamado anidado a ["datos_sim"])
    puntos_set = utils.obtener_puntos_en_curva(data["pp_set"][:, 1], abs(data["pp_set"][:, 2]), puntos_x_set)

    puntos_x_pp_reset = utils.obtener_puntos_en_curva(
        data["pp_reset"][:, 1],
        abs(data["pp_reset"][:, 2]),
        puntos_x_pp_reset,
    )

    puntos_x_sp_reset = utils.obtener_puntos_en_curva(
        data["sp_reset"][:, 1],
        abs(data["sp_reset"][:, 2]),
        puntos_x_sp_reset,
    )

    print("Puntos en la curva I-V:\n")
    for label, (v, i) in {
        **puntos_set,
        **puntos_x_pp_reset,
        **puntos_x_sp_reset,
    }.items():
        print(f"  Punto {label}: V = {v:.6f} V, I = {i:.6e} A")

    # Crear un único diccionario combinando ambos
    puntos_totales = {}
    puntos_totales.update(puntos_set)
    puntos_totales.update(puntos_x_pp_reset)
    puntos_totales.update(puntos_x_sp_reset)

    Representate.plot_IV(
        v_set,
        i_set,
        v_reset,
        i_reset,
        num_simulation - 1,
        titulo_figura="",
        figures_path=str(save_path),
    )
    Representate.plot_IV_marcado(
        v_set,
        i_set,
        v_reset,
        i_reset,
        num_simulation - 1,
        puntos_totales,
        desplazamiento,
    )

    return None
