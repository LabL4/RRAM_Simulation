"""Funciones de seguimiento y actualización de filamentos conductivos (CFs)."""

from typing import Any, List

import numpy as np

from . import Representate, Temperature
from .constants_simulation import SimulationConstants


def procesar_filamentos_creados(
    imagen_path,
    data_save_path,
    existentes,
    CF_creado,
    voltage,
    voltage_CF_creado,
    actual_state,
    num_simulation,
    plot_filamento: bool = False,
):
    """
    Detecta filamentos nuevos, actualiza su estado y guarda imágenes e
    imágenes correspondientes, además guarda el estado actual en PKL.

    Args:
        filamentos (list): Lista de filamentos en el estado actual.
        CF_creado (np.ndarray): Vector booleano que indica si cada filamento fue creado.
        voltage (float): Voltaje actual.
        voltage_CF_creado (np.ndarray): Array para registrar voltajes de creación.
        actual_state (np.ndarray): Estado actual del sistema.
        num_simulation (int): Número de simulación.

    Returns:
        int: Índice actualizado para el filamento.
    """

    filamentos_nuevos = [i for i, v in enumerate(existentes) if v and not CF_creado[i]]

    for i in filamentos_nuevos:
        CF_creado[i] = True
        voltage_CF_creado[i] = voltage
        print(f"El filamento {i + 1} se ha creado en el voltaje {round(voltage, 4)} (V)")

        if plot_filamento:
            nombre_img = imagen_path / f"Filamento_{i + 1}_creado_set_{num_simulation}.png"
            Representate.RepresentateState(actual_state, round(voltage, 5), str(nombre_img))

        # Guardar estado actual en archivo pkl
        data_name = data_save_path / f"filamento_{i + 1}_creado_set_{num_simulation}.npz"
        np.savez_compressed(data_name, actual_state=actual_state)

    return None


def procesar_filamentos_destruidos(
    imagen_path,
    data_save_path,
    existentes,
    CF_destruido,
    voltage,
    voltage_CF_destruido,
    actual_state,
    num_simulation,
    roturas_dict,
    etapa,
    plot_filamento: bool = False,
):
    """
    Detecta filamentos rotos, actualiza su estado y guarda imágenes e
    imágenes correspondientes, además guarda el estado actual en PKL.

    Args:
        existentes (list[bool]): Lista booleana indicando existencia actual de filamentos.
        CF_destruido (np.ndarray): Vector booleano que indica si cada filamento fue destruido.
        voltage (float): Voltaje actual.
        voltage_CF_destruido (np.ndarray): Array para registrar voltajes de destrucción.
        actual_state (np.ndarray): Estado actual del sistema.
        num_simulation (int): Número de simulación.
        imagen_path (Path): Ruta donde guardar imágenes.
        pkl_path (Path): Ruta donde guardar archivos PKL.

    Returns:
        None
    """
    filamentos_rotos = [i for i, v in enumerate(existentes) if not v and not CF_destruido[i]]

    for i in filamentos_rotos:
        CF_destruido[i] = True
        if voltage_CF_destruido[i] == 0:
            voltage_CF_destruido[i] = voltage
            j = len(roturas_dict)  # obtiene el siguiente índice disponible
            roturas_dict[j] = {
                "filamento": i + 1,
                "voltaje": voltage,
                "etapa": etapa,
            }
            print(f"\nEl filamento {i + 1} se ha roto en el voltaje {round(voltage, 4)} (V)")

        if plot_filamento:
            nombre_img = imagen_path / f"Filamento_{i + 1}_roto_reset_{num_simulation}.png"
            Representate.RepresentateState(actual_state, round(voltage, 5), str(nombre_img))

        data_name = data_save_path / f"filamento_{i + 1}_roto_reset_{num_simulation}.npz"
        np.savez_compressed(data_name, actual_state=actual_state)

    return None


def actualizar_parametros_por_filamento(
    filamentos_actuales: int,
    filamentos_previos: int,
    CF_ranges: List[tuple],
    CF_centros: List[int],
    sim_ctes: SimulationConstants,
    all_CFs_created: bool,
) -> tuple[SimulationConstants, bool, Any, Any]:
    """
    Compacta la lógica de actualización de constantes cuando se detectan nuevos filamentos.
    Retorna (sim_ctes_actualizado, all_CFs_created, filas_intermedias, dist_casillas)
    """
    filas_intermedias = None
    dist_casillas = None
    num_esperados = len(CF_ranges)

    # 1. Caso para un solo filamento
    if num_esperados == 1:
        if filamentos_actuales == 1:
            print("Todos los filamentos creados.")
            sim_ctes = sim_ctes.update_gamma(sim_ctes.gamma / 5)
            sim_ctes = sim_ctes.update_generation_energy(1.75)
            all_CFs_created = True

    # 2. Caso para dos filamentos
    elif num_esperados == 2:
        if filamentos_actuales == 1:
            print("Se ha formado el primer filamento de dos.")
            sim_ctes = sim_ctes.update_gamma(sim_ctes.gamma / 5)

        elif filamentos_actuales == 2:
            print("Se ha formado el segundo filamento de dos.")
            all_CFs_created = True
            filas_intermedias, dist_casillas = Temperature.calcular_filas_intermedias(CF_centros)

    # 3. Caso general (3 o más)
    else:
        nuevos_formados = filamentos_actuales - filamentos_previos
        for _ in range(nuevos_formados):
            sim_ctes = sim_ctes.update_gamma(sim_ctes.gamma - 1)

        if filamentos_actuales == num_esperados:
            print("Todos los filamentos creados.")
            all_CFs_created = True

    # Logs comunes
    print(f"Filamentos: {filamentos_actuales}/{num_esperados}. Gamma: {sim_ctes.gamma}")
    if all_CFs_created:
        print(f"Todos los filamentos esperados se han creado: {all_CFs_created}")

    return sim_ctes, all_CFs_created, filas_intermedias, dist_casillas
