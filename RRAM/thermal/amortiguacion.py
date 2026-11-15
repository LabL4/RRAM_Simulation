"""Amortiguación de caídas bruscas de temperatura y perfil de temperatura de los electrodos."""

import numpy as np


def amortiguacion_activa(k) -> bool:
    """True si la constante de amortiguación `k` es un número finito > 0 (None/NaN/<=0 = desactivada)."""
    try:
        return bool(np.isfinite(k) and k > 0)
    except TypeError:
        return False


def amortiguar_enfriamiento(T_calc, T_prev, T_0: float, k, dt: float, umbral: float):
    """
    Frena las caídas bruscas de temperatura (p. ej. V=0 entre pulsos).

    Donde la temperatura calculada CAE respecto a la del paso anterior más que `umbral`:
        T = max(T_calc, T_0 + (T_prev - T_0) * exp(-k * dt))
    En cualquier otro caso (calentamiento, caída pequeña) T = T_calc.

    Args:
        T_calc: temperatura recién calculada (escalar o matriz).
        T_prev: temperatura del paso anterior YA amortiguada (escalar o matriz, mismo
            tamaño que T_calc o escalar; se hace broadcast).
        T_0: temperatura ambiente [K].
        k: constante de amortiguación [1/s]. None/NaN/<=0 desactiva la función.
        dt: paso temporal [s].
        umbral: caída mínima [K] para activar la amortiguación (> 0).

    Returns:
        float si T_calc y T_prev son escalares; si no, un array NUEVO (las entradas
        no se modifican). Con la función desactivada devuelve T_calc tal cual.
    """
    if not amortiguacion_activa(k):
        return T_calc
    if umbral is None or not umbral > 0:
        raise ValueError(f"umbral_caida_temperatura debe ser > 0, recibido {umbral!r}")

    factor = float(np.exp(-k * dt))

    if np.ndim(T_calc) == 0 and np.ndim(T_prev) == 0:
        T_calc_f, T_prev_f = float(T_calc), float(T_prev)
        if T_calc_f < T_prev_f and (T_prev_f - T_calc_f) > umbral:
            return max(T_calc_f, T_0 + (T_prev_f - T_0) * factor)
        return T_calc_f

    T_calc_a = np.asarray(T_calc, dtype=float)
    T_prev_a = np.asarray(T_prev, dtype=float)
    T_amortiguada = T_0 + (T_prev_a - T_0) * factor
    mascara = (T_calc_a < T_prev_a) & ((T_prev_a - T_calc_a) > umbral)
    return np.where(mascara, np.maximum(T_calc_a, T_amortiguada), T_calc_a)


def amortiguar_mapa_fvm(T_fvm: np.ndarray, T_prev: np.ndarray, T_0: float, k, dt: float, umbral: float) -> np.ndarray:
    """
    Aplica `amortiguar_enfriamiento` al mapa FVM (Ny, Nx+2) sin tocar las columnas de
    electrodo (0 y -1, condición de contorno). `T_prev` es (Ny, Nx), sin electrodos.
    Devuelve una copia; con la amortiguación desactivada devuelve `T_fvm` tal cual.
    """
    if not amortiguacion_activa(k):
        return T_fvm
    T_out = np.array(T_fvm, dtype=float, copy=True)
    T_out[:, 1:-1] = amortiguar_enfriamiento(T_fvm[:, 1:-1], T_prev, T_0, k, dt, umbral)
    return T_out


def actualizar_temperatura_electrodo(T_fvm_sin_amortiguar: np.ndarray, T_e_prev, filas: np.ndarray, T_0: float, k, dt: float, umbral: float):
    """
    Perfil de temperatura de los electrodos para el paso siguiente.

    1. Perfil de cada lado: columna adyacente al electrodo (1 y -2) del mapa SIN amortiguar
       en las filas de los filamentos (`filas`) y T_0 en el resto.
    2. Se elige el perfil con menor temperatura máxima; ambos electrodos comparten ese perfil.
    3. Se compara con el perfil anterior: las celdas que caen más que `umbral` se enfrían
       como T_0 + (T_prev - T_0) * exp(-k * dt) (amortiguar_enfriamiento con constantes propias).

    Args:
        T_fvm_sin_amortiguar: mapa del solver (Ny, Nx+2), antes de amortiguar el interior.
        T_e_prev: electrodo usado para resolver este mapa (escalar o perfil (Ny,)) [K].
        filas: máscara (Ny,) de filas de los filamentos (ver filas_filamento).
        T_0: temperatura ambiente [K].
        k: constante de enfriamiento del electrodo [1/s]. None/NaN/<=0 desactiva: electrodo fijo.
        dt: paso temporal [s].
        umbral: caída mínima [K] para enfriar una celda (> 0).

    Returns:
        Perfil (Ny,) del electrodo; con k desactivada, T_e_prev tal cual.
    """
    if not amortiguacion_activa(k):
        return T_e_prev
    perfil_izq = np.where(filas, T_fvm_sin_amortiguar[:, 1], T_0)
    perfil_der = np.where(filas, T_fvm_sin_amortiguar[:, -2], T_0)
    perfil = perfil_izq if perfil_izq.max() <= perfil_der.max() else perfil_der
    return amortiguar_enfriamiento(perfil, T_e_prev, T_0, k, dt, umbral)


def temperatura_electrodo_joule(T_joule: float, T_e_prev, k):
    """
    Temperatura de los electrodos cuando no hay mapa FVM (modelo Joule escalar): el
    electrodo es un escalar que sigue a la temperatura del dispositivo.
    Con k desactivada el electrodo se queda en T_e_prev.
    """
    if not amortiguacion_activa(k):
        return T_e_prev
    return float(T_joule)


def como_mapa(T, forma) -> np.ndarray:
    """Convierte una temperatura escalar en un mapa de forma `forma` (float: sin truncar a entero)."""
    return np.full(forma, T, dtype=float)


def extra_sin_amortiguar(sim_ctes, temperatura_sin_amortiguar) -> dict:
    """Kwargs extra para guardar_estado_intermedio: el mapa de T sin amortiguar, solo si la amortiguación está activa."""
    if amortiguacion_activa(sim_ctes.k_amortiguacion_termica) and temperatura_sin_amortiguar is not None:
        return {"temperatura_sin_amortiguar": temperatura_sin_amortiguar}
    return {}
