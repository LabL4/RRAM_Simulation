"""Semillas y generadores aleatorios por fase."""

import numpy as np
from .voltage_controller import FASES


def semilla_de_fase(seed: int, fase: str) -> int:
    """
    Semilla propia de cada fase, derivada de la semilla de la corrida.

    Con la misma semilla en las cuatro fases todas consumían la misma secuencia U(0,1) y sus
    sorteos quedaban correlacionados. `SeedSequence.spawn` da una semilla estadísticamente
    independiente por fase y sigue siendo determinista: una fase arrancada con `--start-from`
    recibe exactamente la misma que en el ciclo completo.
    """
    hijos = np.random.SeedSequence(seed).spawn(len(FASES))
    return int(hijos[FASES.index(fase)].generate_state(1)[0])


def crear_rng(params, fase: str) -> np.random.RandomState:
    """
    Generador de números aleatorios propio de una fase.

    Con `params.seed` fijado, la secuencia es determinista y distinta en cada fase (ver
    `semilla_de_fase`), también con --start-from/--stop-at. Sin semilla es aleatorio real.
    Es un `RandomState`, el generador legado que usaba `np.random` de forma global, así que una
    semilla da la misma secuencia que antes; ya no se toca el estado global de numpy.
    """
    if params.seed is None:
        return np.random.RandomState()
    return np.random.RandomState(semilla_de_fase(params.seed, fase))
