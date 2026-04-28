"""Decoradores auxiliares de la simulación RRAM."""

import time
from functools import wraps
import logging

logger = logging.getLogger(__name__)


def medir_tiempo(func):
    """Decora una función para imprimir su tiempo de ejecución."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        inicio = time.time()
        resultado = func(*args, **kwargs)
        fin = time.time()
        logger.info(f"Función '{func.__name__}' ejecutada en {fin - inicio:.6f} segundos")
        return resultado

    return wrapper
