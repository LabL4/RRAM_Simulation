"""
Configuración centralizada del sistema de logging del paquete RRAM.

Sustituye los `print(...)` dispersos por loggers jerárquicos `RRAM.<modulo>`
que pueden activarse/desactivarse por subsistema o por nivel.

Uso típico (desde RRAM_mod_simulation.py):

    from RRAM.logging_config import setup_logging, add_trace_method, get_logger
    setup_logging(num_simulation=num, level="TRACE")
    add_trace_method()  # Habilita .trace() en loggers

Control en runtime mediante variable de entorno:

    RRAM_LOG_LEVEL=TRACE     -> tracing completo de valores (valores intermedios, arrays)
    RRAM_LOG_LEVEL=DEBUG     -> traza detallada
    RRAM_LOG_LEVEL=INFO      -> hitos de simulación (default)
    RRAM_LOG_LEVEL=WARNING   -> solo advertencias y errores

Filtrar por subsistema desde código:

    from RRAM.logging_config import set_subsystem_level
    set_subsystem_level("phases_set", "TRACE")  # Tracing solo en phases_set
    set_subsystem_level("Generation", logging.WARNING)

Obtener logger con helpers para tracing de valores:

    from RRAM.logging_config import get_logger, trace_array, trace_scalar
    logger = get_logger("Generation")
    logger.trace(f"gamma={gamma}")  # Requiere add_trace_method()
    trace_array(logger, "vacantes", vacantes_grid)
    trace_scalar(logger, "R", R_current, context="fase SP_reset")
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Union
import numpy as np

# Logger raíz del paquete. Todo handler/configuración cuelga de aquí.
ROOT_LOGGER_NAME = "RRAM"

# Nivel personalizado para tracing de valores (entre DEBUG=10 e INFO=20)
TRACE = 5
logging.addLevelName(TRACE, "TRACE")

_LEVEL_NAMES = {"TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


def _resolve_level(level: Optional[Union[int, str]]) -> int:
    """Resuelve el nivel efectivo combinando argumento explícito y env var."""
    if level is not None:
        if isinstance(level, str):
            up = level.upper().strip()
            if up not in _LEVEL_NAMES:
                raise ValueError(f"Nivel de log no reconocido: {level!r}")
            return TRACE if up == "TRACE" else getattr(logging, up)
        return int(level)

    env_level = os.environ.get("RRAM_LOG_LEVEL", "INFO").upper().strip()
    if env_level not in _LEVEL_NAMES:
        env_level = "INFO"
    return TRACE if env_level == "TRACE" else getattr(logging, env_level)


def setup_logging(
    num_simulation: Optional[int] = None,
    log_dir: Union[str, Path] = "logs",
    level: Optional[Union[int, str]] = None,
    to_console: bool = False,
    file_mode: str = "w",
) -> logging.Logger:
    """
    Configura el logger raíz `RRAM` con un FileHandler por simulación.

    Args:
        num_simulation: Índice de la simulación. Si se proporciona, escribe en
            ``{log_dir}/log_simulacion_{num_simulation}.log``. Si es None, no
            crea FileHandler (útil para tests, notebooks de exploración).
        log_dir: Carpeta de destino para los logs. Se crea si no existe.
        level: Nivel del logger. Si es None, se lee de la env var
            ``RRAM_LOG_LEVEL`` (default INFO). Soporta: TRACE, DEBUG, INFO, WARNING, ERROR, CRITICAL.
        to_console: Añade también un StreamHandler hacia stderr.
        file_mode: Modo de apertura del archivo ('w' sobrescribe, 'a' añade).

    Returns:
        El logger raíz `RRAM` configurado.

    Ejemplos:
        setup_logging(num_simulation=0, level="TRACE")  # Tracing completo
        setup_logging(num_simulation=0, to_console=True)  # Logs a archivo + consola
    """
    effective_level = _resolve_level(level)

    root = logging.getLogger(ROOT_LOGGER_NAME)
    root.setLevel(effective_level)
    root.propagate = False  # Evita duplicados si el root logger global también tiene handlers

    # Limpiar handlers previos para que llamadas repetidas (ej. notebooks) no acumulen
    for h in list(root.handlers):
        root.removeHandler(h)
        h.close()

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # IMPORTANTE: los handlers se dejan en DEBUG para que NO filtren por sí mismos.
    # El filtrado real lo hace el logger (raíz `RRAM` o un subsistema concreto).
    # De este modo, `set_subsystem_level("phases_set", "DEBUG")` deja pasar el mensaje
    # hasta el archivo aunque el resto del paquete siga en INFO.
    if num_simulation is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            log_path / f"log_simulacion_{num_simulation}.log",
            mode=file_mode,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    if to_console:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.DEBUG)
        stream_handler.setFormatter(formatter)
        root.addHandler(stream_handler)

    return root


def set_subsystem_level(subsystem: str, level: Union[int, str]) -> None:
    """
    Cambia el nivel de un logger de subsistema.

    Ejemplos:
        set_subsystem_level("phases_set", "TRACE")  # Tracing completo
        set_subsystem_level("Generation", "DEBUG")
        set_subsystem_level("Temperature", logging.WARNING)
    """
    full_name = f"{ROOT_LOGGER_NAME}.{subsystem}" if not subsystem.startswith(
        ROOT_LOGGER_NAME
    ) else subsystem
    if isinstance(level, int):
        resolved = level
    else:
        up = level.upper()
        resolved = TRACE if up == "TRACE" else getattr(logging, up)
    logging.getLogger(full_name).setLevel(resolved)


# ============================================================================
# HELPERS PARA TRACING DE VALORES (nivel TRACE)
# ============================================================================

def add_trace_method() -> None:
    """
    Extiende logging.Logger con método `.trace()` para nivel personalizado TRACE.

    Llamar una sola vez al inicio de tu módulo/notebook para habilitar:
        logger.trace(f"vacantes={vacantes.sum()}, campo={E_field.max()}")

    Ejemplo:
        from RRAM.logging_config import add_trace_method, get_logger
        add_trace_method()
        logger = get_logger("Generation")
        logger.trace(f"gamma={gamma}")
    """
    def trace(self, message: str, *args, **kwargs):
        """Log un mensaje con nivel TRACE (5)."""
        if self.isEnabledFor(TRACE):
            self._log(TRACE, message, args, **kwargs)

    logging.Logger.trace = trace


def get_logger(name: str) -> logging.Logger:
    """
    Obtiene un logger RRAM jerárquico con método `.trace()` disponible (tras add_trace_method).

    Args:
        name: Nombre del subsistema, ej. "Generation", "phases_set", "CurrentSolver", "Temperature".
              Puede incluir o no el prefijo "RRAM.".

    Returns:
        Logger con namespace "RRAM.{name}" listo para .trace(), .debug(), .info(), etc.

    Ejemplo:
        logger = get_logger("Generation")
        logger.trace(f"gamma={gamma}")
        logger.debug(f"Vacantes generadas: {n}")
        logger.info(f"Fase completada")
    """
    full_name = f"{ROOT_LOGGER_NAME}.{name}" if not name.startswith(ROOT_LOGGER_NAME) else name
    return logging.getLogger(full_name)


def trace_array(logger: logging.Logger, name: str, arr: np.ndarray,
                stats: bool = True, precision: int = 4) -> None:
    """
    Registra un array NumPy con estadísticas en nivel TRACE.

    No hace nada si el logger no está habilitado para TRACE (costo cero si desactivado).

    Args:
        logger: Logger a usar (ej. obtenido con get_logger)
        name: Nombre de la variable
        arr: Array NumPy a loguear
        stats: Si True, muestra min/max/mean. Si False, solo shape y dtype.
        precision: Precisión decimal para estadísticas (ej. 4 -> "0.1234").

    Ejemplo:
        logger = get_logger("Percolation")
        trace_array(logger, "filament_map", filament_map, stats=True)
        # Output: [TRACE] filament_map: shape=(100, 100), dtype=int32,
        #         min=0, max=1, mean=0.4521
    """
    if not logger.isEnabledFor(TRACE):
        return

    msg = f"{name}: shape={arr.shape}, dtype={arr.dtype}"
    if stats:
        msg += (f", min={arr.min():.{precision}g}, max={arr.max():.{precision}g}, "
                f"mean={arr.mean():.{precision}g}")

    logger.trace(msg)


def trace_scalar(logger: logging.Logger, name: str, value: float | int,
                 context: str = "") -> None:
    """
    Registra un escalar en nivel TRACE.

    No hace nada si el logger no está habilitado para TRACE (costo cero si desactivado).

    Args:
        logger: Logger a usar
        name: Nombre de la variable
        value: Valor escalar
        context: Contexto adicional (ej. "paso 50", "V=2.5V", "fase SP_reset")

    Ejemplo:
        logger = get_logger("Temperature")
        trace_scalar(logger, "T", 350.5, context="fase SP_reset")
        # Output: [TRACE] T = 350.5 (fase SP_reset)
    """
    if not logger.isEnabledFor(TRACE):
        return

    msg = f"{name} = {value}"
    if context:
        msg += f" ({context})"
    logger.trace(msg)
