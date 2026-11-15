"""
Configuración centralizada del sistema de logging del paquete RRAM.

Cada módulo usa su logger jerárquico `RRAM.<modulo>` (`logging.getLogger(__name__)`);
`setup_logging` configura el logger raíz `RRAM` con un fichero por simulación.

Nivel en tiempo de ejecución, con una variable de entorno:
    RRAM_LOG_LEVEL=DEBUG     -> traza detallada
    RRAM_LOG_LEVEL=INFO      -> hitos de simulación (por defecto)
    RRAM_LOG_LEVEL=WARNING   -> solo advertencias y errores
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Union

# Logger raíz del paquete. Todo handler/configuración cuelga de aquí.
ROOT_LOGGER_NAME = "RRAM"

# Nivel personalizado para tracing de valores (entre DEBUG=10 e INFO=20)

_LEVEL_NAMES = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


def _resolve_level(level: Optional[Union[int, str]]) -> int:
    """Resuelve el nivel efectivo combinando argumento explícito y env var."""
    if level is not None:
        if isinstance(level, str):
            up = level.upper().strip()
            if up not in _LEVEL_NAMES:
                raise ValueError(f"Nivel de log no reconocido: {level!r}")
            return getattr(logging, up)
        return int(level)

    env_level = os.environ.get("RRAM_LOG_LEVEL", "INFO").upper().strip()
    if env_level not in _LEVEL_NAMES:
        env_level = "INFO"
    return getattr(logging, env_level)


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
            ``RRAM_LOG_LEVEL`` (default INFO). Soporta: DEBUG, INFO, WARNING, ERROR, CRITICAL.
        to_console: Añade también un StreamHandler hacia stderr.
        file_mode: Modo de apertura del archivo ('w' sobrescribe, 'a' añade).

    Returns:
        El logger raíz `RRAM` configurado.

    Ejemplos:
        setup_logging(num_simulation=0, level="DEBUG")  # Traza detallada
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
