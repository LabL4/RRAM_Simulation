"""Fachada de compatibilidad: las utilidades viven en `semillas`, `geometria_filamentos`, `rutas` e `io_estado`.
"""

from .semillas import crear_rng, semilla_de_fase  # noqa: F401
from .geometria_filamentos import acumular_T_max_filamentos, generar_configuracion_filamentos, obtener_puntos_en_curva  # noqa: F401
from .rutas import crear_rutas_simulacion  # noqa: F401
from .io_estado import cargar_estado, guardar_datos, guardar_estado_intermedio, read_csv_to_dic  # noqa: F401
