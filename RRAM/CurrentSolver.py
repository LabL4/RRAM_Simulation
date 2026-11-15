"""Fachada de compatibilidad: la conducción vive en `RRAM.conduccion`."""

from .conduccion import (mapa_resistencias, resistencias_por_columna, calcular_resistencia, Poole_Frenkel, calcular_resistencia_por_filamento, calcular_resistencia_paralelo, OhmCurrent_filamentos)  # noqa: F401

# Nombre antiguo (errata) de OhmCurrent_filamentos
OmhCurrent_filamentos = OhmCurrent_filamentos
