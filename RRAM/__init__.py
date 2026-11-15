"""
Simulador 2D de memorias resistivas (RRAM).

No se importa ningún submódulo al importar el paquete: cada uno carga sus dependencias cuando se
pide (`from RRAM.run_cycle import run_cycle`). El acceso por atributo (`import RRAM; RRAM.Representate`)
funciona: `__getattr__` importa el submódulo al vuelo.

Alias de módulos que ya no existen como archivo (los usan notebooks antiguos):
`Percolation` → `topologia_filamentos`, `CurrentSolver` → `conduccion`, `Temperature` → `thermal`.
"""

import importlib
import pkgutil

#: Nombre antiguo → módulo real.
_ALIAS = {"Percolation": "topologia_filamentos", "CurrentSolver": "conduccion", "Temperature": "thermal"}


def __getattr__(nombre):
    if nombre in _ALIAS:
        return importlib.import_module(f".{_ALIAS[nombre]}", __name__)
    if nombre in {m.name for m in pkgutil.iter_modules(__path__)}:
        return importlib.import_module(f".{nombre}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {nombre!r}")
