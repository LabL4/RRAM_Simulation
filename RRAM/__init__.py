"""
Simulador 2D de memorias resistivas (RRAM).

No se importa ningún submódulo al importar el paquete: cada uno carga sus dependencias cuando se
pide (`from RRAM import Temperature`, `from RRAM.run_cycle import run_cycle`). El acceso por atributo
(`import RRAM; RRAM.Temperature`) sigue funcionando: `__getattr__` importa el submódulo al vuelo.
"""

import importlib
import pkgutil


def __getattr__(nombre):
    if nombre in {m.name for m in pkgutil.iter_modules(__path__)}:
        return importlib.import_module(f".{nombre}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {nombre!r}")
