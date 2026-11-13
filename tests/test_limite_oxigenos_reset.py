"""
Verificación del límite de iones de oxígeno simultáneos en el reset:
`Generation.contar_vacantes_generadas`, `Recombination.resolver_limite_oxigenos`,
`Recombination.cupo_oxigenos`, el cupo en `generate_oxygen` y la clave
`max_oxigenos_reset` de `SimulationConstants`.

Ejecutar desde la raíz del repo:
    python3 tests/test_limite_oxigenos_reset.py
"""

import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from RRAM import Generation, Recombination  # noqa: E402
from RRAM.constants_simulation import SimulationConstants  # noqa: E402
from RRAM.exceptions import LimiteOxigenosNoDefinidoException  # noqa: E402
from RRAM.simulation_config import DEFAULT_PARAMS  # noqa: E402

fallos = []


def check(nombre, cond, detalle=""):
    print(f"{'  OK  ' if cond else ' FALLO'} | {nombre}" + (f"  ({detalle})" if detalle else ""))
    if not cond:
        fallos.append(nombre)


class CapturaLog(logging.Handler):
    def __init__(self):
        super().__init__()
        self.registros = []

    def emit(self, record):
        self.registros.append(record)

    def niveles(self, nivel):
        return [r.getMessage() for r in self.registros if r.levelno == nivel]


captura = CapturaLog()
logging.getLogger("RRAM").addHandler(captura)
logging.getLogger("RRAM").setLevel(logging.DEBUG)

NY, NX = 40, 10

# --- 1. Vacantes generadas = final - inicial ---
ini = np.zeros((NY, NX), dtype=int)
ini[3, 3] = ini[4, 4] = 1
fin = ini.copy()
fin[10:13, 0] = 1
check("contar_vacantes_generadas: 5 - 2 = 3", Generation.contar_vacantes_generadas(ini, fin) == 3)
check("contar_vacantes_generadas: sin cambios = 0", Generation.contar_vacantes_generadas(ini, ini) == 0)

# --- 2. Resolver: solo el valor del set ---
captura.registros.clear()
check("resolver: solo set -> valor del set", Recombination.resolver_limite_oxigenos(120, None) == 120)
check("resolver: solo set -> sin aviso", not captura.niveles(logging.WARNING))

# --- 3. Resolver: solo el manual ---
captura.registros.clear()
check("resolver: solo manual -> manual", Recombination.resolver_limite_oxigenos(None, 50) == 50)
check("resolver: solo manual -> sin aviso", not captura.niveles(logging.WARNING))

# --- 4. Resolver: los dos -> manual y aviso en el log ---
captura.registros.clear()
check("resolver: los dos -> manual", Recombination.resolver_limite_oxigenos(120, 50) == 50)
avisos = captura.niveles(logging.WARNING)
check("resolver: los dos -> aviso de que se usa la configuración", any("archivo de configuración" in m for m in avisos), str(avisos))

# --- 5. Resolver: ninguno -> excepción y error en el log ---
captura.registros.clear()
try:
    Recombination.resolver_limite_oxigenos(None, None)
    check("resolver: ninguno -> excepción", False)
except LimiteOxigenosNoDefinidoException:
    check("resolver: ninguno -> excepción", True)
check("resolver: ninguno -> error en el log", bool(captura.niveles(logging.ERROR)))

# --- 6. Resolver: 0 es un valor válido (no se confunde con None) ---
check("resolver: manual 0 -> 0", Recombination.resolver_limite_oxigenos(120, 0) == 0)
check("resolver: set 0 -> 0", Recombination.resolver_limite_oxigenos(0, None) == 0)

# --- 7. Cupo ---
oxy = np.zeros((NY, NX), dtype=np.int8)
oxy[0:4, 2] = 1
check("cupo: 10 - 4 = 6", Recombination.cupo_oxigenos(oxy, 10) == 6)
check("cupo: nunca negativo", Recombination.cupo_oxigenos(oxy, 2) == 0)

# --- 8. generate_oxygen nunca supera el límite ---
banda = np.ones(NY, dtype=bool)
LIMITE = 12
np.random.seed(0)
oxy = np.zeros((NY, NX), dtype=np.int8)
maximo = 0
for _ in range(300):
    # se desplazan los iones de la interfaz a la columna 1 para que la interfaz no sea el tope
    oxy[:, 1] |= oxy[:, 0]
    oxy[:, 0] = 0
    cupo = Recombination.cupo_oxigenos(oxy, LIMITE)
    oxy = Recombination.generate_oxygen(oxy, 5, Recombination.filas_libres_interfaz(oxy, banda), cupo)
    maximo = max(maximo, int(oxy.sum()))
check("300 pasos: nunca se supera el límite", maximo <= LIMITE, f"máximo={maximo}")
check("300 pasos: se llega al límite", maximo == LIMITE, f"máximo={maximo}")

# --- 9. Con el límite alcanzado no se genera ninguno ---
antes = oxy.copy()
oxy = Recombination.generate_oxygen(oxy, 5, Recombination.filas_libres_interfaz(oxy, banda), Recombination.cupo_oxigenos(oxy, LIMITE))
check("límite alcanzado: no se genera ninguno", np.array_equal(oxy, antes))

# --- 10. Un ion que desaparece (recombinación o salida) libera cupo ---
filas_con_ion = np.where(oxy[:, 1] == 1)[0][:3]
oxy[filas_con_ion, 1] = 0  # desaparecen 3 iones
check("desaparecen 3 iones: cupo = 3", Recombination.cupo_oxigenos(oxy, LIMITE) == 3)
np.random.seed(5)
for _ in range(50):
    oxy = Recombination.generate_oxygen(oxy, 5, Recombination.filas_libres_interfaz(oxy, banda), Recombination.cupo_oxigenos(oxy, LIMITE))
check("tras liberar cupo se vuelve justo al límite", int(oxy.sum()) == LIMITE, f"iones={int(oxy.sum())}")

# --- 11. max_oxigenos_reset en la configuración y en SimulationConstants ---
check("DEFAULT_PARAMS: existe la clave max_oxigenos_reset", "max_oxigenos_reset" in DEFAULT_PARAMS)
base = {k: v for k, v in DEFAULT_PARAMS.items()}
base.update({"cte_red": 1.0, "permitividad_relativa_set": 1.0, "permitividad_relativa_reset": 1.0})
csv_vacio = SimulationConstants.from_dict({**{k: str(v) for k, v in base.items()}, "max_oxigenos_reset": ""})
check("CSV celda vacía -> None", csv_vacio.max_oxigenos_reset is None, repr(csv_vacio.max_oxigenos_reset))
csv_none = SimulationConstants.from_dict({**{k: str(v) for k, v in base.items()}, "max_oxigenos_reset": "None"})
check("CSV 'None' -> None", csv_none.max_oxigenos_reset is None, repr(csv_none.max_oxigenos_reset))
csv_valor = SimulationConstants.from_dict({**{k: str(v) for k, v in base.items()}, "max_oxigenos_reset": "300"})
check("CSV '300' -> 300 (int)", csv_valor.max_oxigenos_reset == 300 and isinstance(csv_valor.max_oxigenos_reset, int), repr(csv_valor.max_oxigenos_reset))
sin_columna = SimulationConstants.from_dict({k: str(v) for k, v in base.items() if k != "max_oxigenos_reset"})
check("CSV antiguo sin la columna -> None", sin_columna.max_oxigenos_reset is None)

print()
if fallos:
    print(f"{len(fallos)} FALLO(S): {fallos}")
    sys.exit(1)
print("Todos los checks OK")
