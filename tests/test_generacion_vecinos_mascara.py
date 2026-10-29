"""
Verificación de `Generation.get_generation_probabilities_matrix`: lógica de vecinos
combinada con la máscara de grosor (`custom_mask`) y `neighbor_mode`.

Ejecutar desde la raíz del repo:
    python3 tests/test_generacion_vecinos_mascara.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from RRAM import Generation  # noqa: E402
from RRAM.constants_simulation import SimulationConstants  # noqa: E402

fallos = []


def check(nombre, cond, detalle=""):
    print(f"{'  OK  ' if cond else ' FALLO'} | {nombre}" + (f"  ({detalle})" if detalle else ""))
    if not cond:
        fallos.append(nombre)


KW = dict(
    paso_temporal=1e-3,
    Electric_field=1e8,
    temperatura=300.0,
    vibration_frequency=1e13,
    generation_energy=1.04,
    cte_red=2.5e-10,
    gamma=10.0,
    num_iteracion=None,
)


def prob(state, fv, fl, mode="both", mask=None):
    return Generation.get_generation_probabilities_matrix(
        state=state, factor_vecinos=fv, factor_sin_vecinos=fl, neighbor_mode=mode, custom_mask=mask, **KW
    )


# Matriz con una vacante aislada en el centro y otra en una esquina
state = np.zeros((9, 7), dtype=int)
state[4, 3] = 1
state[0, 0] = 1
mask = Generation.create_custom_mask(state=state, centros_CF=[4], grosor_CF=1)  # filas 3-5

# probabilidad base: sin vecinos y factores 1 → es la misma para todas las celdas libres
base = prob(state, 1.0, 1.0)[8, 6]
check("probabilidad base > 0 (la prueba es informativa)", 0 < base < 1, f"{base:.3e}")

FV, FL = 5.0, 0.1
p = prob(state, FV, FL, "both")
check("both: celda con vecino arriba (3,3) = base*FV", np.isclose(p[3, 3], base * FV))
check("both: celda con vecino a la izquierda (4,4) = base*FV", np.isclose(p[4, 4], base * FV))
check("both: celda sin vecinos (8,6) = base*FL", np.isclose(p[8, 6], base * FL))
check("celda ocupada = 0", p[4, 3] == 0.0 and p[0, 0] == 0.0)

# modos
ph = prob(state, FV, FL, "horizontal")
pv = prob(state, FV, FL, "vertical")
check("horizontal: vecino izquierda cuenta", np.isclose(ph[4, 4], base * FV))
check("horizontal: vecino arriba NO cuenta", np.isclose(ph[3, 3], base * FL))
check("vertical: vecino arriba cuenta", np.isclose(pv[3, 3], base * FV))
check("vertical: vecino izquierda NO cuenta", np.isclose(pv[4, 4], base * FL))
check("los tres modos dan mapas distintos", not np.array_equal(ph, pv) and not np.array_equal(ph, p))

# máscara: se aplica después de los vecinos
pm = prob(state, FV, FL, "both", mask)
check("con máscara: fuera de la banda la probabilidad es 0", np.all(pm[~mask] == 0.0))
check("con máscara: dentro, vecino = base*FV", np.isclose(pm[3, 3], base * FV))
check("con máscara: dentro, sin vecino = base*FL", np.isclose(pm[3, 0], base * FL))
check("con máscara: coincide con (sin máscara * máscara)", np.allclose(pm, p * mask))
check("máscara con factores 1/1 = solo máscara (comportamiento previo de PP_set)",
      np.allclose(prob(state, 1.0, 1.0, "both", mask), (state == 0) * mask * base))

# sin máscara: igual que antes
check("sin máscara: celdas con vecino y sin vecino se distinguen", p[3, 3] != p[8, 6])

# tope en 1
check("tope en 1", np.max(prob(state, 1e12, 1e12)) <= 1.0)

# modo inválido
try:
    prob(state, 1.0, 1.0, "diagonal")
    check("modo inválido lanza ValueError", False)
except ValueError:
    check("modo inválido lanza ValueError", True)

# constantes: el modo viaja como texto y los antiguos sin la clave cargan con "both"
c = SimulationConstants.from_dict({**{k: 1.0 for k in SimulationConstants.__dataclass_fields__ if k not in ("num_filamentos", "grosor_filamento", "neighbor_mode_pp_set", "neighbor_mode_sp_set", "centros_filamento", "grosor_filamento_init", "k_amortiguacion_termica", "umbral_caida_temperatura", "k_electrodo_enfriamiento", "pasos_guardar_estado", "num_oxigenos_pp_reset_1", "num_oxigenos_pp_reset_2", "num_oxigenos_sp_reset")}, "num_filamentos": 1, "grosor_filamento": [3], "pasos_guardar_estado": 10, "num_oxigenos_pp_reset_1": 1, "num_oxigenos_pp_reset_2": 1, "num_oxigenos_sp_reset": 1})
check("constantes sin la clave: neighbor_mode = both", c.neighbor_mode_pp_set == "both" and c.neighbor_mode_sp_set == "both")
c2 = SimulationConstants.from_dict({**c.__dict__, "neighbor_mode_pp_set": "horizontal", "neighbor_mode_sp_set": "vertical"})
check("constantes: el modo se conserva como texto", (c2.neighbor_mode_pp_set, c2.neighbor_mode_sp_set) == ("horizontal", "vertical"))

print()
if fallos:
    print(f"{len(fallos)} FALLO(S): {fallos}")
    sys.exit(1)
print("Todo OK")
