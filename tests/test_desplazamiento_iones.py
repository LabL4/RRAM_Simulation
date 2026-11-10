"""
Verificación del tiempo acumulado en el movimiento de iones de oxígeno:
`Recombination.desplazamiento_maximo`, `move_oxygen_ions` y el bucle reloj + reset al mover.

Con v = 5.2e-7 m/s, dt = 1e-6 s y celda = 0.25e-9 m, round(v·n·dt/celda) >= 1 desde n = 241.

Ejecutar desde la raíz del repo:
    python3 tests/test_desplazamiento_iones.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from RRAM import Recombination  # noqa: E402
from RRAM.acumulador_tiempo import AcumuladorTiempo  # noqa: E402

fallos = []


def check(nombre, cond, detalle=""):
    print(f"{'  OK  ' if cond else ' FALLO'} | {nombre}" + (f"  ({detalle})" if detalle else ""))
    if not cond:
        fallos.append(nombre)


DT, CELDA, V_UMBRAL = 1e-6, 0.25e-9, 5.2e-7
THRESHOLDS = {1.15: V_UMBRAL}
FILAS, COLS, FILA_ION = 10, 40, 5

# ── desplazamiento_maximo ────────────────────────────────────────────────
dm = Recombination.desplazamiento_maximo
check("n=240 -> 0 celdas", int(dm(V_UMBRAL, 240 * DT, CELDA)) == 0)
check("n=241 -> 1 celda", int(dm(V_UMBRAL, 241 * DT, CELDA)) == 1)
check("v=3e-7: n=416 -> 0", int(dm(3e-7, 416 * DT, CELDA)) == 0)
check("v=3e-7: n=417 -> 1", int(dm(3e-7, 417 * DT, CELDA)) == 1)
check("v=0 nunca mueve", int(dm(0, 1e6 * DT, CELDA)) == 0)


def mover(estado, tiempo, voltaje):
    return Recombination.move_oxygen_ions(
        tiempo_mov=tiempo,
        oxygen_state=estado,
        temperature=300.0,
        E_field_filas=np.zeros(FILAS),
        grid_size=CELDA,
        vibration_frequency=1e13,
        gamma_drift=1.0,
        migration_energy=1.0,
        cte_red=CELDA,
        voltage=voltaje,
        velocity_thresholds=THRESHOLDS,
    )


def un_ion(col=0):
    e = np.zeros((FILAS, COLS), dtype=np.int8)
    e[FILA_ION, col] = 1
    return e


def columna(e):
    return int(np.where(e[FILA_ION] == 1)[0][0])


# ── move_oxygen_ions con un ion ──────────────────────────────────────────
e, v, posible = mover(un_ion(), 240 * DT, -1.2)
check("t=240·dt: el ion no se mueve", columna(e) == 0 and not posible)
e, v, posible = mover(un_ion(), 241 * DT, -1.2)
check("t=241·dt: el ion avanza exactamente 1 celda", columna(e) == 1 and posible)
e, v, posible = mover(un_ion(), 1000 * DT, -1.0)
check("V bajo el umbral (|V|<1.15): v=0, no se mueve", columna(e) == 0 and not posible and v == 0)
e, v, posible = mover(np.zeros((FILAS, COLS), dtype=np.int8), 241 * DT, -1.2)
check("sin iones: movimiento_posible True igualmente", posible and e.sum() == 0)


# ── bucle reloj + reset al mover (misma lógica que PP_reset/SP_reset) ────
def simular(voltajes):
    reloj = AcumuladorTiempo(DT)
    estado = un_ion()
    pasos_mov = []
    for k, V in enumerate(voltajes, start=1):
        t = reloj.actualizar(V)
        estado, _, posible = mover(estado, t, V)
        if posible:
            reloj.reiniciar()
            pasos_mov.append(k)
    return pasos_mov, columna(estado)


pasos, col = simular([-1.2] * 800)
check("V constante: mueve en los pasos 241 y 482 y 723", pasos == [241, 482, 723], str(pasos))
check("columna final = 3", col == 3)

pasos, col = simular([-1.2] * 100 + [-1.3] * 300)
check("cambio de V reinicia: mueve en 100+241=341", pasos == [341], str(pasos))

pasos, col = simular([-1.2] * 200 + [0.0] * 5 + [-1.2] * 200)
check("V=0 intermedio reinicia: no hay movimiento (200+200<241 cada tramo)", pasos == [] and col == 0, str(pasos))

if fallos:
    print(f"\n{len(fallos)} fallos: {fallos}")
    sys.exit(1)
print("\nTodo OK")
