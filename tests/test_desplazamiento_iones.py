"""
Verificación del reloj por ion en el movimiento de iones de oxígeno:
`Recombination.move_oxygen_ions` (núcleo `_mover_con_bloqueo`) y `RelojesIones`.

Con v = 5.2e-7 m/s, dt = 1e-6 s y celda = 0.25e-9 m, d = round(v·n·dt/celda):
  n = 240 -> 0, n = 241 -> 1, n = 722 -> 2, n = 1202 -> 3.

Ejecutar desde la raíz del repo:
    python3 tests/test_desplazamiento_iones.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from RRAM import Recombination  # noqa: E402
from RRAM.acumulador_tiempo import RelojesIones  # noqa: E402

fallos = []


def check(nombre, cond, detalle=""):
    print(f"{'  OK  ' if cond else ' FALLO'} | {nombre}" + (f"  ({detalle})" if detalle else ""))
    if not cond:
        fallos.append(nombre)


DT, CELDA, V_UMBRAL = 1e-6, 0.25e-9, 5.2e-7
THRESHOLDS = {1.15: V_UMBRAL}
FILAS, COLS = 10, 40
N1, N2, N3 = 241, 722, 1202  # pasos para 1, 2 y 3 celdas


def mover(estado, reloj, voltaje=-1.2):
    return Recombination.move_oxygen_ions(
        reloj_iones=reloj,
        oxygen_state=estado,
        temperature=300.0,
        E_field_filas=np.zeros(FILAS),
        grid_size=CELDA,
        paso_temporal=DT,
        vibration_frequency=1e13,
        gamma_drift=1.0,
        migration_energy=1.0,
        cte_red=CELDA,
        voltage=voltaje,
        velocity_thresholds=THRESHOLDS,
    )


def malla(iones):
    """iones: {(fila, col): reloj}. Devuelve (oxygen_state int8, reloj int32)."""
    e = np.zeros((FILAS, COLS), dtype=np.int8)
    r = np.zeros((FILAS, COLS), dtype=np.int32)
    for (f, c), n in iones.items():
        e[f, c] = 1
        r[f, c] = n
    return e, r


def iones_de(e, r, f):
    """Lista [(col, reloj)] de los iones de la fila f."""
    return [(int(c), int(r[f, c])) for c in np.where(e[f] == 1)[0]]


def invariante(e, r):
    return bool(np.all(r[e == 0] == 0))


# ── Umbral de una celda (round) ───────────────────────────────────────────
e, r = malla({(5, 0): N1 - 1})
mover(e, r)
check("n=240: no se mueve y conserva el reloj", iones_de(e, r, 5) == [(0, N1 - 1)])
e, r = malla({(5, 0): N1})
mover(e, r)
check("n=241: avanza 1 celda y su reloj pasa a 0", iones_de(e, r, 5) == [(1, 0)])
e, r = malla({(5, 0): N3})
mover(e, r)
check("n=1202: avanza directamente 3 celdas (sin sorteo)", iones_de(e, r, 5) == [(3, 0)])
e, r = malla({(5, 0): 10**6})
_, v = mover(e, r, voltaje=-1.0)
check("V bajo el umbral: v=0, no se mueve", v == 0 and iones_de(e, r, 5) == [(0, 10**6)])

# ── Bloqueo ───────────────────────────────────────────────────────────────
e, r = malla({(5, 10): 0, (5, 9): N1})
mover(e, r)
check("bloqueo total: el de detrás no avanza y conserva su reloj", iones_de(e, r, 5) == [(9, N1), (10, 0)])
e, r = malla({(5, 10): 0, (5, 7): N3})
mover(e, r)
check("avance parcial: quiere 3, para en el hueco (col 9), reloj 0", iones_de(e, r, 5) == [(9, 0), (10, 0)])
e, r = malla({(5, 10): N1, (5, 9): N1})
mover(e, r)
check("el de delante se mueve y el de detrás ocupa su hueco en el mismo paso", iones_de(e, r, 5) == [(10, 0), (11, 0)])
e, r = malla({(5, 10): N1 - 1, (5, 9): N2})
mover(e, r)
check("cada ion con su reloj: delante quieto, detrás bloqueado", iones_de(e, r, 5) == [(9, N2), (10, N1 - 1)])
e, r = malla({(5, 10): 0, (6, 9): N1})
mover(e, r)
check("filas independientes: un ion en otra fila no bloquea", iones_de(e, r, 6) == [(10, 0)])

# ── Borde derecho ─────────────────────────────────────────────────────────
e, r = malla({(5, COLS - 1): N1})
mover(e, r)
check("sale por el borde: ion y reloj desaparecen", e.sum() == 0 and r.sum() == 0)
e, r = malla({(5, COLS - 1): 0, (5, COLS - 3): N3})
mover(e, r)
check("con un ion delante no sale: para antes del borde", iones_de(e, r, 5) == [(COLS - 2, 0), (COLS - 1, 0)])
e, r = malla({(5, COLS - 1): N1, (5, COLS - 2): N1})
mover(e, r)
check("el de delante sale y el de detrás ocupa su sitio", iones_de(e, r, 5) == [(COLS - 1, 0)])

# ── RelojesIones ──────────────────────────────────────────────────────────
e, _ = malla({(5, 3): 0, (6, 3): 0})
rel = RelojesIones(np.zeros((FILAS, COLS), dtype=np.int32))
for _ in range(3):
    rel.actualizar(-1.2, e)
check("mismo V: cada ion suma 1 por paso", rel.reloj[5, 3] == 3 and rel.reloj[6, 3] == 3 and rel.reloj.sum() == 6)
rel.actualizar(-1.3, e)
check("cambio de V: todos a 0 y suman el paso actual (1)", rel.reloj[5, 3] == 1 and rel.reloj[6, 3] == 1)
rel.actualizar(0.0, e)
check("V=0: todos a 0 sin sumar", rel.reloj.sum() == 0)
rel.actualizar(-1.2, e)
rel.actualizar(-1.2, e)
e[5, 3] = 0  # se recombina
rel.limpiar(e)
check("ion recombinado: su reloj se borra, el resto sigue", rel.reloj[5, 3] == 0 and rel.reloj[6, 3] == 2)

reloj_pp, v_pp = rel.estado()
heredado = RelojesIones(reloj_pp, v_pp)
heredado.actualizar(-1.2, e)
check("herencia con el mismo V: sigue acumulando", heredado.reloj[6, 3] == 3)
check("herencia: copia, no comparte memoria con el origen", rel.reloj[6, 3] == 2)
heredado = RelojesIones(reloj_pp, v_pp)
heredado.actualizar(-1.5, e)
check("herencia con distinto V: reinicia", heredado.reloj[6, 3] == 1)


# ── Bucle completo (mismo orden que PP_reset/SP_reset) ────────────────────
def simular(voltajes, generar=None, fila=5):
    """
    generar: {paso: fila} -> en ese paso se genera un ion en (fila, 0) DESPUÉS de actualizar
    el reloj (como generate_oxygen). Devuelve {fila: [pasos en que el ion avanzó]} y si el
    invariante reloj == 0 donde no hay ion se cumplió en todos los pasos.
    """
    generar = generar or {}
    estado = np.zeros((FILAS, COLS), dtype=np.int8)
    relojes = RelojesIones(np.zeros((FILAS, COLS), dtype=np.int32))
    if not generar:
        estado[fila, 0] = 1  # ion presente desde antes del primer paso
    movs, ok = {}, True
    for k, V in enumerate(voltajes, start=1):
        relojes.actualizar(V, estado)
        if k in generar:
            estado[generar[k], 0] = 1
        antes = estado.copy()
        mover(estado, relojes.reloj, V)
        relojes.limpiar(estado)
        ok &= invariante(estado, relojes.reloj)
        for f in np.where((antes != estado).any(axis=1))[0]:
            movs.setdefault(int(f), []).append(k)
    return movs, ok


movs, ok = simular([-1.2] * 800)
check("V constante: mueve en los pasos 241, 482 y 723", movs.get(5) == [241, 482, 723], str(movs))
check("invariante en V constante", ok)

movs, ok = simular([-1.2] * 400, generar={1: 3, 101: 7})
check("ion nuevo empieza en 0: generado en 1 mueve en 242", movs.get(3) == [242], str(movs.get(3)))
check("cada ion con su reloj: generado en 101 mueve en 342", movs.get(7) == [342], str(movs.get(7)))
check("invariante con generación", ok)

movs, ok = simular([-1.2] * 100 + [-1.3] * 300)
check("cambio de V reinicia: mueve en 100+241=341", movs.get(5) == [341], str(movs))

movs, ok = simular([-1.2] * 200 + [0.0] * 5 + [-1.2] * 200)
check("V=0 intermedio reinicia: no hay movimiento", movs == {}, str(movs))
check("invariante con cambios de V", ok)

if fallos:
    print(f"\n{len(fallos)} fallos: {fallos}")
    sys.exit(1)
print("\nTodo OK")
