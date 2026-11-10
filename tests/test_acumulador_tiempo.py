"""
Verificación de `AcumuladorTiempo`: t_ef = n·dt mientras el voltaje no cambia.

Ejecutar desde la raíz del repo:
    python3 tests/test_acumulador_tiempo.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from RRAM.acumulador_tiempo import AcumuladorTiempo  # noqa: E402

fallos = []


def check(nombre, cond, detalle=""):
    print(f"{'  OK  ' if cond else ' FALLO'} | {nombre}" + (f"  ({detalle})" if detalle else ""))
    if not cond:
        fallos.append(nombre)


def serie(voltajes, dt=0.1):
    a = AcumuladorTiempo(dt)
    return [round(a.actualizar(v) / dt) for v in voltajes]


check("pulso constante acumula 1..5", serie([1, 1, 1, 1, 1]) == [1, 2, 3, 4, 5])
check("cambio de V reinicia a 1", serie([1, 1, 1, 2, 2]) == [1, 2, 3, 1, 2])
check("V=0 no acumula (siempre dt)", serie([1, 1, 0, 0, 0]) == [1, 2, 1, 1, 1])
check("tras V=0 el mismo V anterior vuelve a empezar", serie([1, 1, 0, 1, 1]) == [1, 2, 1, 1, 2])
check("rampa: siempre dt", serie([0.1, 0.2, 0.3, 0.4]) == [1, 1, 1, 1])
check("t_ef en segundos", np.isclose(AcumuladorTiempo(0.1).actualizar(2.0), 0.1))

# reiniciar(): la cuenta vuelve a 0 pero se recuerda el voltaje -> siguiente paso con el mismo V vale 1·dt
a = AcumuladorTiempo(0.1)
[a.actualizar(1.0) for _ in range(4)]
a.reiniciar()
check("tras reiniciar, mismo V vale 1·dt", round(a.actualizar(1.0) / 0.1) == 1)
check("y vuelve a acumular", round(a.actualizar(1.0) / 0.1) == 2)

if fallos:
    print(f"\n{len(fallos)} fallos: {fallos}")
    sys.exit(1)
print("\nTodo OK")
