"""
Verificación de `Temperature.actualizar_temperatura_electrodo` / `temperatura_electrodo_joule`.

Ejecutar desde la raíz del repo:
    python3 tests/test_temperatura_electrodo.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from RRAM import Temperature  # noqa: E402

T0, K, DT = 300.0, 10.0, 1e-3
F = float(np.exp(-K * DT))
NY, NX = 6, 4

fallos = []


def check(nombre, cond, detalle=""):
    print(f"{'  OK  ' if cond else ' FALLO'} | {nombre}" + (f"  ({detalle})" if detalle else ""))
    if not cond:
        fallos.append(nombre)


def mapa(T_e, delta_izq=0.0, delta_der=0.0):
    """Mapa FVM (NY, NX+2): electrodos a T_e y columnas adyacentes subidas delta."""
    T = np.full((NY, NX + 2), T_e, dtype=float)
    T[:, 1] += delta_izq
    T[:, -2] += delta_der
    return T


actualizar = lambda T, Te, Tady, k=K: Temperature.actualizar_temperatura_electrodo(T, Te, Tady, T0, k, DT)  # noqa: E731

# desactivada -> electrodo fijo
Te, Tady = actualizar(mapa(300.0, 5.0, 5.0), 300.0, -np.inf, k=0.0)
check("k = 0: electrodo fijo", Te == 300.0)
check("k = 0: devuelve T_ady igualmente", Tady == 305.0)
check("k = None: electrodo fijo", actualizar(mapa(300.0, 5.0, 5.0), 300.0, -np.inf, k=None)[0] == 300.0)

# menor de las dos columnas adyacentes
Te, Tady = actualizar(mapa(300.0, 2.0, 0.5), 300.0, -np.inf)
check("elige la menor columna adyacente", Tady == 300.5, f"{Tady}")
check("calienta hasta T_ady", Te == 300.5)
check("devuelve float de Python", type(Te) is float and type(Tady) is float)

# la media es por columna, no el máximo
T = mapa(300.0)
T[0, 1] += 6.0
T[:, -2] += 10.0
check("usa la media de la columna", actualizar(T, 300.0, -np.inf)[1] == 301.0)

# calienta si T_ady sube
Te, Tady = actualizar(mapa(310.0, 0.25, 0.25), 310.0, 310.0)
check("T_ady sube: calienta", Te == 310.25)

# V = 0: mapa uniforme, T_ady == anterior -> enfría
Te, Tady = actualizar(mapa(400.0), 400.0, 400.0)
check("mapa uniforme con T_ady igual: enfría", abs(Te - (T0 + 100.0 * F)) < 1e-9, f"{Te:.4f}")

# T_ady baja: enfría
Te, _ = actualizar(mapa(380.0), 380.0, 400.0)
check("T_ady baja: enfría", abs(Te - (T0 + 80.0 * F)) < 1e-9)

# V = 0 sostenido: decaimiento exponencial hacia T0
Te, Tady, trayectoria = 400.0, 400.0, []
for _ in range(5):
    Te, Tady = actualizar(mapa(Te), Te, Tady)
    trayectoria.append(Te)
esp = [T0 + 100.0 * F**n for n in range(1, 6)]
check("V=0 sostenido: decaimiento exponencial hacia T0", np.allclose(trayectoria, esp), str(np.round(trayectoria, 3)))

# tren de pulsos sintético: 1000 pasos con delta = 0.25 K suben 250 K
Te, Tady = 300.0, -np.inf
for _ in range(1000):
    Te, Tady = actualizar(mapa(Te, 0.25, 0.25), Te, Tady)
check("pulso de 1000 pasos con delta 0.25 K: 550 K", abs(Te - 550.0) < 1e-6, f"{Te:.3f}")
for _ in range(2000):
    Te, Tady = actualizar(mapa(Te), Te, Tady)
check("2000 pasos sin tensión (k=10): vuelve a T0", abs(Te - T0) < 0.01, f"{Te:.4f}")

# rama Joule
check("Joule activa: electrodo = T_joule", Temperature.temperatura_electrodo_joule(320.0, 300.0, K) == (320.0, 320.0))
check("Joule desactivada: electrodo fijo", Temperature.temperatura_electrodo_joule(320.0, 300.0, 0.0) == (300.0, 320.0))

print()
if fallos:
    print(f"{len(fallos)} FALLO(S): {fallos}")
    sys.exit(1)
print("Todo OK")
