"""
Verificación de `Temperature.amortiguar_enfriamiento` / `amortiguar_mapa_fvm`.

Ejecutar desde la raíz del repo:
    python3 tests/test_amortiguacion_termica.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from RRAM import Temperature  # noqa: E402

T0, K, DT, UMBRAL = 300.0, 1e5, 1e-6, 10.0
F = float(np.exp(-K * DT))

fallos = []


def check(nombre, cond, detalle=""):
    print(f"{'  OK  ' if cond else ' FALLO'} | {nombre}" + (f"  ({detalle})" if detalle else ""))
    if not cond:
        fallos.append(nombre)


amort = lambda Tc, Tp, k=K, u=UMBRAL: Temperature.amortiguar_enfriamiento(Tc, Tp, T0, k, DT, u)  # noqa: E731

# escalar con escalar -> float de Python
r = amort(300.0, 600.0)
esperado = T0 + 300.0 * F
check("escalar/escalar caída V=0: valor", abs(r - esperado) < 1e-9, f"{r:.3f} vs {esperado:.3f}")
check("escalar/escalar devuelve float de Python", type(r) is float)
check("escalar/escalar sin caída (np.float64 de entrada) devuelve float", type(amort(np.float64(500.0), np.float64(500.0))) is float)

# calentamiento / caída pequeña -> identidad
check("calentamiento: identidad (escalar)", amort(700.0, 600.0) == 700.0)
check("caída < umbral: identidad (escalar)", amort(595.0, 600.0) == 595.0)
check("caída == umbral: identidad (escalar)", amort(590.0, 600.0) == 590.0)

# max de seguridad: nunca por debajo de T_calc
r = amort(580.0, 600.0, k=1e7)  # f ~ 0 -> amortiguada ~ T0 < T_calc
check("max de seguridad: T >= T_calc", r >= 580.0 and abs(r - 580.0) < 1e-9, f"{r}")

# matrices
Tp = np.full((4, 5), 600.0)
Tc = np.array([[300.0] * 5, [700.0] * 5, [595.0] * 5, [600.0] * 5])
r = amort(Tc, Tp)
check("matriz: celdas con caída grande amortiguadas", np.allclose(r[0], T0 + 300.0 * F))
check("matriz: calentamiento intacto", np.array_equal(r[1], Tc[1]))
check("matriz: caída pequeña intacta", np.array_equal(r[2], Tc[2]))
check("matriz: sin cambio intacto", np.array_equal(r[3], Tc[3]))
check("matriz: no modifica las entradas", np.array_equal(Tp, np.full((4, 5), 600.0)) and Tc[0, 0] == 300.0)

# calentamiento puro: salida == T_calc en todas las celdas
Tc_cal = Tp + np.random.default_rng(0).uniform(0, 400, Tp.shape)
check("T_calc >= T_prev en todo el mapa: identidad", np.array_equal(amort(Tc_cal, Tp), Tc_cal))

# T_calc escalar T0 (V=0) con T_prev matriz caliente -> matriz
r = amort(300.0, Tp)
check("escalar T0 con T_prev matriz: devuelve matriz amortiguada", isinstance(r, np.ndarray) and np.allclose(r, T0 + 300.0 * F))
# T_calc matriz con T_prev escalar
r = amort(Tc, 600.0)
check("matriz con T_prev escalar: broadcast", r.shape == Tc.shape and np.allclose(r[0], T0 + 300.0 * F))

# desactivada
for k in (0.0, None, float("nan"), -1.0):
    check(f"k={k}: desactivada (devuelve T_calc)", amort(300.0, 600.0, k=k) == 300.0)
check("k=0 con matriz: devuelve el mismo objeto", amort(Tc, Tp, k=0.0) is Tc)

# umbral inválido
for u in (0.0, -5.0, None, float("nan")):
    try:
        amort(300.0, 600.0, u=u)
        check(f"umbral={u}: rechazado", False)
    except ValueError:
        check(f"umbral={u}: rechazado", True)

# mapa FVM: electrodos intactos
Ny, Nx = 4, 5
T_fvm = np.full((Ny, Nx + 2), 300.0)
T_fvm[:, 0] = 123.0
T_fvm[:, -1] = 456.0
T_prev = np.full((Ny, Nx), 600.0)
copia = T_fvm.copy()
r = Temperature.amortiguar_mapa_fvm(T_fvm, T_prev, T0, K, DT, UMBRAL)
check("FVM: forma conservada", r.shape == (Ny, Nx + 2))
check("FVM: columnas de electrodo intactas", np.array_equal(r[:, 0], copia[:, 0]) and np.array_equal(r[:, -1], copia[:, -1]))
check("FVM: interior amortiguado", np.allclose(r[:, 1:-1], T0 + 300.0 * F))
check("FVM: entrada sin modificar", np.array_equal(T_fvm, copia))
check("FVM desactivada: mismo objeto", Temperature.amortiguar_mapa_fvm(T_fvm, T_prev, T0, 0.0, DT, UMBRAL) is T_fvm)

# decaimiento exponencial sostenido (V=0 durante varios pasos)
T = 800.0
trayectoria = []
for _ in range(5):
    T = amort(300.0, T)
    trayectoria.append(T)
esp = [T0 + 500.0 * F ** n for n in range(1, 6)]
check("V=0 sostenido: decaimiento exponencial hacia T0", np.allclose(trayectoria, esp), str(np.round(trayectoria, 1)))

print()
if fallos:
    print(f"{len(fallos)} FALLO(S): {fallos}")
    sys.exit(1)
print("Todo OK")
