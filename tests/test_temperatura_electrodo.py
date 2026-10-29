"""
Verificación del perfil de temperatura de los electrodos:
`Temperature.filas_filamento`, `actualizar_temperatura_electrodo`, `temperatura_electrodo_joule`
y el perfil como condición de contorno de `solve_thermal_state`.

Ejecutar desde la raíz del repo:
    python3 tests/test_temperatura_electrodo.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from RRAM import Temperature  # noqa: E402

T0, K, DT, UMBRAL = 300.0, 10.0, 1e-3, 0.1
F = float(np.exp(-K * DT))
NY, NX = 12, 6

fallos = []


def check(nombre, cond, detalle=""):
    print(f"{'  OK  ' if cond else ' FALLO'} | {nombre}" + (f"  ({detalle})" if detalle else ""))
    if not cond:
        fallos.append(nombre)


# --- filas del filamento: centro ± grosor
f = Temperature.filas_filamento(NY, [5], 1)
check("filas: centro ± grosor (entero)", np.array_equal(np.where(f)[0], [4, 5, 6]))
f = Temperature.filas_filamento(NY, [2, 8], [0, 2])
check("filas: un grosor por filamento", np.array_equal(np.where(f)[0], [2, 6, 7, 8, 9, 10]))
f = Temperature.filas_filamento(NY, [0, 11], 2)
check("filas: recortadas en los bordes", np.array_equal(np.where(f)[0], [0, 1, 2, 9, 10, 11]))
try:
    Temperature.filas_filamento(NY, [2, 8], [1])
    check("filas: nº de grosores distinto lanza ValueError", False)
except ValueError:
    check("filas: nº de grosores distinto lanza ValueError", True)

filas = Temperature.filas_filamento(NY, [5], 1)


def mapa(izq, der, base=T0):
    """Mapa (NY, NX+2) con las columnas adyacentes izquierda/derecha dadas."""
    T = np.full((NY, NX + 2), base, dtype=float)
    T[:, 1] = izq
    T[:, -2] = der
    return T


actualizar = lambda T, prev, k=K: Temperature.actualizar_temperatura_electrodo(T, prev, filas, T0, k, DT, UMBRAL)  # noqa: E731

izq = T0 + np.arange(NY, dtype=float)  # filas 4-6 → 304, 305, 306
der = T0 + 2.0 * np.arange(NY, dtype=float)  # filas 4-6 → 308, 310, 312

# --- perfil: filas del filamento copiadas, resto a ambiente, lado de menor máximo
p = actualizar(mapa(izq, der), T0)
esperado = np.where(filas, izq, T0)
check("perfil: copia las filas del filamento y ambiente fuera", np.array_equal(p, esperado), str(p[3:8]))
check("perfil: elige el lado de menor máximo (izquierda)", p.max() == 306.0)
p = actualizar(mapa(der, izq), T0)
check("perfil: elige el lado de menor máximo (derecha)", np.array_equal(p, esperado))
check("perfil: forma (Ny,)", p.shape == (NY,))

# --- k desactivada → electrodo fijo
check("k = 0: devuelve el anterior (escalar)", actualizar(mapa(izq, der), 300.0, k=0.0) == 300.0)
check("k = None: devuelve el anterior", actualizar(mapa(izq, der), 300.0, k=None) == 300.0)

# --- umbral: caída > 0.1 K → enfría con exp(-k dt); caída <= 0.1 K o subida → valor nuevo
prev = np.where(filas, 320.0, T0)
nuevo_col = np.where(filas, 305.0, T0)  # caída de 15 K en las filas del filamento
p = actualizar(mapa(nuevo_col, nuevo_col + 1), prev)
esp = T0 + (320.0 - T0) * F
check("caída > umbral: enfriamiento exponencial", np.allclose(p[filas], esp), f"{p[5]:.4f} vs {esp:.4f}")
check("fuera de las filas: ambiente", np.all(p[~filas] == T0))
p = actualizar(mapa(np.where(filas, 319.95, T0), np.full(NY, 400.0)), prev)
check("caída <= umbral: toma el valor nuevo", np.allclose(p[filas], 319.95))
p = actualizar(mapa(np.where(filas, 330.0, T0), np.full(NY, 400.0)), prev)
check("calentamiento: toma el valor nuevo", np.allclose(p[filas], 330.0))

# --- anterior escalar (primer paso FVM)
p = actualizar(mapa(np.where(filas, 305.0, T0), np.full(NY, 400.0)), 320.0)
check("anterior escalar: perfil (Ny,) y enfría en las filas", p.shape == (NY,) and np.allclose(p[filas], T0 + 20.0 * F))

# --- V = 0 sostenido: decaimiento exponencial del perfil (mapa ya a ambiente en las adyacentes)
perfil, tray = np.where(filas, 400.0, T0), []
for _ in range(5):
    perfil = actualizar(mapa(np.full(NY, T0), np.full(NY, T0)), perfil)
    tray.append(perfil[5])
check("V=0 sostenido: decaimiento exponencial", np.allclose(tray, [T0 + 100.0 * F**n for n in range(1, 6)]), str(np.round(tray, 3)))

# --- Joule: escalar
check("Joule activa: electrodo = T_joule", Temperature.temperatura_electrodo_joule(320.0, 300.0, K) == 320.0)
check("Joule desactivada: electrodo fijo", Temperature.temperatura_electrodo_joule(320.0, 300.0, 0.0) == 300.0)
check("Joule tras perfil: vuelve a escalar", Temperature.temperatura_electrodo_joule(320.0, np.full(NY, 350.0), K) == 320.0)

# --- solve_thermal_state con perfil
types = np.zeros((NY, NX + 2), dtype=int)
types[:, 0] = types[:, -1] = 3
types[5, 1:-1] = 1
Q = np.zeros(types.shape)
Q[5, 1:-1] = 1e17
props = {0: {"k": 2.3}, 1: {"k": 15.0}, 2: {"k": 1e-6}, 3: {"k": 5.0}}
a = Temperature.solve_thermal_state(types, Q, props, 2.5e-10, 310.0)
b = Temperature.solve_thermal_state(types, Q, props, 2.5e-10, np.full(NY, 310.0))
check("solver: perfil uniforme = escalar (bit a bit)", np.array_equal(a, b))
perfil = np.where(filas, 330.0, T0)
c = Temperature.solve_thermal_state(types, Q, props, 2.5e-10, perfil)
check("solver: las celdas de electrodo toman el perfil", np.array_equal(c[:, 0], perfil) and np.array_equal(c[:, -1], perfil))

print()
if fallos:
    print(f"{len(fallos)} FALLO(S): {fallos}")
    sys.exit(1)
print("Todo OK")
