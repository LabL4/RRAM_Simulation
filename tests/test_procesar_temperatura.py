"""
Verificación de `Temperature.procesar_temperatura_amortiguamiento`: cada rama debe dar
lo mismo que el código que antes estaba repetido en las fases (el electrodo, con la ley del perfil).

Ejecutar desde la raíz del repo:
    python3 tests/test_procesar_temperatura.py
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from RRAM import Temperature  # noqa: E402

fallos = []


def check(nombre, cond, detalle=""):
    print(f"{'  OK  ' if cond else ' FALLO'} | {nombre}" + (f"  ({detalle})" if detalle else ""))
    if not cond:
        fallos.append(nombre)


params = SimpleNamespace(init_temp=300.0, paso_temporal=1e-3)
ctes = SimpleNamespace(
    k_amortiguacion_termica=1.0, umbral_caida_temperatura=1.0, k_electrodo_enfriamiento=2.0,
    umbral_caida_electrodo=0.1, grosor_filamento=1,
)
centros, N = [3], 1
rng = np.random.default_rng(0)
procesar = Temperature.procesar_temperatura_amortiguamiento


def referencia_mapa(T, ant, Te, amortiguar=True):
    """Rama FVM: perfil del electrodo con el mapa SIN amortiguar, luego el código previo de las fases."""
    filas = Temperature.filas_filamento(T.shape[0], [3], 1)
    Te = Temperature.actualizar_temperatura_electrodo(T, Te, filas, T_0=300.0, k=2.0, dt=1e-3, umbral=0.1)
    if amortiguar:
        T = Temperature.amortiguar_mapa_fvm(T, ant, T_0=300.0, k=1.0, dt=1e-3, umbral=1.0)
    ant = T[:, 1:-1]
    perfiles = Temperature.extraer_perfiles_filamentos(matriz_temperaturas=ant, filas_centros=centros)
    return T, ant, Te, [float(np.max(p)) if p is not None else 300.0 for p in perfiles]


def iguales(a, b):
    return all(
        np.array_equal(x, y) if isinstance(x, np.ndarray) or isinstance(y, np.ndarray) else x == y
        for x, y in zip(a, b)
    )


# --- mapa FVM: con caída (actúa la amortiguación) y con calentamiento
ant = 300.0 + 50.0 * rng.random((7, 5))
for nombre, T in [("caída", 300.0 + 5.0 * rng.random((7, 7))), ("calentamiento", 400.0 + 5.0 * rng.random((7, 7)))]:
    for amort in (True, False):
        r = procesar(T, ant, 310.0, params, ctes, centros, N, amortiguar=amort)
        check(f"mapa ({nombre}, amortiguar={amort}) = referencia", iguales(r, referencia_mapa(T, ant, 310.0, amort)))

# --- mapa contra una anterior escalar (primer paso tras percolar)
T = 300.0 + 5.0 * rng.random((7, 7))
check("mapa con anterior escalar = referencia", iguales(procesar(T, 450.0, 300.0, params, ctes, centros, N),
                                                       referencia_mapa(T, 450.0, 300.0)))

# --- el perfil del electrodo sale del mapa SIN amortiguar
T = np.full((7, 7), 300.0)
T[2:5, 1] = 301.0  # columnas adyacentes bajas: la amortiguación las levantaría hacia la anterior caliente
T[2:5, -2] = 301.0
ant = np.full((7, 5), 500.0)
temp, _, Te, _ = procesar(T, ant, 300.0, params, ctes, centros, N)
check("la amortiguación sí cambia las columnas adyacentes", temp[3, 1] > 301.0, f"{temp[3, 1]:.3f}")
check("perfil calculado con el mapa sin amortiguar", np.allclose(Te[2:5], 301.0) and np.all(Te[[0, 1, 5, 6]] == 300.0), str(Te))

# --- Joule sin percolar, amortiguando (PP_set, PP_reset, SP_reset)
for T_raw, ant in [(300.5, 302.0), (305.0, 302.0), (300.2, 310.0)]:
    Tref = Temperature.amortiguar_enfriamiento(T_raw, ant, T_0=300.0, k=1.0, dt=1e-3, umbral=1.0)
    Te_ref = Temperature.temperatura_electrodo_joule(Tref, 300.0, k=2.0)
    r = procesar(T_raw, ant, 300.0, params, ctes, centros, N)
    check(f"Joule amortiguado (T={T_raw}, ant={ant}) = código previo", r == (Tref, Tref, Te_ref, [Tref]))

# --- Joule tras la rotura: la anterior es un mapa → no se amortigua
ant_mapa = np.full((7, 5), 400.0)
T_out, ant_out, *_ = procesar(300.5, ant_mapa, 300.0, params, ctes, centros, N)
check("Joule tras rotura: sin amortiguar", T_out == 300.5 and ant_out == 300.5)

# --- Joule sin amortiguar ni actualizar la anterior (SP_set)
r = procesar(300.5, ant_mapa, 300.0, params, ctes, centros, N, amortiguar=False, actualizar_anterior=False)
check("SP_set Joule: T sin tocar", r[0] == 300.5 and r[3] == [300.5])
check("SP_set Joule: anterior sin actualizar", r[1] is ant_mapa)

# --- PP_set percolado con filamentos incompletos: T_fils dadas y salida como matriz
r = procesar(301.0, ant_mapa, 300.0, params, ctes, centros, N,
             amortiguar=False, actualizar_anterior=False, T_fils=[302.5], forma=(7, 5))
check("percolado incompleto: matriz float del tamaño pedido",
      isinstance(r[0], np.ndarray) and r[0].shape == (7, 5) and r[0].dtype == float and np.all(r[0] == 301.0))
check("percolado incompleto: T_fils dadas y anterior sin tocar", r[3] == [302.5] and r[1] is ant_mapa)
check("percolado incompleto: electrodo sigue a la Joule escalar", r[2] == 301.0)

# --- como_mapa igual que el np.full_like anterior
estado = np.zeros((7, 5), dtype=np.int64)
check("como_mapa = np.full_like(..., dtype=float)",
      np.array_equal(Temperature.como_mapa(302.25, estado.shape), np.full_like(estado, 302.25, dtype=float)))

print()
if fallos:
    print(f"{len(fallos)} FALLO(S): {fallos}")
    sys.exit(1)
print("Todo OK")
