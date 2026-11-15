"""
Verificación de `RRAM.estado_termico` (D3) y `RRAM.fisica_paso.PasoSimulacion` (D1).

Las comprobaciones de amortiguación y electrodo vienen de tests/test_procesar_temperatura.py, que
verificaba `Temperature.procesar_temperatura_amortiguamiento`: misma referencia independiente,
ahora contra `EstadoTermico.registrar_fvm` / `registrar_joule`, que la sustituyen.

Ejecutar desde la raíz del repo:
    python3 tests/test_estado_termico.py
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from RRAM.thermal import amortiguacion, muro  # noqa: E402
from RRAM.estado_fase import EstadoFase  # noqa: E402
from RRAM.estado_termico import CampoTemperatura, EstadoTermico  # noqa: E402
from RRAM.fisica_paso import PasoSimulacion  # noqa: E402

fallos = []


def check(nombre, cond, detalle=""):
    print(f"{'  OK  ' if cond else ' FALLO'} | {nombre}" + (f"  ({detalle})" if detalle else ""))
    if not cond:
        fallos.append(nombre)


params = SimpleNamespace(init_temp=300.0, paso_temporal=1e-3)
ctes = SimpleNamespace(
    k_amortiguacion_termica=1.0, umbral_caida_temperatura=1.0, k_electrodo_enfriamiento=2.0,
    umbral_caida_electrodo=0.1, grosor_filamento=1, Temperatura_electrodo=300.0,
)
centros, N = [3], 1
rng = np.random.default_rng(0)


def estado(anterior, Te, actual=None):
    return EstadoTermico(actual or CampoTemperatura.joule(300.0), anterior, Te, params, centros, N)


def referencia_mapa(T, ant, Te, amortiguar=True):
    """Rama FVM: perfil del electrodo con el mapa SIN amortiguar, luego el código previo de las fases."""
    filas = muro.filas_filamento(T.shape[0], [3], 1)
    Te = amortiguacion.actualizar_temperatura_electrodo(T, Te, filas, T_0=300.0, k=2.0, dt=1e-3, umbral=0.1)
    if amortiguar:
        T = amortiguacion.amortiguar_mapa_fvm(T, ant, T_0=300.0, k=1.0, dt=1e-3, umbral=1.0)
    ant = T[:, 1:-1]
    perfiles = muro.extraer_perfiles_filamentos(matriz_temperaturas=ant, filas_centros=centros)
    return T, ant, Te, [float(np.max(p)) if p is not None else 300.0 for p in perfiles]


def tupla(e, T_fils):
    return (e.actual.legado(), e.anterior, e.T_electrodo, T_fils)


def iguales(a, b):
    return all(
        np.array_equal(x, y) if isinstance(x, np.ndarray) or isinstance(y, np.ndarray) else x == y
        for x, y in zip(a, b)
    )


# --- mapa FVM: con caída (actúa la amortiguación) y con calentamiento
ant = 300.0 + 50.0 * rng.random((7, 5))
for nombre, T in [("caída", 300.0 + 5.0 * rng.random((7, 7))), ("calentamiento", 400.0 + 5.0 * rng.random((7, 7)))]:
    for amort in (True, False):
        e = estado(ant, 310.0)
        check(f"mapa ({nombre}, amortiguar={amort}) = referencia", iguales(tupla(e, e.registrar_fvm(T, ctes, amortiguar=amort)), referencia_mapa(T, ant, 310.0, amort)))

# --- mapa contra una anterior escalar (primer paso tras percolar)
T = 300.0 + 5.0 * rng.random((7, 7))
e = estado(450.0, 300.0)
check("mapa con anterior escalar = referencia", iguales(tupla(e, e.registrar_fvm(T, ctes)), referencia_mapa(T, 450.0, 300.0)))

# --- el perfil del electrodo sale del mapa SIN amortiguar
T = np.full((7, 7), 300.0)
T[2:5, 1] = 301.0
T[2:5, -2] = 301.0
e = estado(np.full((7, 5), 500.0), 300.0)
e.registrar_fvm(T, ctes)
check("la amortiguación sí cambia las columnas adyacentes", e.actual.mapa_fvm[3, 1] > 301.0, f"{e.actual.mapa_fvm[3, 1]:.3f}")
check("perfil calculado con el mapa sin amortiguar", np.allclose(e.T_electrodo[2:5], 301.0) and np.all(e.T_electrodo[[0, 1, 5, 6]] == 300.0), str(e.T_electrodo))

# --- Joule sin percolar, amortiguando (PP_set, PP_reset, SP_reset)
for T_raw, ant in [(300.5, 302.0), (305.0, 302.0), (300.2, 310.0)]:
    Tref = amortiguacion.amortiguar_enfriamiento(T_raw, ant, T_0=300.0, k=1.0, dt=1e-3, umbral=1.0)
    Te_ref = amortiguacion.temperatura_electrodo_joule(Tref, 300.0, k=2.0)
    e = estado(ant, 300.0)
    T_fils = e.registrar_joule(T_raw, ctes)
    check(f"Joule amortiguado (T={T_raw}, ant={ant}) = código previo", tupla(e, T_fils) == (Tref, Tref, Te_ref, [Tref]))

# --- Joule tras la rotura: la anterior es un mapa → no se amortigua
ant_mapa = np.full((7, 5), 400.0)
e = estado(ant_mapa, 300.0)
e.registrar_joule(300.5, ctes)
check("Joule tras rotura: sin amortiguar", e.actual.legado() == 300.5 and e.anterior == 300.5)

# --- Joule sin amortiguar ni actualizar la anterior (SP_set)
e = estado(ant_mapa, 300.0)
T_fils = e.registrar_joule(300.5, ctes, amortiguar=False, actualizar_anterior=False)
check("SP_set Joule: T sin tocar", e.actual.legado() == 300.5 and T_fils == [300.5])
check("SP_set Joule: anterior sin actualizar", e.anterior is ant_mapa)

# --- como_mapa igual que el np.full_like anterior
check("como_mapa = np.full_like(..., dtype=float)",
      np.array_equal(amortiguacion.como_mapa(302.25, (7, 5)), np.full_like(np.zeros((7, 5), dtype=np.int64), 302.25, dtype=float)))

# --- CampoTemperatura
mapa = 300.0 + rng.random((6, 8))
c = CampoTemperatura.desde_legado(mapa)
check("desde_legado(mapa 2D) es FVM; interior sin columnas de electrodo",
      c.mapa_fvm is mapa and np.array_equal(c.interior, mapa[:, 1:-1]) and c.legado() is mapa and not c.es_escalar)
c = CampoTemperatura.desde_legado(312.5)
check("desde_legado(escalar) es Joule", c.es_escalar and c.interior == 312.5 and c.legado() == 312.5 and c.como_escalar() == 312.5)
e = estado(300.0, 300.0)
check("Joule: T de cada filamento = [T] * N", e.registrar_joule(305.0, ctes, amortiguar=False) == [305.0] * N)
try:
    CampoTemperatura.fvm(mapa).como_escalar()
    check("como_escalar de un mapa lanza ValueError", False)
except ValueError:
    check("como_escalar de un mapa lanza ValueError", True)
check("maximo", CampoTemperatura.fvm(mapa).maximo() == float(np.max(mapa)))

# --- EstadoTermico: arranques, perfiles y paso a mapa
e = EstadoTermico.heredado(EstadoFase("sp_set", None, mapa, 0.0, perfil_electrodo=np.full(6, 305.0)), params, ctes, centros, N)
check("heredado: anterior = mapa sin electrodos y electrodo heredado",
      np.array_equal(e.anterior, mapa[:, 1:-1]) and np.all(e.T_electrodo == 305.0))
e = EstadoTermico.heredado(EstadoFase("pp_reset", None, 350.0, 0.0), params, ctes, centros, N)
check("heredado sin perfil: electrodo = constante", e.anterior == 350.0 and e.T_electrodo == 300.0)
try:
    e.perfiles_anteriores()
    check("perfiles con anterior escalar lanza ValueError", False)
except ValueError:
    check("perfiles con anterior escalar lanza ValueError", True)
e = EstadoTermico.inicial(params, ctes, centros, N)
e.anterior_a_mapa_si_escalar((7, 5))
check("primera percolación: anterior pasa a mapa lleno de la actual",
      isinstance(e.anterior, np.ndarray) and e.anterior.shape == (7, 5) and np.all(e.anterior == 300.0))
check("perfiles con anterior mapa", np.array_equal(e.perfiles_anteriores()[0], np.full(5, 300.0)))

# --- PasoSimulacion
p = PasoSimulacion()
check("paso nuevo vacío", p.R_local is None and p.matriz_muro is None and p.matriz_muro_para_plot([1]) is None)
p.sin_percolacion(2)
p.T_fils = [301.0, 302.0]
f = p.fila(0.5, 1.1, 1e-6, 2e-3)
check("fila sin percolación", f[:3] == [0.5, 1.1, 1e-6] and np.isnan(f[3]) and all(np.isnan(f[4:8])) and f[8:] == [301.0, 302.0, 2e-3])

print()
if fallos:
    print(f"{len(fallos)} FALLO(S): {fallos}")
    sys.exit(1)
print("Todo OK")
