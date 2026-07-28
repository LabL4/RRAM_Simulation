"""
Verificación del cálculo de calor POR FILAMENTO sobre el mapa de resistencias.

Ejecutar desde la raíz del repo:
    python3 tests/test_calor_por_filamento.py

Comprueba:
  1. EQUIVALENCIA con el modelo de resistencia constante (alpha_T = 0):
     media armónica == R/N, y Q == la fórmula anterior sigma*(dV/h)^2
  2. Invariante de potencia: sum(Q)*h^2 == V*I_total*factor/h
  3. Reparto entre filamentos == I_f^2 * R_f
  4. Geometría: el mapa es autodescriptivo y el inf se excluye solo
  5. Filamento no formado: no genera calor y no afecta al resto
  6. alpha_T != 0: cada celda disipa según SU resistencia
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from RRAM import CurrentSolver, Temperature  # noqa: E402

Ny, Nx = 160, 40
R_CELL = 4.3
H = 2.5e-10
SIGMA_0 = 930232558.1395348  # = 1 / (R_CELL * H)
T_0 = 300.0
FACTOR = 5.5e-5
V = 0.6
RANGES = [(0, 52), (53, 105), (106, 159)]
CENTROS = [26, 79, 132]

# Tolerancia: la media armónica reasocia operaciones respecto a R/N, así que la
# equivalencia es a precisión de máquina, no bit a bit.
RTOL = 1e-12

fallos = []


def check(nombre, cond, detalle=""):
    print(f"{'  OK  ' if cond else ' FALLO'} | {nombre}" + (f"  ({detalle})" if detalle else ""))
    if not cond:
        fallos.append(nombre)


def q_referencia(cf, I_fils):
    """Fórmula ANTERIOR (R constante, sigma*(dV/h)^2), en marco extendido."""
    Q = np.zeros((Ny, Nx + 2))
    for f, (lo, hi) in enumerate(RANGES):
        I_f = I_fils[f]
        if not np.isfinite(I_f) or I_f == 0.0:
            continue
        bloque = cf[lo : hi + 1, :] == 1
        for jj in range(Nx):
            idx = np.where(bloque[:, jj])[0]
            if len(idx) == 0:
                continue
            R_col = R_CELL / len(idx)
            Q[idx + lo, jj + 1] = (1.0 / (R_CELL * H)) * ((I_f * R_col / H) ** 2) * FACTOR
    return Q


def mapa(cf, temperatura=T_0, alpha_T=0.0):
    return CurrentSolver.mapa_resistencias(cf, temperatura, SIGMA_0, alpha_T, T_0, H)


def electrica(R_local):
    R_fils = CurrentSolver.calcular_resistencia_por_filamento(R_local, RANGES)
    R_tot = CurrentSolver.calcular_resistencia_paralelo(R_fils)
    I_fils = [V / r if r != np.inf else 0.0 for r in R_fils]
    return R_fils, R_tot, V / R_tot, I_fils


def construir(grosores):
    cf = np.zeros((Ny, Nx), dtype=int)
    for f, gfun in enumerate(grosores):
        c = CENTROS[f]
        for j in range(Nx):
            g = gfun(j)
            cf[c - g // 2 : c - g // 2 + g, j] = 1
    return cf


# ---------------------------------------------------------------- 1
print("\n[1] Equivalencia con el modelo de R constante (alpha_T = 0)")
rng = np.random.default_rng(0)
peor = 0.0
for _ in range(200):
    m = (rng.random((12, 9)) < 0.6).astype(int)
    if not m.any():
        continue
    R = CurrentSolver.mapa_resistencias(m, T_0, SIGMA_0, 0.0, T_0, H)
    R_cols = CurrentSolver.resistencias_por_columna(R)
    n = m.sum(axis=0).astype(float)
    ref = np.where(n > 0, R_CELL / np.maximum(n, 1), np.inf)
    fin = np.isfinite(ref) & np.isfinite(R_cols)
    if fin.any():
        peor = max(peor, float(np.max(np.abs(R_cols[fin] - ref[fin]) / ref[fin])))
    if not np.array_equal(np.isfinite(R_cols), np.isfinite(ref)):
        fallos.append("centinela de columna vacía difiere")
check("media armónica == R/N", peor < RTOL, f"peor error relativo = {peor:.2e} en 200 matrices")


# ---------------------------------------------------------------- 2 y 3
print("\n[2/3] Grosor uniforme: invariante de potencia + Q == fórmula anterior")
cf_u = construir([lambda j: 5, lambda j: 7, lambda j: 9])
R_u = mapa(cf_u)
R_fils, R_tot, I_tot, I_fils = electrica(R_u)
Q_new = Temperature.calculate_heat_source(H, R_u, FACTOR, RANGES, I_fils)
Q_ref = q_referencia(cf_u, I_fils)

P_num = Q_new.sum() * H * H
P_esp = V * I_tot * FACTOR / H
check("invariante de potencia total", np.isclose(P_num, P_esp, rtol=1e-10), f"{P_num:.6e} vs {P_esp:.6e}")
dif = np.max(np.abs(Q_new - Q_ref)) / Q_ref.max()
check("Q == fórmula anterior", dif < RTOL, f"error relativo máx = {dif:.2e}")
check("forma extendida (Nx+2)", Q_new.shape == (Ny, Nx + 2), f"{Q_new.shape}")


# ---------------------------------------------------------------- 4
print("\n[4] Grosor anticorrelado: el reparto sigue a I_f^2*R_f")
cf_n = construir([lambda j: 10 if j < 20 else 2, lambda j: 6, lambda j: 2 if j < 20 else 10])
R_n = mapa(cf_n)
R_fils, R_tot, I_tot, I_fils = electrica(R_n)
Q_new = Temperature.calculate_heat_source(H, R_n, FACTOR, RANGES, I_fils)
check("invariante de potencia total", np.isclose(Q_new.sum() * H * H, V * I_tot * FACTOR / H, rtol=1e-10))

tot_true = sum(I_fils[i] ** 2 * R_fils[i] for i in range(3))
print("\n       fil |   Q reparto  |  I_f^2*R_f")
ok = True
for i, (lo, hi) in enumerate(RANGES):
    a = Q_new[lo : hi + 1].sum() / Q_new.sum() * 100
    b = I_fils[i] ** 2 * R_fils[i] / tot_true * 100
    print(f"        {i + 1}  |    {a:6.2f}%   |   {b:6.2f}%")
    ok &= bool(np.isclose(a, b, rtol=1e-10))
check("reparto == I_f^2*R_f", ok)


# ---------------------------------------------------------------- 5
print("\n[5] Mapa autodescriptivo y exclusión automática del inf")
check("isfinite(R) == (cf == 1)", bool(np.array_equal(np.isfinite(R_n), cf_n == 1)))
check("Q = 0 donde R = inf", bool(np.all(Q_new[:, 1:-1][~np.isfinite(R_n)] == 0.0)))
check("columnas de electrodo a cero", bool(np.all(Q_new[:, 0] == 0) and np.all(Q_new[:, -1] == 0)))


# ---------------------------------------------------------------- 6
print("\n[6] Filamento no formado")
R_roto = mapa(np.where(np.isin(np.arange(Ny)[:, None], np.arange(53, 106)), 0, cf_n))
R_f2, _, _, I_f2 = electrica(R_roto)
check("banda vacía -> R_fil = inf", R_f2[1] == np.inf, f"R_fils={[f'{r:.1f}' if np.isfinite(r) else 'inf' for r in R_f2]}")
check("banda vacía -> I_fil = 0", I_f2[1] == 0.0)
Q_roto = Temperature.calculate_heat_source(H, R_roto, FACTOR, RANGES, I_f2)
check("banda vacía no genera calor", Q_roto[53:106].sum() == 0.0)


# ---------------------------------------------------------------- 7
print("\n[7] alpha_T != 0: cada celda disipa según SU resistencia")
T_map = np.full((Ny, Nx), T_0)
T_map[CENTROS[0] - 2 : CENTROS[0] + 1, :] = 700.0  # media banda caliente
R_t = mapa(cf_u, T_map, alpha_T=2e-3)
R_fils_t, _, _, I_fils_t = electrica(R_t)
Q_t = Temperature.calculate_heat_source(H, R_t, FACTOR, RANGES, I_fils_t)

fila_cal, fila_fria = CENTROS[0] - 1, CENTROS[0] + 1
print(f"       R: caliente={R_t[fila_cal, 5]:.3f} Ohm  fría={R_t[fila_fria, 5]:.3f} Ohm")
print(f"       Q: caliente={Q_t[fila_cal, 6]:.4e}      fría={Q_t[fila_fria, 6]:.4e}")
check("celda caliente más resistiva", R_t[fila_cal, 5] > R_t[fila_fria, 5])
check("celda caliente disipa menos (autolimitante)", Q_t[fila_cal, 6] < Q_t[fila_fria, 6])

# Misma geometría (cf_u) con alpha_T = 0 como referencia fría
R_fils_frio, _, I_tot_frio, _ = electrica(mapa(cf_u))
print(f"       R_fil[0]: alpha_T=0 -> {R_fils_frio[0]:.3f} Ohm | alpha_T=2e-3 -> {R_fils_t[0]:.3f} Ohm")
check("R del filamento sube con T", R_fils_t[0] > R_fils_frio[0])
check("las bandas frías no cambian", np.isclose(R_fils_t[1], R_fils_frio[1], rtol=1e-12))


print("\n" + "=" * 60)
if fallos:
    print(f"FALLOS ({len(fallos)}): " + ", ".join(sorted(set(fallos))))
    sys.exit(1)
print("TODOS LOS TESTS PASAN")
