"""
Verificación del cálculo de calor POR FILAMENTO.

Ejecutar desde la raíz del repo:
    python3 tests/test_calor_por_filamento.py

Comprueba 5 cosas:
  1. resistencias_por_columna no altera calcular_resistencia (no-regresión eléctrica)
  2. Invariante de potencia: sum(Q)*h^2 == V*I_total*factor/h  (exacto)
  3. NO-REGRESIÓN: con grosor uniforme en x, Q nuevo == Q viejo celda a celda
  4. CORRECCIÓN: con grosor no uniforme, el reparto pasa a coincidir con I_f^2*R_f
  5. Geometría: types_map y cf_clean_matrix ven las mismas celdas de filamento
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from RRAM import CurrentSolver, Temperature  # noqa: E402

Ny, Nx = 160, 40
R_CELL = 4.3
H = 2.5e-10
FACTOR = 5.5e-5
V = 0.6
RANGES = [(0, 52), (53, 105), (106, 159)]
CENTROS = [26, 79, 132]

fallos = []


def check(nombre, cond, detalle=""):
    print(f"{'  OK  ' if cond else ' FALLO'} | {nombre}" + (f"  ({detalle})" if detalle else ""))
    if not cond:
        fallos.append(nombre)


def q_antiguo(types_map, I_total):
    """Implementación ANTERIOR (por columnas, con I_total). Referencia de regresión."""
    Q = np.zeros(types_map.shape)
    for j in range(1, types_map.shape[1] - 1):
        idx = np.where(types_map[:, j] == 1)[0]
        if len(idx) == 0:
            continue
        R_col = R_CELL / len(idx)
        Q[idx, j] = (1.0 / (R_CELL * H)) * ((I_total * R_col / H) ** 2) * FACTOR
    return Q


def electrica(cf):
    R_fils = CurrentSolver.calcular_resistencia_por_filamento(cf, RANGES, R_CELL)
    R_tot = CurrentSolver.calcular_resistencia_paralelo(R_fils)
    I_fils = [V / r if r != np.inf else 0.0 for r in R_fils]
    return R_fils, R_tot, V / R_tot, I_fils


def construir(grosores_por_columna):
    """grosores_por_columna[f] -> callable(j) que da el grosor del filamento f en la columna j"""
    cf = np.zeros((Ny, Nx), dtype=int)
    for f, gfun in enumerate(grosores_por_columna):
        c = CENTROS[f]
        for j in range(Nx):
            g = gfun(j)
            cf[c - g // 2 : c - g // 2 + g, j] = 1
    return cf


# ---------------------------------------------------------------- 1
print("\n[1] No-regresión de la rama eléctrica")
rng = np.random.default_rng(0)
for _ in range(200):
    m = (rng.random((12, 9)) < 0.5).astype(int)
    ref = 0.0
    for j in range(m.shape[1]):
        n = m[:, j].sum()
        if n:
            ref += R_CELL / n
    check("calcular_resistencia identica", np.isclose(CurrentSolver.calcular_resistencia(m, R_CELL), ref)) if _ == 0 else None
    if not np.isclose(CurrentSolver.calcular_resistencia(m, R_CELL), ref):
        fallos.append("calcular_resistencia difiere")
        break
print("       200 matrices aleatorias comparadas contra la fórmula original")

# ---------------------------------------------------------------- 2 y 3 (grosor uniforme)
print("\n[2/3] Grosor uniforme en x  ->  invariante de potencia + no-regresión")
cf_u = construir([lambda j: 5, lambda j: 7, lambda j: 9])
tm_u = Temperature.crear_matriz_materiales(cf_u)
R_fils, R_tot, I_tot, I_fils = electrica(cf_u)
Q_new = Temperature.calculate_heat_source(tm_u, H, R_CELL, FACTOR, RANGES, I_fils)
Q_old = q_antiguo(tm_u, I_tot)

P_num = Q_new.sum() * H * H
P_esp = V * I_tot * FACTOR / H
check("invariante potencia total", np.isclose(P_num, P_esp, rtol=1e-10), f"{P_num:.6e} vs {P_esp:.6e}")
check("Q nuevo == Q antiguo (celda a celda)", np.allclose(Q_new, Q_old, rtol=1e-12), f"max dif={np.abs(Q_new - Q_old).max():.3e}")

# ---------------------------------------------------------------- 4 (grosor no uniforme)
print("\n[4] Grosor anticorrelado en x  ->  reparto correcto entre filamentos")
cf_n = construir([lambda j: 10 if j < 20 else 2, lambda j: 6, lambda j: 2 if j < 20 else 10])
tm_n = Temperature.crear_matriz_materiales(cf_n)
R_fils, R_tot, I_tot, I_fils = electrica(cf_n)
Q_new = Temperature.calculate_heat_source(tm_n, H, R_CELL, FACTOR, RANGES, I_fils)
Q_old = q_antiguo(tm_n, I_tot)

P_num = Q_new.sum() * H * H
check("invariante potencia total", np.isclose(P_num, V * I_tot * FACTOR / H, rtol=1e-10))

print("\n       fil |  antiguo  |   nuevo   |  I_f^2*R_f  (referencia)")
tot_true = sum(I_fils[i] ** 2 * R_fils[i] for i in range(3))
ok_reparto = True
for i, (lo, hi) in enumerate(RANGES):
    a = Q_old[lo : hi + 1].sum() / Q_old.sum() * 100
    b = Q_new[lo : hi + 1].sum() / Q_new.sum() * 100
    ref = I_fils[i] ** 2 * R_fils[i] / tot_true * 100
    print(f"        {i + 1}  |  {a:6.2f}%  |  {b:6.2f}%  |  {ref:6.2f}%")
    ok_reparto &= bool(np.isclose(b, ref, atol=1e-9))
check("reparto nuevo == I_f^2*R_f", ok_reparto)
check("el antiguo SI se desviaba (test discrimina)", not np.isclose(Q_old[0:53].sum() / Q_old.sum() * 100, I_fils[0] ** 2 * R_fils[0] / tot_true * 100, atol=0.5))

# ---------------------------------------------------------------- 5
print("\n[5] Coherencia geométrica types_map <-> cf_clean_matrix")
for nombre, cf, tm in [("uniforme", cf_u, tm_u), ("no uniforme", cf_n, tm_n)]:
    check(f"mismas celdas de filamento ({nombre})", np.array_equal(tm[:, 1:-1] == 1, cf == 1))
    for i, (lo, hi) in enumerate(RANGES):
        rc_t = CurrentSolver.resistencias_por_columna(tm[lo : hi + 1, 1:-1] == 1, R_CELL)
        rc_e = CurrentSolver.resistencias_por_columna(cf[lo : hi + 1, :], R_CELL)
        if not np.allclose(rc_t, rc_e, equal_nan=True):
            fallos.append(f"R_cols difieren fil{i + 1} ({nombre})")
print("       R_cols de la rama térmica == R_cols de la rama eléctrica")

# ---------------------------------------------------------------- filamento roto
print("\n[6] Filamento no formado (I_f = 0)")
I_rotos = [I_fils[0], 0.0, I_fils[2]]
Q_r = Temperature.calculate_heat_source(tm_n, H, R_CELL, FACTOR, RANGES, I_rotos)
check("banda sin corriente no genera calor", Q_r[53:106].sum() == 0.0)
check("las otras bandas no se ven afectadas", np.allclose(Q_r[0:53], Q_new[0:53]))

print("\n" + "=" * 60)
if fallos:
    print(f"FALLOS ({len(fallos)}): " + ", ".join(sorted(set(fallos))))
    sys.exit(1)
print("TODOS LOS TESTS PASAN")
