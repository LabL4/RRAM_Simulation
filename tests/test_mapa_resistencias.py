"""
Verificación de `CurrentSolver.mapa_resistencias`.

Ejecutar desde la raíz del repo:
    python3 tests/test_mapa_resistencias.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from RRAM import CurrentSolver  # noqa: E402

H = 2.5e-10
SIGMA_0 = 930232558.1395348  # = 1 / (4.3 * H)
R_REF = 4.3
T_0 = 300.0
ALPHA = 2e-3

fallos = []


def check(nombre, cond, detalle=""):
    print(f"{'  OK  ' if cond else ' FALLO'} | {nombre}" + (f"  ({detalle})" if detalle else ""))
    if not cond:
        fallos.append(nombre)


# Geometría de prueba: 3 filamentos horizontales en una matriz (160, 40)
Ny, Nx = 160, 40
CENTROS = [26, 79, 132]
cf = np.zeros((Ny, Nx), dtype=int)
for c in CENTROS:
    cf[c - 2 : c + 3, :] = 1


# ---------------------------------------------------------------- 1
print("\n[1] Punto de referencia: T = T_0 reproduce R_ref exacto")
R = CurrentSolver.mapa_resistencias(cf, T_0, SIGMA_0, ALPHA, T_0, H)
check("R en filamento == 4.3 (bit a bit)", bool(np.all(R[cf == 1] == R_REF)), f"valor={R[cf == 1][0]!r}")
check("R en óxido == inf", bool(np.all(np.isinf(R[cf == 0]))))
check("forma preservada", R.shape == cf.shape, f"{R.shape}")


# ---------------------------------------------------------------- 2
print("\n[2] alpha_T = 0 reproduce el modelo de resistencia constante")
R0 = CurrentSolver.mapa_resistencias(cf, np.full((Ny, Nx), 700.0), SIGMA_0, 0.0, T_0, H)
check("R uniforme = 4.3 pese a T=700 K", bool(np.all(R0[cf == 1] == R_REF)))


# ---------------------------------------------------------------- 3
print("\n[3] Dependencia con T: R(T) = R_ref * (1 + alpha_T*(T - T_0))")
for T in [300.0, 400.0, 500.0, 700.0]:
    R = CurrentSolver.mapa_resistencias(cf, T, SIGMA_0, ALPHA, T_0, H)
    esperado = R_REF * (1.0 + ALPHA * (T - T_0))
    val = float(R[CENTROS[0], 0])
    ok = np.isclose(val, esperado, rtol=1e-12)
    print(f"       T={T:6.1f} K -> R={val:.4f} Ohm  (esperado {esperado:.4f})")
    if not ok:
        fallos.append(f"R(T={T})")
check("R(T) sigue la ley lineal", not any(f.startswith("R(T=") for f in fallos))


# ---------------------------------------------------------------- 4
print("\n[4] Mapa de temperaturas no uniforme -> R por celda")
T_map = np.full((Ny, Nx), T_0)
T_map[CENTROS[0], :] = 700.0  # una fila caliente
T_map[CENTROS[2], :] = 500.0
R = CurrentSolver.mapa_resistencias(cf, T_map, SIGMA_0, ALPHA, T_0, H)
r_cal = float(R[CENTROS[0], 5])
r_med = float(R[CENTROS[2], 5])
r_fria = float(R[CENTROS[1], 5])
print(f"       fila 700 K -> R={r_cal:.4f} | fila 500 K -> R={r_med:.4f} | fila 300 K -> R={r_fria:.4f}")
check("celda más caliente = más resistiva", r_cal > r_med > r_fria)
check("celdas de la misma fila comparten R", bool(np.all(R[CENTROS[0], :] == r_cal)))


# ---------------------------------------------------------------- 5
print("\n[5] Mapa autodescriptivo: isfinite devuelve la geometría")
check("isfinite(R) == (cf == 1)", bool(np.array_equal(np.isfinite(R), cf == 1)))


# ---------------------------------------------------------------- 6
print("\n[6] El inf se excluye solo de la aritmética")
check("conductancia del óxido = 0", float(1.0 / R[0, 0]) == 0.0)
# Columna de un filamento: suma de conductancias sobre TODA la columna (sin máscara)
col = R[:, 7]
R_col_sin_mascara = 1.0 / np.sum(1.0 / col)
# Referencia: solo las celdas de filamento de esa columna
solo_fil = col[np.isfinite(col)]
R_col_con_mascara = 1.0 / np.sum(1.0 / solo_fil)
check(
    "sumar toda la columna == sumar solo el filamento",
    R_col_sin_mascara == R_col_con_mascara,
    f"{R_col_sin_mascara:.6f}",
)

# Matriz sin ningún filamento -> todo inf
R_vacio = CurrentSolver.mapa_resistencias(np.zeros((10, 10), dtype=int), T_0, SIGMA_0, ALPHA, T_0, H)
check("matriz sin filamento -> todo inf", bool(np.all(np.isinf(R_vacio))))
check("columna vacía -> conductancia total 0", float(np.sum(1.0 / R_vacio[:, 0])) == 0.0)


# ---------------------------------------------------------------- 7
print("\n[7] Acepta temperatura escalar (primer paso percolante)")
R_esc = CurrentSolver.mapa_resistencias(cf, 450.0, SIGMA_0, ALPHA, T_0, H)
R_mat = CurrentSolver.mapa_resistencias(cf, np.full((Ny, Nx), 450.0), SIGMA_0, ALPHA, T_0, H)
check("escalar y matriz uniforme coinciden", bool(np.array_equal(R_esc, R_mat)))


print("\n" + "=" * 60)
if fallos:
    print(f"FALLOS ({len(fallos)}): " + ", ".join(sorted(set(fallos))))
    sys.exit(1)
print("TODOS LOS TESTS PASAN")
