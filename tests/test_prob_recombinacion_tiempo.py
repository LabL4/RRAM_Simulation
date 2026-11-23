"""
Verificación de `Recombination.Prob_Recombination` y `Recombination.Recombine_opt`:

    P = ν · t · exp(-E_r / (k_B·T)) · β · g(t),   g(t) = 1 + (G - 1) · t / (t + τ_c)

con t = reloj de igual voltaje · dt (RelojesIones.reloj): el mismo tiempo en la tasa y en g.
Valores: ν = 1e13 Hz, E_r = 1.27 eV, β = 3000, G = 1.5, τ_c = 400 µs, dt = 1e-6 s, T = 350 K.

Ejecutar desde la raíz del repo:
    python3 tests/test_prob_recombinacion_tiempo.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from RRAM import Recombination  # noqa: E402

fallos = []


def check(nombre, cond, detalle=""):
    print(f"{'  OK  ' if cond else ' FALLO'} | {nombre}" + (f"  ({detalle})" if detalle else ""))
    if not cond:
        fallos.append(nombre)


DT, T, G, TAU_C = 1e-6, 350.0, 1.5, 4e-4
CTES = dict(
    vibration_frequency=1e13,
    recom_enchancement_factor=3e3,
    recombination_energy=1.27,
    realce_max=G,
    tau_c=TAU_C,
)


def prob(t, temp=T):
    return Recombination.Prob_Recombination(tiempo_tasa=t, temp=temp, **CTES)


def prob_base(t, temp=T):
    """P sin realce (g = 1)."""
    arr = np.exp(-CTES["recombination_energy"] / (Recombination.k_b_ev * temp))
    return t * CTES["vibration_frequency"] * arr * CTES["recom_enchancement_factor"]


def g(t):
    return 1 + (G - 1) * t / (t + TAU_C)


# --- 1. t = 0 (ion recién llegado o recién creado): P = 0 ---
check("t = 0: P = 0", prob(0.0) == 0.0)

# --- 2. t = dt (un paso, también V = 0): g casi 1 ---
p_dt = prob(DT)
check("t = dt: g = 1 + 0.5·dt/(dt+τ_c) ≈ 1.00125", np.isclose(p_dt / prob_base(DT), g(DT)) and abs(g(DT) - 1.00125) < 1e-5, f"g = {p_dt / prob_base(DT):.6f}")
check("t = dt: valor de referencia ~1.55e-8", 1.5e-8 < p_dt < 1.6e-8, f"{p_dt:.3e}")

# --- 3. Forma de g(t) ---
check("t = τ_c: g = (1 + G)/2 = 1.25", np.isclose(prob(TAU_C) / prob_base(TAU_C), 1.25))
check("t = 481 µs: g ≈ 1.273", np.isclose(prob(481e-6) / prob_base(481e-6), 1 + 0.5 * 481 / 881))
check("t >> τ_c: g → G", np.isclose(prob(1e3) / prob_base(1e3), G, rtol=1e-5))
gs = [prob(t) / prob_base(t) for t in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2)]
check("g crece con t y nunca baja de 1", all(np.diff(gs) > 0) and min(gs) >= 1.0, str(np.round(gs, 4)))

# --- 4. P crece más que linealmente en t (por g) ---
check("P(300·dt) = 300·P_base(dt)·g(300·dt)", np.isclose(prob(300 * DT), 300 * prob_base(DT) * g(300 * DT)))
check("P(300·dt) > 300·P(dt)", prob(300 * DT) > 300 * p_dt)

# --- 5. Celda a celda (t y T como matrices) ---
t_m = np.array([[0.0, 1e-6], [2e-4, 4.8e-4]])
temp_m = np.array([[300.0, 350.0], [340.0, 333.0]])
p_m = prob(t_m, temp_m)
check("matrices: misma forma y valores", p_m.shape == (2, 2) and np.allclose(p_m, prob_base(t_m, temp_m) * g(t_m)))
check("matrices con T escalar", np.allclose(prob(t_m), prob_base(t_m) * g(t_m)))

# --- 6. Recombine_opt sortea con la matriz recibida ---
np.random.seed(0)
vac = np.ones((50, 50), dtype=int)
oxy = np.ones((50, 50), dtype=np.int8)
vac2, oxy2 = Recombination.Recombine_opt(vac, oxy, np.ones((50, 50)))
check("Recombine_opt: P = 1 -> recombinan todas", vac2.sum() == 0 and oxy2.sum() == 0)
vac3, oxy3 = Recombination.Recombine_opt(vac, oxy, np.zeros((50, 50)))
check("Recombine_opt: P = 0 -> ninguna", vac3.sum() == vac.size and oxy3.sum() == oxy.size)

# Solo donde coinciden ion y vacante: P = 1 en toda la matriz, pero sin ion o sin vacante no cambia nada
vac4 = np.zeros((4, 4), dtype=int)
oxy4 = np.zeros((4, 4), dtype=np.int8)
vac4[0, 0] = oxy4[0, 0] = 1  # coinciden
vac4[1, 1] = 1  # vacante sola
oxy4[2, 2] = 1  # ion solo
vac5, oxy5 = Recombination.Recombine_opt(vac4, oxy4, np.ones((4, 4)))
check(
    "Recombine_opt: solo recombina donde coinciden ion y vacante",
    vac5[0, 0] == 0 and oxy5[0, 0] == 0 and vac5[1, 1] == 1 and oxy5[2, 2] == 1,
)
check("Recombine_opt: no modifica las entradas", vac4[0, 0] == 1 and oxy4[0, 0] == 1)

print()
if fallos:
    print(f"{len(fallos)} FALLO(S): {fallos}")
    sys.exit(1)
print("Todos los checks OK")
