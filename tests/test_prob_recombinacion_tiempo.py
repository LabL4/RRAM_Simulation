"""
Verificación de cómo entra el tiempo en `Recombination.Prob_Recombination`:
el tiempo efectivo (t_ef) solo multiplica la tasa (ν·t); el exponente exp(-dt·v/L)
usa siempre el dt del paso.

Valores de referencia del caso Results_pulsos_reset_1-3_t (pulso de -0.5 V):
dt = 1e-3 s, v = 5.2e-7 m/s, T = 350 K, Er = 1.27 eV, ν = 1e13 Hz, L = 1e-9 m, factor 3000.

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


DT, V, T = 1e-3, 5.2e-7, 350.0
CTES = dict(vibration_frequency=1e13, recom_enchancement_factor=3e3, recombination_energy=1.27, long_decaimiento_concentracion=1e-9)


def prob(t_tasa, v=V, temp=T):
    return Recombination.Prob_Recombination(tiempo_tasa=t_tasa, paso_temporal=DT, velocidad=v, temp=temp, **CTES)


def prob_esperada(t_tasa, v=V, temp=T):
    pref = t_tasa * CTES["vibration_frequency"] * np.exp(-CTES["recombination_energy"] / (Recombination.k_b_ev * temp))
    return pref * np.exp(-DT * v / CTES["long_decaimiento_concentracion"]) * CTES["recom_enchancement_factor"]


# --- 1. t_ef = dt: igual que la fórmula original ---
p_dt = prob(DT)
check("t_ef = dt: valor de referencia ~9.2e-6", np.isclose(p_dt, prob_esperada(DT)) and 9.0e-6 < p_dt < 9.4e-6, f"{p_dt:.3e}")

# --- 2. t_ef > dt: escala lineal con t_ef (ya no cae en el exponente) ---
for t_ef in (0.101, 0.301):
    p = prob(t_ef)
    check(f"t_ef = {t_ef}: P = (t_ef/dt)·P(dt)", np.isclose(p, p_dt * t_ef / DT), f"{p:.3e}")
check("t_ef = 0.301: ya no colapsa (> 1e-3)", prob(0.301) > 1e-3, f"{prob(0.301):.3e}")

# --- 3. Velocidad 0 (V = 0): exponente = 1 ---
check("v = 0: exponente = 1", np.isclose(prob(DT, v=0.0), prob_esperada(DT, v=0.0)), f"{prob(DT, v=0.0):.3e}")

# --- 4. Funciona con mapas (temperatura y velocidad por celda) ---
temps = np.array([[300.0, 350.0], [340.0, 333.0]])
vels = np.array([[0.0, V], [V, 3e-7]])
p_mapa = Recombination.Prob_Recombination(tiempo_tasa=0.301, paso_temporal=DT, velocidad=vels, temp=temps, **CTES)
check("mapas: misma forma y valores", p_mapa.shape == (2, 2) and np.allclose(p_mapa, prob_esperada(0.301, v=vels, temp=temps)))

# --- 5. Recombine_opt pasa los dos tiempos ---
np.random.seed(0)
vac = np.ones((50, 50), dtype=int)
oxy = np.ones((50, 50), dtype=np.int8)
# Con t_tasa enorme la probabilidad satura a 1: todas las coincidencias recombinan
vac2, oxy2 = Recombination.Recombine_opt(vac, oxy, tiempo_tasa=1e12, paso_temporal=DT, velocidad=V, temperatura=T, **CTES)
check("Recombine_opt: P saturada -> recombinan todas", vac2.sum() == 0 and oxy2.sum() == 0)
vac3, oxy3 = Recombination.Recombine_opt(vac, oxy, tiempo_tasa=DT, paso_temporal=DT, velocidad=V, temperatura=T, **CTES)
check("Recombine_opt: P ~1e-5 -> casi ninguna", (vac.size - vac3.sum()) <= 3, f"recombinadas={vac.size - vac3.sum()}")

print()
if fallos:
    print(f"{len(fallos)} FALLO(S): {fallos}")
    sys.exit(1)
print("Todos los checks OK")
