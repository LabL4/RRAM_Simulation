from pathlib import Path
import numpy as np
import sys

sys.path.insert(0, "/Users/antonio_lopez_torres/Documents/GitHub/RRAM_Simulation")
from RRAM import CurrentSolver

atom_size = 0.25e-9
sigma_0 = 930232558.1395348
T0 = 300.0
alpha_T = 2e-3

print("=" * 70)
print("CASO 1: filamento rectangular regular, w=5, Nx=10 columnas")
print("=" * 70)
Ny, Nx, w = 12, 10, 5
cf = np.zeros((Ny, Nx), dtype=int)
cf[3 : 3 + w, :] = 1  # w filas ocupadas en TODAS las columnas

R_local = CurrentSolver.mapa_resistencias(cf, T0, sigma_0, alpha_T, T0, atom_size)
R_codigo = CurrentSolver.calcular_resistencia(R_local)

# Cálculo manual EXACTO según la descripción del usuario:
# - columna: w celdas en paralelo, todas con la misma R_celda -> R_col = R_celda / w
# - entre columnas: Nx columnas en serie -> R_total = Nx * R_col
R_celda = 1.0 / (sigma_0 * atom_size)
R_col_manual = R_celda / w
R_manual = Nx * R_col_manual

print(f"R_celda (una celda)      = {R_celda:.6f} Ohm")
print(f"R_col manual (paralelo)  = {R_col_manual:.6f} Ohm")
print(f"R_total manual (serie)   = {R_manual:.10f} Ohm")
print(f"R_total del código       = {R_codigo:.10f} Ohm")
print(f"Coinciden exactamente:     {np.isclose(R_manual, R_codigo, rtol=1e-14)}")

L = Nx * atom_size
R_formula_cerrada = L / (sigma_0 * atom_size**2 * w)
print(f"R_total formula cerrada  = {R_formula_cerrada:.10f} Ohm  (L/(sigma_0*atom_size^2*w))")
print(f"Coincide con el código:    {np.isclose(R_formula_cerrada, R_codigo, rtol=1e-14)}")

print()
print("=" * 70)
print("CASO 2: filamento IRREGULAR (w distinto por columna)")
print("=" * 70)
Ny, Nx = 12, 6
cf = np.zeros((Ny, Nx), dtype=int)
anchos = [3, 4, 5, 5, 2, 6]  # w por columna, deliberadamente irregular
for j, w_j in enumerate(anchos):
    cf[3 : 3 + w_j, j] = 1

R_local = CurrentSolver.mapa_resistencias(cf, T0, sigma_0, alpha_T, T0, atom_size)
R_codigo = CurrentSolver.calcular_resistencia(R_local)

# Manual: cada columna j en paralelo con w_j celdas -> R_col_j = R_celda/w_j
# luego las columnas en serie -> suma
R_manual = sum(R_celda / w_j for w_j in anchos)

print(f"anchos por columna        = {anchos}")
print(f"R_total manual (serie de paralelos por columna) = {R_manual:.10f} Ohm")
print(f"R_total del código                               = {R_codigo:.10f} Ohm")
print(f"Coinciden exactamente:     {np.isclose(R_manual, R_codigo, rtol=1e-14)}")

# La fórmula cerrada con "w total" NO debería coincidir aquí (para demostrar
# que la fórmula cerrada solo vale para filamento regular)
w_promedio = np.mean(anchos)
L = Nx * atom_size
R_formula_cerrada_mal_aplicada = L / (sigma_0 * atom_size**2 * w_promedio)
print(f"R con fórmula cerrada usando w medio (INCORRECTO en este caso) = {R_formula_cerrada_mal_aplicada:.10f} Ohm")
print(f"Diferencia relativa con el código: {abs(R_formula_cerrada_mal_aplicada - R_codigo) / R_codigo * 100:.2f} %")
