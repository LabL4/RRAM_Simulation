"""
Verificación del solver térmico FVM (B3 en PLAN_IMPLEMENTACION.md).

El oráculo es una copia literal del ensamblado celda a celda + `spsolve` que había antes. El solver
nuevo debe dar EXACTAMENTE los mismos bits, tanto al factorizar como al reutilizar la factorización.

Ejecutar desde la raíz del repo:
    python3 tests/test_solver_termico.py
"""

import sys
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from RRAM.thermal import fvm  # noqa: E402

PROPS = {0: {"k": 2.3}, 1: {"k": 15.0}, 2: {"k": 1e-6}, 3: {"k": 5.0}}


def oraculo(types_map, Q_map, thermal_props, atom_size, T_electrodo, matriz_muros=None):
    Ny, Nx = types_map.shape
    T_e = np.broadcast_to(np.asarray(T_electrodo, dtype=float), (Ny,))
    N = Ny * Nx
    data, rows, cols = [], [], []
    b = np.zeros(N)
    cell_volume = atom_size * atom_size
    for i in range(Ny):
        for j in range(Nx):
            n = i * Nx + j
            mat_id = types_map[i, j]
            if mat_id == 3:
                data.append(1.0); rows.append(n); cols.append(n); b[n] = T_e[i]
                continue
            if matriz_muros is not None and matriz_muros[i, j] > 0.0 and mat_id == 0:
                data.append(1.0); rows.append(n); cols.append(n); b[n] = matriz_muros[i, j]
                continue
            k_center = thermal_props[mat_id]["k"]
            diag_sum = 0.0
            vecinos = []
            if i > 0:
                vecinos.append(((i - 1, j), (i - 1) * Nx + j))
            if i < Ny - 1:
                vecinos.append(((i + 1, j), (i + 1) * Nx + j))
            if j > 0:
                vecinos.append(((i, j - 1), i * Nx + (j - 1)))
            if j < Nx - 1:
                vecinos.append(((i, j + 1), i * Nx + (j + 1)))
            for (vi, vj), vn in vecinos:
                k_vec = thermal_props[types_map[vi, vj]]["k"]
                denom = k_center + k_vec
                k_eff = 0.0 if denom == 0 else (2 * k_center * k_vec) / denom
                data.append(-k_eff); rows.append(n); cols.append(vn)
                diag_sum += k_eff
            data.append(diag_sum); rows.append(n); cols.append(n)
            b[n] = Q_map[i, j] * cell_volume
    A = coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
    return spsolve(A, b).reshape((Ny, Nx))


def caso(rng, Ny=30, Nx=12):
    cf = (rng.random((Ny, Nx)) < 0.25).astype(int)
    cf[Ny // 2, :] = 1
    types_map = fvm.crear_matriz_materiales(cf)
    Q = rng.random(types_map.shape) * 1e17
    return types_map, Q


def test_identico_bit_a_bit_al_original_con_y_sin_muro():
    rng = np.random.default_rng(0)
    for _ in range(15):
        types_map, Q = caso(rng)
        muros = np.where(rng.random(types_map.shape) < 0.05, rng.uniform(300, 600, types_map.shape), 0.0)
        for Te, m in ((300.0, None), (np.linspace(300, 400, types_map.shape[0]), muros)):
            s = fvm.SolverTermico()
            assert np.array_equal(s.resolver(types_map, Q, PROPS, 0.25e-9, Te, m), oraculo(types_map, Q, PROPS, 0.25e-9, Te, m))


def test_reutilizar_la_factorizacion_da_los_mismos_bits():
    rng = np.random.default_rng(1)
    types_map, Q = caso(rng)
    mascara = rng.random(types_map.shape) < 0.05
    s = fvm.SolverTermico()
    for rep in range(6):
        Q2 = Q * (1 + rep)
        muros = np.where(mascara, 300.0 + 20 * rep, 0.0)
        Te = 300.0 + 5 * rep
        assert np.array_equal(s.resolver(types_map, Q2, PROPS, 0.25e-9, Te, muros), oraculo(types_map, Q2, PROPS, 0.25e-9, Te, muros))
    assert (s.factorizaciones, s.reutilizaciones) == (1, 5)


def test_cambio_de_geometria_o_de_muro_refactoriza():
    rng = np.random.default_rng(2)
    types_map, Q = caso(rng)
    s = fvm.SolverTermico()
    s.resolver(types_map, Q, PROPS, 0.25e-9, 300.0)
    otro = types_map.copy()
    otro[3, 3] = 1 if otro[3, 3] == 0 else 0
    assert np.array_equal(s.resolver(otro, Q, PROPS, 0.25e-9, 300.0), oraculo(otro, Q, PROPS, 0.25e-9, 300.0))
    muros = np.zeros(types_map.shape)
    muros[types_map == 0] = 0.0
    fila = np.flatnonzero((types_map == 0).any(axis=1))[0]
    muros[fila, types_map[fila] == 0] = 450.0
    assert np.array_equal(s.resolver(otro, Q, PROPS, 0.25e-9, 300.0, muros), oraculo(otro, Q, PROPS, 0.25e-9, 300.0, muros))
    assert s.factorizaciones == 3


def test_conductividades_distintas_refactoriza():
    rng = np.random.default_rng(3)
    types_map, Q = caso(rng)
    s = fvm.SolverTermico()
    s.resolver(types_map, Q, PROPS, 0.25e-9, 300.0)
    otras = {**PROPS, 1: {"k": 20.0}}
    assert np.array_equal(s.resolver(types_map, Q, otras, 0.25e-9, 300.0), oraculo(types_map, Q, otras, 0.25e-9, 300.0))
    assert s.factorizaciones == 2


if __name__ == "__main__":
    fallos = 0
    for nombre, f in list(globals().items()):
        if nombre.startswith("test_"):
            try:
                f()
                print("ok  ", nombre)
            except Exception as ex:  # noqa: BLE001
                fallos += 1
                print("FALLA", nombre, repr(ex))
    print("Todo OK" if fallos == 0 else f"{fallos} fallos")
    sys.exit(1 if fallos else 0)
