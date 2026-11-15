"""
Verificación de las optimizaciones del bloque B (ver PLAN_IMPLEMENTACION.md).

Ejecutar desde la raíz del repo:
    python3 tests/test_bloque_B.py
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from RRAM import ElectricField, Recombination, state_updates  # noqa: E402
from RRAM.acumulador_tiempo import RelojesIones  # noqa: E402


def test_campo_vectorizado_identico_bit_a_bit_al_escalar():
    rng = np.random.default_rng(0)
    for _ in range(50):
        estado = (rng.random((120, 40)) < rng.random()).astype(np.int64)
        V = float(rng.uniform(-2, 2))
        esperado = np.array(
            [ElectricField.GapElectricField(V, i, estado, device_size_x=10e-9, grid_size=0.25e-9) for i in range(120)]
        )
        obtenido = ElectricField.GapElectricFieldVector(V, estado, 10e-9, 0.25e-9)
        assert obtenido.dtype == np.float64 and np.array_equal(esperado, obtenido)


def test_campo_vectorizado_fila_llena_L_cero():
    # 4 celdas de 0.5 en un dispositivo de 2.0: una fila llena anula el hueco (L == 0)
    estado = np.array([[1, 1, 1, 1], [1, 0, 1, 0], [0, 0, 0, 0]])
    for V in (1.5, -1.5):
        esperado = np.array([ElectricField.GapElectricField(V, i, estado, 2.0, 0.5) for i in range(3)])
        obtenido = ElectricField.GapElectricFieldVector(V, estado, 2.0, 0.5)
        assert np.array_equal(esperado, obtenido), (V, esperado, obtenido)
        assert np.isfinite(obtenido).all()


def test_simple_electric_field_sin_njit_mismo_valor():
    assert ElectricField.SimpleElectricField(-1.1, 10e-9) == abs(-1.1 / 10e-9)


def test_campo_probabilidad_recombinacion_diferido_igual_al_directo():
    params = SimpleNamespace(paso_temporal=1e-3, atom_size=0.25e-9, device_size_x=10e-9)
    ctes = SimpleNamespace(
        vibration_frequency=1e13, gamma_drift=10.0, E_m=0.6, cte_red=0.25e-9, recom_enchancement_factor=3e3,
        recombination_energy=1.27, long_decaimiento_concentracion=1e-9,
    )
    estado = np.zeros((6, 5), dtype=np.int64)
    estado[2, :] = 1
    oxigeno = np.zeros((6, 5), dtype=np.int8)
    temperatura = np.full((6, 5), 500.0)
    relojes = RelojesIones(np.zeros((6, 5), dtype=np.int32))
    np.random.seed(1)
    _, _, campo = state_updates.update_state_recombinate(
        voltage=-0.5, E_field_filas=np.full(6, 1e8), oxygen_config={0.3: 1}, sim_ctes=ctes, params=params,
        actual_state=estado, oxygen_state=oxigeno, temperatura=temperatura, velocity_thresholds={0.3: 3e-7},
        relojes=relojes, filas_permitidas=np.ones(6, dtype=bool), max_oxigenos=10, tiempo_efectivo=2e-3,
    )
    assert callable(campo), "debe devolverse sin evaluar"
    esperado = np.minimum(
        Recombination.Prob_Recombination(
            tiempo_tasa=2e-3, paso_temporal=1e-3, velocidad=3e-7, temp=temperatura, vibration_frequency=1e13,
            recom_enchancement_factor=3e3, recombination_energy=1.27, long_decaimiento_concentracion=1e-9,
        ),
        1.0,
    )
    obtenido = state_updates.evaluar_campo_probabilidad(campo)
    assert obtenido.shape == estado.shape and np.array_equal(obtenido, esperado)
    assert np.array_equal(campo(), obtenido), "evaluarlo dos veces da lo mismo"


def test_evaluar_campo_none_devuelve_none():
    assert state_updates.evaluar_campo_probabilidad(None) is None


def test_import_del_paquete_perezoso():
    import importlib
    import subprocess

    codigo = "import sys, RRAM; assert 'matplotlib' not in sys.modules and 'RRAM.Temperature' not in sys.modules; RRAM.Temperature; assert 'RRAM.Temperature' in sys.modules"
    r = subprocess.run([sys.executable, "-W", "ignore", "-c", codigo], capture_output=True, text=True, cwd=Path(__file__).resolve().parent.parent)
    assert r.returncode == 0, r.stderr[-300:]
    importlib.invalidate_caches()


def test_exec_no_carga_matplotlib():
    import subprocess

    codigo = "import sys, RRAM.__main__, RRAM.run_cycle; assert 'matplotlib' not in sys.modules, 'matplotlib cargado'"
    r = subprocess.run([sys.executable, "-W", "ignore", "-c", codigo], capture_output=True, text=True, cwd=Path(__file__).resolve().parent.parent)
    assert r.returncode == 0, r.stderr[-300:]


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
