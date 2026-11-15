"""
Verificación de las correcciones del bloque A (ver PLAN_IMPLEMENTACION.md).

Ejecutar desde la raíz del repo:
    python3 tests/test_bloque_A.py
"""

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from RRAM import exceptions, Temperature, utils  # noqa: E402
from RRAM.run_cycle import _guardar_estado_fallo  # noqa: E402

FASES = ("pp_set", "sp_set", "pp_reset", "sp_reset")


def test_semillas_distintas_por_fase_y_deterministas():
    s = [utils.semilla_de_fase(123, f) for f in FASES]
    assert len(set(s)) == 4, "las cuatro fases deben tener semillas distintas"
    assert s == [utils.semilla_de_fase(123, f) for f in FASES], "deben ser deterministas"
    assert s != [utils.semilla_de_fase(124, f) for f in FASES], "dependen de la semilla de la corrida"


def test_crear_rng_reproduce_por_fase_y_no_toca_el_estado_global():
    p = SimpleNamespace(seed=7)
    a = utils.crear_rng(p, "sp_set").rand(3)
    assert np.array_equal(a, utils.crear_rng(p, "sp_set").rand(3))
    assert not np.array_equal(a, utils.crear_rng(p, "pp_reset").rand(3)), "otra fase no debe repetir la secuencia"
    # Misma secuencia que la semilla global de antes (los resultados con semilla no cambian)
    np.random.seed(utils.semilla_de_fase(7, "sp_set"))
    assert np.array_equal(a, np.random.rand(3))
    np.random.seed(5)
    esperado = np.random.rand(2)
    np.random.seed(5)
    utils.crear_rng(SimpleNamespace(seed=7), "pp_set").rand(4)
    utils.crear_rng(SimpleNamespace(seed=None), "pp_set").rand(4)
    assert np.array_equal(esperado, np.random.rand(2)), "crear_rng no debe tocar np.random"


def test_muro_para_plot_none_si_no_hubo_muro():
    assert Temperature.componer_matriz_muro_plot(None, [3], [np.ones(4)]) is None
    assert Temperature.componer_matriz_muro_plot(np.zeros((6, 4)), [3], None) is None


def test_muro_para_plot_pone_el_perfil_en_la_fila_central_sin_modificar_la_entrada():
    muros = np.zeros((6, 4))
    out = Temperature.componer_matriz_muro_plot(muros, [1, 4, None], [np.full(4, 9.0), np.full(4, 7.0), None])
    assert np.all(out[1] == 9.0) and np.all(out[4] == 7.0) and out[0].sum() == 0
    assert muros.sum() == 0, "la matriz de entrada no debe modificarse"


def test_excepciones_no_escriben_al_construirse():
    with tempfile.TemporaryDirectory() as d:
        base = Path(d) / "no_existe_aun"
        estado = np.ones((4, 4), dtype=int)
        exceptions.FilamentosNoDestruidosException(base, base / "Figures", 0.5, 3, estado, 0, 1)
        exceptions.FilamentosNoFormadosException(base, 3, estado, 0, 1)
        exceptions.NullResistanceException(base, 0.5, 3, estado)
        assert not base.exists(), "construir la excepción no debe crear carpetas ni ficheros"


def test_num_simulation_no_se_incrementa_de_nuevo():
    e = exceptions.NullResistanceException("x", 0.1, 3, np.zeros((2, 2)))
    assert e.num_simulation == 3


def test_guardar_estado_fallo_escribe_npz_verdadero():
    estado = (np.random.rand(5, 5) > 0.5).astype(int)
    with tempfile.TemporaryDirectory() as d:
        e = exceptions.FilamentosNoFormadosException(Path(d) / "sim", 2, estado, 0, 1)
        _guardar_estado_fallo(e)
        z = np.load(e.ruta_estado_fallo)  # sin allow_pickle: debe ser un npz de verdad
        assert np.array_equal(z["actual_state"], estado)


def test_guardar_estado_fallo_ignora_excepciones_sin_estado():
    _guardar_estado_fallo(ValueError("sin estado"))  # no debe lanzar


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
