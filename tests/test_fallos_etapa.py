"""
Verificación de lo que se guarda cuando una fase se para por no formar o no destruir el filamento.

Ejecutar desde la raíz del repo:
    python3 tests/test_fallos_etapa.py
"""

import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from RRAM import exceptions  # noqa: E402
from RRAM.persistence import load_phase_state  # noqa: E402
from RRAM.run_cycle import SUFIJO_FASE_FALLIDA, _guardar_estado_fase_fallida  # noqa: E402

@dataclass
class _Config:
    """Como SimulationParameters/SimulationConstants: se reconstruyen del CSV, no se guardan."""

    x: float = 1.0


ESTADO_FASE = {
    "actual_state": np.eye(4, dtype=int),
    "voltaje_percolacion": 0.795,
    "creaciones_dict": {0: {"filamento": 1, "voltaje": 0.795, "etapa": "pp_set", "k": 159}},
    "T_max_fils": [344.1, None],
    "params": _Config(),
}


def test_las_excepciones_de_formacion_y_destruccion_transportan_el_estado_de_la_fase():
    est = np.zeros((3, 3), dtype=int)
    casos = [
        exceptions.NoPercolationException(estado_fase=ESTADO_FASE),
        exceptions.FilamentosNoFormadosException("x", 1, est, 1, 2, estado_fase=ESTADO_FASE),
        exceptions.FilamentosNoDestruidosException("x", "x/Figures", -1.0, 1, est, 0, 1, estado_fase=ESTADO_FASE),
    ]
    for e in casos:
        assert isinstance(e, exceptions.ExcepcionConEstado) and e.estado_fase is ESTADO_FASE, type(e).__name__


def test_sin_estado_de_fase_por_defecto():
    assert exceptions.NoPercolationException().estado_fase is None
    assert exceptions.NoPercolationException().ruta_estado_fallo is None, "no percolar no deja fichero de estado propio"


def test_percolacion_sin_filamento_guarda_su_estado():
    e = exceptions.PercolacionSinFilamentoException("sim", 0.7, 2, np.ones((2, 2)))
    assert e.ruta_estado_fallo.name == "simulation_2_percolacion_sin_filamento.npz" and e.estado_fase is None


def test_variables_de_la_fase_fallida_se_guardan_con_sufijo_y_no_se_reanudan():
    with tempfile.TemporaryDirectory() as d:
        _guardar_estado_fase_fallida(ESTADO_FASE, "pp_set", 1, d)
        base = Path(d) / f"phase_state_1_pp_set{SUFIJO_FASE_FALLIDA}"
        z = np.load(str(base) + ".npz")
        meta = json.load(open(str(base) + ".json"))
        assert np.array_equal(z["actual_state"], ESTADO_FASE["actual_state"])
        assert meta["voltaje_percolacion"] == 0.795 and meta["T_max_fils"] == [344.1, None] and "params" not in meta
        cfg = SimpleNamespace(params=None, sim_ctes=None)
        try:
            load_phase_state("pp_set", 1, d, cfg)
        except FileNotFoundError:
            return
        raise AssertionError("--start-from no debe encontrar el estado de una fase fallida")


def test_valor_no_serializable_no_deja_ficheros_a_medias():
    from RRAM.persistence import save_phase_state

    with tempfile.TemporaryDirectory() as d:
        try:
            save_phase_state({"actual_state": np.eye(2), "raro": object()}, "pp_set", 1, d)
        except TypeError:
            pass
        else:
            raise AssertionError("debía lanzar TypeError")
        assert list(Path(d).iterdir()) == [], "no debe quedar ni el .npz ni un .json truncado"


def test_un_fallo_al_guardar_no_tapa_el_error_original():
    _guardar_estado_fase_fallida({"x": object()}, "pp_set", 1, "/ruta/que/no/existe/\0")  # no debe lanzar


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
