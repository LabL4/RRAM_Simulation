"""
Verificación de `RRAM.estado_fase` (D2): el estado entre fases tipado y su formato en disco, que no cambia.

Ejecutar desde la raíz del repo:
    python3 tests/test_estado_fase.py
"""

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from RRAM.estado_fase import CLAVES_OBSOLETAS, EstadoFase, como_estado_fase  # noqa: E402
from RRAM.persistence import cargar_estado_fase, load_phase_state, save_phase_state  # noqa: E402

CFG = SimpleNamespace(params="P", sim_ctes="C")


def test_cargan_todos_los_phase_state_antiguos_del_repo():
    ficheros = sorted(f for f in RAIZ.glob("Init_data*/phase_state_*.npz") if not f.stem.endswith("_fallida"))  # _fallida no es estado de arranque
    assert ficheros, "el repo debería traer estados guardados"
    for npz in ficheros:
        nombre = npz.stem  # phase_state_<N>_<fase>
        _, _, n, *fase = nombre.split("_")
        fase = "_".join(fase)
        e = cargar_estado_fase(fase, int(n), npz.parent, CFG)
        assert e.fase == fase and e.actual_state is not None and e.params == "P", nombre
        if fase != "sp_reset":  # fin del ciclo: SP_reset nunca ha guardado los centros
            assert e.centros is not None, f"{nombre}: centros por CF_centros o centros_calculados"
        assert not (set(e.extra) & CLAVES_OBSOLETAS), f"{nombre}: las claves obsoletas se descartan"


def test_estado_como_lo_preparan_los_notebooks():
    # Preparar_pp_reset.ipynb / crear_estado_densidad.ipynb: dict con claves antiguas + save_phase_state
    estado = {
        "actual_state": np.eye(4, dtype=int), "sim_ctes": None, "params": None,
        "Temperatura_final": np.full((4, 6), 300.0), "percola": True, "centros_calculados": [2],
        "tiempo_sp_set": 1.5, "vecindad_inicial": {"total_vacantes_zona": 4},
    }
    with tempfile.TemporaryDirectory() as d:
        save_phase_state(estado, "sp_set", 3, d)
        dict_cargado = load_phase_state("sp_set", 3, d, CFG)  # los notebooks siguen recibiendo un dict
        assert np.array_equal(dict_cargado["actual_state"], estado["actual_state"])
        assert np.allclose(dict_cargado["Temperatura_final"], 300.0)
        e = cargar_estado_fase("sp_set", 3, d, CFG)
        assert e.percola is True and e.centros == [2] and e.tiempo_fin == 1.5
        assert e.extra == {"vecindad_inicial": {"total_vacantes_zona": 4}}, e.extra
        assert e.perfil_electrodo is None and e.vacantes_generadas_set is None


def test_ida_y_vuelta_y_formato_en_disco():
    e = EstadoFase(
        fase="pp_reset", actual_state=np.ones((3, 3), int), temperatura_final=351.5, tiempo_fin=2.25,
        voltaje={"V_fin": -1.1}, centros=[1], percola=False, T_max_fils=[400.0, None], estados_guardados=[0, 10],
        oxygen_state=np.zeros((3, 3), np.int8), reloj_iones=np.zeros((3, 3), np.int32), v_prev_reloj=-1.1,
        CF_destruido=np.array([True]), voltage_CF_destruido=np.array([-0.9]),
        roturas_dict={0: {"filamento": 1, "voltaje": -0.9, "etapa": "pp_reset", "k": 5}}, max_oxigenos_reset=150,
    )
    with tempfile.TemporaryDirectory() as d:
        save_phase_state(e, "pp_reset", 1, d)
        meta = json.load(open(Path(d) / "phase_state_1_pp_reset.json"))
        assert meta["tiempo_pp_reset"] == 2.25 and meta["sistema_percola"] is False and meta["Temperatura_final"] == 351.5
        assert "creaciones_dict" not in meta, "los diccionarios de eventos vacíos no se escriben"
        r = cargar_estado_fase("pp_reset", 1, d, CFG)
        assert r.tiempo_fin == 2.25 and r.roturas_dict[0]["k"] == 5 and r.T_max_fils == [400.0, None]
        assert np.array_equal(r.CF_destruido, [True]) and r.max_oxigenos_reset == 150 and r.estados_guardados == [0, 10]


def test_alias_antiguos_y_errores_claros():
    d = {"actual_state": 1, "Temperatura_final": 2, "tiempo_pp_set": 3.0, "CF_centros": [5], "current_final": 1e-4,
         "intensidad_final": 9.9, "k_maxima": 7, "sistema_percola": True}
    e = como_estado_fase(d, "pp_set")
    assert e.centros == [5] and e.corriente_final == 1e-4 and e.percola is True and e.extra == {}
    assert como_estado_fase(e, "pp_set") is e
    for falta in ("tiempo_pp_set", "Temperatura_final"):
        try:
            EstadoFase.desde_dict({k: v for k, v in d.items() if k != falta}, "pp_set")
        except KeyError as err:
            assert falta in str(err)
        else:
            raise AssertionError(f"debía fallar sin {falta}")


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
