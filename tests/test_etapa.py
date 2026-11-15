"""
Verificación del bucle genérico de etapa (D5): contrato de las cuatro etapas y procesos estocásticos.

La equivalencia numérica con las fases anteriores se comprobó con una batería de 14 simulaciones
completas (ciclos, reanudaciones, roturas y fallos), idénticas bit a bit; aquí queda lo que se puede
comprobar en milisegundos.

Ejecutar desde la raíz del repo:
    python3 tests/test_etapa.py
"""

import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from RRAM import Simulation  # noqa: E402
from RRAM.acumulador_tiempo import RelojesIones  # noqa: E402
from RRAM.etapa import Etapa  # noqa: E402
from RRAM.phases_reset import EtapaPPReset, EtapaReset, EtapaSPReset, PP_reset, SP_reset  # noqa: E402
from RRAM.phases_set import EtapaPPSet, EtapaSet, EtapaSPSet, PP_set, SP_set  # noqa: E402
from RRAM.procesos_estocasticos import ProcesoGeneracion, ProcesoRecombinacion  # noqa: E402

ETAPAS = {"pp_set": EtapaPPSet, "sp_set": EtapaSPSet, "pp_reset": EtapaPPReset, "sp_reset": EtapaSPReset}
OBLIGATORIOS = ("filamentos_para_el_controlador", "campo_electrico", "rama_percolante", "rama_no_percolante",
                "actualizar_estado", "estado_a_guardar")


def test_cada_etapa_define_lo_que_el_bucle_necesita():
    for fase, cls in ETAPAS.items():
        assert cls.FASE == fase and cls.NOMBRE and cls.SUBCARPETA == fase.split("_")[1]
        for metodo in OBLIGATORIOS:
            assert getattr(cls, metodo) is not getattr(Etapa, metodo), f"{cls.__name__} no define {metodo}"


def test_un_solo_bucle():
    # Ninguna etapa redefine el bucle ni el cierre: están escritos una vez, en Etapa
    for cls in (EtapaSet, EtapaReset, *ETAPAS.values()):
        assert "ejecutar" not in vars(cls) and "_cerrar" not in vars(cls), cls.__name__
    for modulo in ("RRAM.phases_set", "RRAM.phases_reset"):
        fuente = inspect.getsource(sys.modules[modulo])
        assert "for k in range" not in fuente and "controller.next" not in fuente, modulo


def test_diferencias_set_reset():
    assert not EtapaSet.CON_T_EF and EtapaReset.CON_T_EF, "solo el RESET guarda el tiempo efectivo"
    assert set(EtapaPPSet.SERIES_EXTRA) == {"vacantes", "resistencia"} and EtapaSPSet.SERIES_EXTRA is None
    e = SimpleNamespace(num_simulation=7, rutas={"simulation_path": Path("r")})
    assert EtapaPPSet.ruta_estado_final(e).name == "Final_state_7_pp_set.npz"
    for fase in ("sp_set", "pp_reset", "sp_reset"):
        e.FASE = fase
        assert Etapa.ruta_estado_final(e).name == f"Final_state_{fase}_7.npz"


def test_las_funciones_de_fase_siguen_donde_estaban():
    assert Simulation.PP_set is PP_set and Simulation.SP_set is SP_set
    assert Simulation.PP_reset is PP_reset and Simulation.SP_reset is SP_reset
    assert list(inspect.signature(PP_set).parameters)[:7] == [
        "num_simulation", "params", "sim_ctes", "CF_ranges", "CF_creado", "protocolo", "num_pasos_guardar_estado"]
    assert list(inspect.signature(SP_reset).parameters) == [
        "final_state_pp_reset", "num_simulation", "CF_ranges", "protocolo", "num_pasos_guardar_estado", "results_dir"]


PARAMS = SimpleNamespace(paso_temporal=1e-3, atom_size=0.25e-9)
CTES = SimpleNamespace(vibration_frequency=1e13, generation_energy=1.04, cte_red=0.25e-9, gamma=10.0, gamma_drift=10.0, E_m=0.6,
                       recom_enchancement_factor=3e3, recombination_energy=1.27, long_decaimiento_concentracion=1e-9)


def test_generacion_respeta_el_cupo_y_deja_de_sortear():
    estado = np.zeros((6, 5), dtype=np.int64)
    g = ProcesoGeneracion(1.0, 1.0, max_vacantes=4, neighbor_mode="both")
    np.random.seed(0)
    nuevo, prob = g.actualizar(estado, 0, PARAMS, CTES, np.full(6, 5e9), 600.0, k=0, tiempo_efectivo=1e-3)
    assert prob is not None and 0 < nuevo.sum() <= 4 and not g.cupo_alcanzado and estado.sum() == 0
    antes = np.random.get_state()[1].copy()
    mismo, prob = g.actualizar(nuevo, 4, PARAMS, CTES, np.full(6, 5e9), 600.0, k=1, tiempo_efectivo=1e-3)
    assert mismo is nuevo and prob is None and g.cupo_alcanzado
    assert np.array_equal(antes, np.random.get_state()[1]), "con el cupo lleno no se consume ningún número aleatorio"


def test_recombinacion_guarda_su_estado():
    estado = np.zeros((6, 5), dtype=np.int64)
    estado[2, :] = 1
    r = ProcesoRecombinacion(np.zeros((6, 5), dtype=np.int8), RelojesIones(np.zeros((6, 5), dtype=np.int32)),
                             oxygen_config={0.3: 2}, velocity_thresholds={0.3: 3e-7},
                             filas_permitidas=np.ones(6, dtype=bool), max_oxigenos=3)
    assert r.probabilidad() is None, "sin pasos no hay campo de probabilidad"
    np.random.seed(1)
    for _ in range(40):
        r.empezar_paso(-0.5)
        estado = r.actualizar(estado, -0.5, np.full(6, 1e8), 500.0, PARAMS, CTES, tiempo_efectivo=1e-3)
        assert r.oxygen_state.sum() <= 3, "nunca más iones que el máximo"
        assert np.all(r.relojes.reloj[r.oxygen_state == 0] == 0), "sin ion, reloj a 0"
    assert r.oxygen_state.sum() > 0 and r.probabilidad().shape == estado.shape


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
