"""
Verificación de `RRAM.grabador` (C1 en PLAN_IMPLEMENTACION.md).

Ejecutar desde la raíz del repo:
    python3 tests/test_grabador.py
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from RRAM.grabador import TIPOS_ESTADO, GrabadorEstados, RegistroDatos, columnas_datos, compactar  # noqa: E402


def test_columnas():
    assert columnas_datos(2) == ["t[s]", "V[V]", "I_total[A]", "R_total[Ohm]", "I_1[A]", "I_2[A]", "R_1[Ohm]", "R_2[Ohm]", "T_1[K]", "T_2[K]"]
    assert columnas_datos(1, con_t_ef=True)[-1] == "t_ef[s]" and len(columnas_datos(1, con_t_ef=True)) == 8


def test_registro_recorta_y_guarda_cabeceras():
    with tempfile.TemporaryDirectory() as d:
        reg = RegistroDatos(Path(d) / "Data_pp_set_1", presupuesto=9, N=1, filas_electrodo=3,
                            series_extra={"vacantes": "paso, V, n"})
        for k in range(4):
            reg.escribir(k, [k, 0.1 * k, 1e-6, np.nan, 0.0, np.nan, 300.0], np.full(3, 300.0 + k), vacantes=[k, 0.1 * k, 10 + k])
        assert reg.n_filas == 4 and reg.filas.shape == (4, 7)
        reg.guardar()
        z = np.load(Path(d) / "Data_pp_set_1.npz")
        assert list(z.files) == ["datos_sim", "header_datos_sim", "perfil_electrodo", "vacantes", "header_vacantes"]
        assert z["datos_sim"].shape == (4, 7) and np.isnan(z["datos_sim"][0, 3]), "los NaN legítimos se conservan"
        assert z["header_datos_sim"][0] == ",".join(columnas_datos(1))
        assert np.array_equal(z["perfil_electrodo"][:, 0], [300, 301, 302, 303]) and z["vacantes"][-1, 2] == 13


def test_volcar_si_falla_guarda_y_relanza():
    with tempfile.TemporaryDirectory() as d:
        reg = RegistroDatos(Path(d) / "Data_pp_reset_1.txt", presupuesto=5, N=1, filas_electrodo=2, con_t_ef=True)
        try:
            with reg.volcar_si_falla():
                for k in range(3):
                    reg.escribir(k, np.zeros(8), np.zeros(2))
                raise RuntimeError("fallo a mitad")
        except RuntimeError as e:
            assert str(e) == "fallo a mitad"
        else:
            raise AssertionError("debía relanzar")
        assert np.load(Path(d) / "Data_pp_reset_1.npz")["datos_sim"].shape == (3, 8), "se guardan las filas ya escritas"


def test_volcar_sin_fallo_no_guarda():
    with tempfile.TemporaryDirectory() as d:
        reg = RegistroDatos(Path(d) / "Data", presupuesto=2, N=1, filas_electrodo=1)
        with reg.volcar_si_falla():
            reg.escribir(0, np.zeros(7), np.zeros(1))
        assert not (Path(d) / "Data.npz").exists(), "sin fallo, el guardado lo decide la fase"


def test_grabador_cadencia_cierre_y_contenido_perezoso():
    with tempfile.TemporaryDirectory() as d:
        g = GrabadorEstados(Path(d), "pp_set", 1, cada=10)
        llamadas = []

        def contenido():
            llamadas.append(1)
            return {"actual_state": np.eye(2), "temperatura": None}

        for k in range(25):
            g.paso(k, contenido)
        g.cerrar(25, contenido)
        nombres = sorted(p.name for p in Path(d).iterdir())
        assert nombres == sorted(f"Estado_pp_set_sim_1_paso_{k}.npz" for k in (0, 10, 20, 25)), nombres
        assert len(llamadas) == 4, "el contenido solo se construye cuando se guarda"
        z = np.load(Path(d) / "Estado_pp_set_sim_1_paso_0.npz")
        assert z["temperatura"].size == 0, "None se guarda como array vacío (igual que antes)"


def test_cerrar_en_la_cadencia_no_duplica_y_cada_cero_desactiva():
    with tempfile.TemporaryDirectory() as d:
        g = GrabadorEstados(Path(d), "sp_set", 1, cada=5)
        g.paso(10, lambda: {"a": np.ones(1)})
        g.cerrar(10, lambda: {"a": np.ones(1)})
        assert g.guardados == 1
        g0 = GrabadorEstados(Path(d) / "x", "sp_set", 1, cada=0)
        g0.paso(0, lambda: 1 / 0)
        g0.cerrar(7, lambda: 1 / 0)
        assert g0.guardados == 0


def test_compactar_tipos_valores_y_casos_limite():
    rng = np.random.default_rng(0)
    estado = (rng.random((5, 4)) < 0.5).astype(np.int64)
    T = 300 + 200 * rng.random((5, 6))
    R = np.where(estado == 1, 8.9, np.inf)
    P = np.full((5, 4), 1e-40)
    c = compactar({"actual_state": estado, "temperatura": T, "mapa_resistencias": R, "probabilidad_matrix": P,
                   "cf_clean_matrix": None, "reloj_iones": np.full((5, 4), 70000, np.int32),
                   "matriz_para_plot_muro": np.array([])})
    assert c["actual_state"].dtype == np.int8 and np.array_equal(c["actual_state"], estado)
    assert c["temperatura"].dtype == np.float32 and np.max(np.abs(c["temperatura"] - T)) < 1e-4
    assert c["mapa_resistencias"].dtype == np.float32 and np.isinf(c["mapa_resistencias"][estado == 0]).all()
    assert c["probabilidad_matrix"] is P and c["probabilidad_matrix"].min() > 0, "las probabilidades no se tocan"
    assert c["cf_clean_matrix"] is None and c["matriz_para_plot_muro"].size == 0
    assert c["reloj_iones"].dtype == np.int32 and c["reloj_iones"].max() == 70000, "el reloj no se recorta a 16 bits"
    assert np.int8(1) - np.int8(0) == 1 and int(np.int8(0) - np.int8(1)) == -1, "restas de estados con signo"
    assert "aleatorios_matrix" not in TIPOS_ESTADO


def test_grabador_guarda_compactado():
    with tempfile.TemporaryDirectory() as d:
        g = GrabadorEstados(Path(d), "pp_reset", 1, cada=1)
        g.paso(0, lambda: {"actual_state": np.eye(3, dtype=np.int64), "temperatura": np.full((3, 5), 400.0)})
        z = np.load(Path(d) / "Estado_pp_reset_sim_1_paso_0.npz")
        assert z["actual_state"].dtype == np.int8 and z["temperatura"].dtype == np.float32 and z["temperatura"].shape == (3, 5)


def test_eventos_e_indice_de_pasos_guardados():
    with tempfile.TemporaryDirectory() as d:
        g = GrabadorEstados(Path(d), "pp_set", 1, cada=10)
        for k in range(23):
            if k in (7, 10):
                g.marcar_evento()
            g.paso(k, lambda: {"a": np.ones(1)})
        g.cerrar(23, lambda: {"a": np.ones(1)})
        assert g.pasos_guardados == [0, 7, 10, 20, 23], g.pasos_guardados
        g0 = GrabadorEstados(Path(d) / "x", "pp_set", 1, cada=0)
        g0.marcar_evento()
        g0.paso(3, lambda: 1 / 0)
        assert g0.pasos_guardados == [], "--sin-estados también anula los eventos"


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
