"""
Batería de regresión del simulador: 14 simulaciones completas cuya salida se compara fichero a fichero.

Sirve para comprobar que un cambio que no toca la física ni el formato deja los resultados
IDÉNTICOS (datos, estados intermedios, estados de fase y metadata).

    python3 tests/bateria/bateria.py ejecutar base            # antes del cambio (~2 min)
    ...cambio...
    python3 tests/bateria/bateria.py ejecutar nuevo
    python3 tests/bateria/bateria.py comparar base nuevo      # debe acabar en "TODO IDÉNTICO"

`ejecutar <etiqueta> [--raiz CARPETA]` simula con el paquete RRAM de CARPETA (por defecto, este repo)
y deja la salida en tests/bateria/_trabajo/salidas/<etiqueta>/. Con --raiz se puede ejecutar una
copia anterior del paquete para compararla con la actual.

Los casos son autocontenidos (la configuración está en este fichero) y cada uno comprueba que acaba
como se espera, para que la batería siga cubriendo todas las ramas:

    ciclo1, ciclo2                    ciclo con 1 y 2 filamentos (muro térmico). El RESET no rompe el filamento:
                                      con los umbrales de oxígeno base (1.1 y 1.15 V) no se generan iones.
    inicio_sp_set, inicio_pp_reset    --start-from desde los estados que deja ciclo2
    sin_estados, sin_muro             --sin-estados, --no-muro
    rotura, rotura2                   PP_reset que rompe 1 y 2 filamentos, y SP_reset completo
    sp_reset_ok, sp_reset_percola     SP_reset desde un estado preparado; rechazo si aún percola
    caso_nopercola, caso_noformados   fallos de PP_set (no percola / se forma 1 filamento de 2)
    fallo_mitad                       condición de voltaje exigida que no se cumple: parada a mitad de bucle
    cupo                              cupo de vacantes lleno antes de percolar (corriente congelada)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
REPO = AQUI.parent.parent
TRABAJO = AQUI / "_trabajo"
SEMILLA = "123"

PARAMS_BASE = {
    "device_size_x": 1e-08, "device_size_y": 3e-08, "atom_size": 2.5e-10, "num_trampas": 100,
    "paso_temporal": 1e-3, "init_temp": 300.0, "densidad_vacantes": 4.0,
}
CTES_BASE = {
    "cte_red": 2.5e-10, "permitividad_relativa_set": 299.977, "permitividad_relativa_reset": 399.998,
    "generation_energy": 1.04, "recombination_energy": 1.27, "pb_metal_insul_set": 0.0026, "pb_metal_insul_reset": 0.0307,
    "recom_enchancement_factor": 3000.0, "long_decaimiento_concentracion": 1e-09, "sigma_0": 450000000.0, "alpha_T": 0.002,
    "delta_z_r": 2.5e-10, "num_filamentos": 1, "grosor_filamento": "[3]", "grosor_filamento_init": None,
    "vibration_frequency": 1e13, "gamma": 10.0, "gamma_drift": 10.0, "E_m": 0.6, "I_0_set": 0.00029806, "I_0_reset": 0.00072588,
    "r_termica_no_percola": 8000, "conductividad_termica_dielectrico": 2.3, "conductividad_termica_CF": 15.0,
    "conductividad_termica_aislante": 1e-06, "conductividad_termica_electrodo": 5.0, "Temperatura_electrodo": 300,
    "factor_generar_calor": 6e-05, "pendiente_temperatura": -28, "k_amortiguacion_termica": 1.0, "umbral_caida_temperatura": 1,
    "k_electrodo_enfriamiento": 2.0, "umbral_caida_electrodo": 0.1, "ocupacion_max_pp_set": 0.99, "ocupacion_max_sp_set": 0.0,
    "pasos_guardar_estado": 10, "factor_vecinos_pp_set": 1.0, "factor_libre_pp_set": 1.0, "factor_vecinos_sp_set": 1.0,
    "factor_libre_sp_set": 0.9, "neighbor_mode_pp_set": "both", "neighbor_mode_sp_set": "both",
    "voltaje_gen_oxigeno_pp_1": 1.1, "num_oxigenos_pp_reset_1": 7, "voltaje_gen_oxigeno_pp_2": 1.15, "num_oxigenos_pp_reset_2": 4,
    "voltaje_gen_oxigeno_sp": -0.2, "num_oxigenos_sp_reset": 5, "centros_filamento": None,
}
DOS_FILAMENTOS = {"num_filamentos": 2, "grosor_filamento": "[3, 3]"}
#: Umbrales con los que el RESET sí genera oxígeno (los de simulation_config.py)
OXIGENO_ACTIVO = {"voltaje_gen_oxigeno_pp_1": 0.3, "voltaje_gen_oxigeno_pp_2": 0.45, "voltaje_gen_oxigeno_sp": -0.2}


def protocolo(pp_set=None, pp_reset=None, sp_reset=None) -> dict:
    return {
        "pp_set": pp_set or [("rampa", {"desde": 0.0, "hasta": 1.1, "dV": 0.005})],
        "sp_set": [("constante", {"V": "anterior", "pasos": 50})],
        "pp_reset": pp_reset or [("rampa", {"desde": 0.0, "hasta": -1.1, "dV": 0.005})],
        "sp_reset": sp_reset or [("constante", {"V": "anterior", "pasos": 50})],
    }


def reset_con_meseta(pasos: int) -> dict:
    return protocolo(
        pp_reset=[("rampa", {"desde": 0.0, "hasta": -1.1, "dV": 0.01}), ("constante", {"V": -1.1, "pasos": pasos})],
        sp_reset=[("constante", {"V": "anterior", "pasos": 60}), ("rampa", {"desde": "anterior", "hasta": 0.0, "dV": 0.02})],
    )


class Bateria:
    def __init__(self, etiqueta: str, raiz: Path):
        self.raiz = Path(raiz).resolve()
        self.casos = TRABAJO / "casos"  # misma ruta para todas las etiquetas: los resultados no dependen de ella
        self.salida = TRABAJO / "salidas" / etiqueta
        for d in (self.casos, self.salida):
            shutil.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True)
        self.resumen: list[tuple[str, int, str]] = []

    # ------------------------------------------------------------------ utilidades

    def _rram(self, carpeta: Path, *args: str) -> int:
        env = {**os.environ, "PYTHONPATH": str(self.raiz), "MPLBACKEND": "Agg"}
        r = subprocess.run([sys.executable, "-W", "ignore", "-m", "RRAM", *args], cwd=carpeta, env=env, capture_output=True, text=True)
        return r.returncode

    def crear(self, nombre: str, ctes: dict | None = None, proto: dict | None = None, copiar_estados_de: str | None = None) -> Path:
        """Carpeta del caso con sus CSV y su estado inicial (sorteado con la semilla de la batería)."""
        d = self.casos / nombre
        (d / "Init_data").mkdir(parents=True)
        pd.DataFrame([PARAMS_BASE]).to_csv(d / "Init_data/simulation_parameters.csv", index=False)
        pd.DataFrame([{**CTES_BASE, **(ctes or {})}]).to_csv(d / "Init_data/simulation_constants.csv", index=False)
        pd.DataFrame({"protocolo_voltaje": [repr(proto or protocolo())]}).to_csv(d / "Init_data/simulation_voltage.csv", index=False)
        assert self._rram(d, "init", "--seed", SEMILLA, "--init-data-dir", "Init_data") == 0, f"{nombre}: init falló"
        if copiar_estados_de:
            for f in (self.casos / copiar_estados_de / "Init_data").glob("phase_state_1_*"):
                if "_fallida" not in f.name:
                    shutil.copy(f, d / "Init_data")
        return d

    def ejecutar(self, nombre: str, *args: str, espera: str, carpeta: Path | None = None) -> None:
        """Simula el caso, recoge su salida y comprueba que termina como se espera (`espera` ⊂ "estado · error")."""
        d = carpeta or self.casos / nombre
        shutil.rmtree(d / "Results", ignore_errors=True)
        shutil.rmtree(d / "logs", ignore_errors=True)
        rc = self._rram(d, "exec", "0", "--seed", SEMILLA, "--init-data-dir", "Init_data", "--results-dir", "Results", *args)
        out = self.salida / nombre
        if (d / "Results/simulation_1").is_dir():
            shutil.copytree(d / "Results/simulation_1", out)
        (out / "_phase_states").mkdir(parents=True, exist_ok=True)
        for f in (d / "Init_data").glob("phase_state_1_*"):
            shutil.copy(f, out / "_phase_states")
        (out / "_rc.json").write_text(json.dumps(rc))
        estado = "sin metadata"
        if (out / "sim_metadata_1.json").exists():
            e = json.loads((out / "sim_metadata_1.json").read_text())["extra"]
            estado = e["status"] + (" · " + e["error"].split(":")[0] if "error" in e else "")
        assert espera in estado, f"{nombre}: se esperaba '{espera}' y terminó como '{estado}' (log en {d / 'logs'})"
        self.resumen.append((nombre, rc, estado))
        print(f"  {nombre:18s} rc={rc}  {estado}", flush=True)

    def _editar_npz(self, ruta: Path, **cambios) -> None:
        z = dict(np.load(ruta))
        z.update(cambios)
        np.savez_compressed(ruta, **z)

    # ------------------------------------------------------------------ casos

    def todos(self) -> None:
        # --- ciclos completos
        self.crear("ciclo1")
        self.ejecutar("ciclo1", espera="failed_at_pp_reset · FilamentosNoDestruidosException")
        self.crear("ciclo2", ctes=DOS_FILAMENTOS)
        self.ejecutar("ciclo2", espera="failed_at_pp_reset · FilamentosNoDestruidosException")

        # --- reanudaciones y opciones, sobre la configuración de ciclo2
        for inicio in ("sp_set", "pp_reset"):
            self.crear(f"inicio_{inicio}", ctes=DOS_FILAMENTOS, copiar_estados_de="ciclo2")
            self.ejecutar(f"inicio_{inicio}", "--start-from", inicio, espera="failed_at_pp_reset")
        for nombre, opcion in (("sin_estados", "--sin-estados"), ("sin_muro", "--no-muro")):
            self.crear(nombre, ctes=DOS_FILAMENTOS)
            self.ejecutar(nombre, opcion, espera="failed_at_pp_reset")

        # --- fallos de PP_set
        self.crear("caso_nopercola", ctes=DOS_FILAMENTOS, proto=protocolo(pp_set=[("rampa", {"desde": 0.0, "hasta": 0.7, "dV": 0.005})]))
        self.ejecutar("caso_nopercola", "--stop-at", "pp_set", espera="failed_at_pp_set · NoPercolationException")
        self.crear("caso_noformados", ctes=DOS_FILAMENTOS, proto=protocolo(pp_set=[("rampa", {"desde": 0.0, "hasta": 0.8, "dV": 0.005})]))
        self.ejecutar("caso_noformados", "--stop-at", "pp_set", espera="failed_at_pp_set · FilamentosNoFormadosException")
        self.crear("fallo_mitad", ctes=DOS_FILAMENTOS, proto=protocolo(pp_set=[
            ("rampa", {"desde": 0.0, "hasta": 0.3, "dV": 0.005, "hasta_percolacion": True, "exigir": True}),
            ("rampa", {"desde": "anterior", "hasta": 1.1, "dV": 0.005}),
        ]))
        self.ejecutar("fallo_mitad", "--stop-at", "pp_set", espera="failed_at_pp_set · CondicionExigidaNoCumplida")
        self.crear("cupo", ctes={"ocupacion_max_pp_set": 0.0231})  # cupo de 105 vacantes: se llena sin percolar
        self.ejecutar("cupo", "--stop-at", "pp_set", espera="failed_at_pp_set · NoPercolationException")

        # --- RESET con rotura: un filamento fino (desde el sp_set de ciclo1) y dos (desde el de ciclo2)
        d = self.crear("rotura", ctes=OXIGENO_ACTIVO, proto=reset_con_meseta(400), copiar_estados_de="ciclo1")
        npz = d / "Init_data/phase_state_1_sp_set.npz"
        fino = np.zeros_like(np.load(npz)["actual_state"])
        fino[59, :] = 1
        fino[58, :15] = 1
        fino[60, 25:] = 1
        fino[10:14, 3:9] = 1
        self._editar_npz(npz, actual_state=fino)
        self.ejecutar("rotura", "--start-from", "pp_reset", espera="completed")
        self.crear("rotura2", ctes={**DOS_FILAMENTOS, **OXIGENO_ACTIVO}, proto=reset_con_meseta(1500), copiar_estados_de="ciclo2")
        self.ejecutar("rotura2", "--start-from", "pp_reset", espera="completed")
        for caso, n in (("rotura", 1), ("rotura2", 2)):
            roturas = json.loads((self.salida / caso / "sim_metadata_1.json").read_text())["roturas_dict"]
            assert len(roturas) == n, f"{caso}: se esperaban {n} roturas y hay {len(roturas)}"

        # --- SP_reset desde un estado preparado a partir del final de PP_reset de ciclo1
        estado_final = np.load(self.salida / "ciclo1/Final_state_pp_reset_1.npz")["arr_0"].astype(np.int64)
        ultimo = max((self.salida / "ciclo1/reset").glob("Estado_pp_reset_sim_1_paso_*.npz"), key=lambda p: int(p.stem.rsplit("_", 1)[1]))
        mapa_T = np.load(ultimo)["temperatura"].astype(np.float64)
        rng = np.random.default_rng(7)
        iones = np.zeros(estado_final.shape, dtype=np.int8)
        iones[rng.integers(0, estado_final.shape[0], 12), rng.integers(0, 6, 12)] = 1
        reloj = (iones * rng.integers(1, 4, estado_final.shape)).astype(np.int32)
        comun = dict(tiempo_pp_reset=0.5, roturas_dict={}, centros_calculados=[59], max_oxigenos_reset=150)
        for nombre, percola in (("sp_reset_ok", False), ("sp_reset_percola", True)):
            d = self.crear(nombre)
            estado = estado_final.copy()
            if percola:  # el filamento sigue formado: SP_reset debe negarse a arrancar
                arrays = dict(actual_state=estado, oxygen_state=np.zeros_like(iones), reloj_iones=np.zeros_like(reloj),
                              CF_destruido=np.array([False]), voltage_CF_destruido=np.array([0.0]),
                              Temperatura_final=mapa_T, perfil_electrodo=np.linspace(300, 330, estado.shape[0]))
                meta = dict(comun, sistema_percola=True, v_prev_reloj=-0.1, voltaje={"V_fin": -0.1})
            else:  # filamento cortado y registrado como destruido, con iones en vuelo
                estado[:, 20] = 0
                arrays = dict(actual_state=estado, oxygen_state=iones, reloj_iones=reloj,
                              CF_destruido=np.array([True]), voltage_CF_destruido=np.array([-1.0]))
                meta = dict(comun, sistema_percola=False, Temperatura_final=352.5, perfil_electrodo=300.0,
                            v_prev_reloj=-1.1, voltaje={"V_fin": -1.1})
            np.savez_compressed(d / "Init_data/phase_state_1_pp_reset.npz", **arrays)
            (d / "Init_data/phase_state_1_pp_reset.json").write_text(json.dumps(meta))
            self.ejecutar(nombre, "--start-from", "sp_reset",
                          espera="failed_at_sp_reset · FilamentosNoDestruidosException" if percola else "completed")


# ====================================================================== comparación


def _ficheros(d: Path) -> list[str]:
    return sorted(str(p.relative_to(d)) for p in d.rglob("*") if p.is_file() and p.suffix != ".png")


def _iguales(a: Path, b: Path) -> list[str]:
    """Diferencias entre dos ficheros (lista vacía = idénticos)."""
    if a.suffix == ".npz":
        za, zb = np.load(a), np.load(b)
        difs = [f"clave solo en un lado: {k}" for k in sorted(set(za.files) ^ set(zb.files))]
        for k in sorted(set(za.files) & set(zb.files)):
            x, y = za[k], zb[k]
            if x.dtype != y.dtype or x.shape != y.shape or not np.array_equal(x, y, equal_nan=x.dtype.kind == "f"):
                difs.append(f"difiere {k}")
        return difs
    if a.suffix == ".json":
        return [] if json.loads(a.read_text()) == json.loads(b.read_text()) else ["difiere"]
    return [] if a.read_bytes() == b.read_bytes() else ["difiere"]


def comparar(etiqueta_a: str, etiqueta_b: str) -> int:
    a, b = TRABAJO / "salidas" / etiqueta_a, TRABAJO / "salidas" / etiqueta_b
    for d in (a, b):
        if not d.is_dir():
            sys.exit(f"No existe {d}: ejecuta antes `bateria.py ejecutar {d.name}`.")
    casos_a, casos_b = sorted(p.name for p in a.iterdir()), sorted(p.name for p in b.iterdir())
    if casos_a != casos_b:
        sys.exit(f"Las dos ejecuciones no tienen los mismos casos: {sorted(set(casos_a) ^ set(casos_b))}")
    malos = 0
    for caso in casos_a:
        fa, fb = _ficheros(a / caso), _ficheros(b / caso)
        difs = [f"solo en {etiqueta_a}: {f}" for f in sorted(set(fa) - set(fb))] + [f"solo en {etiqueta_b}: {f}" for f in sorted(set(fb) - set(fa))]
        for f in sorted(set(fa) & set(fb)):
            difs += [f"{f}: {d}" for d in _iguales(a / caso / f, b / caso / f)]
        malos += bool(difs)
        print(f"{'MAL' if difs else 'OK '} {caso:18s} {len(set(fa) & set(fb)):4d} ficheros" + (f" | {len(difs)} diferencias" if difs else ""))
        for d in difs[:6]:
            print("       ", d)
    print("TODO IDÉNTICO" if not malos else f"{malos} casos con diferencias")
    return 1 if malos else 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="orden", required=True)
    e = sub.add_parser("ejecutar", help="simula los 14 casos")
    e.add_argument("etiqueta")
    e.add_argument("--raiz", default=str(REPO), help="carpeta que contiene el paquete RRAM a ejecutar (por defecto, este repo)")
    c = sub.add_parser("comparar", help="compara dos ejecuciones fichero a fichero")
    c.add_argument("a")
    c.add_argument("b")
    args = p.parse_args()
    if args.orden == "comparar":
        return comparar(args.a, args.b)
    bateria = Bateria(args.etiqueta, Path(args.raiz))
    print(f"Batería '{args.etiqueta}' con el paquete de {bateria.raiz}")
    bateria.todos()
    print(f"{len(bateria.resumen)} casos en {bateria.salida}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
