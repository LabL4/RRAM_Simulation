"""
Verificación del viaje del protocolo de voltaje por la configuración y de los
cambios de constantes durante la simulación.

  1. ConfigManager → Init_data/simulation_voltage.csv → load_simulation_config:
     el protocolo llega intacto a SimulationConfig.protocolo.
  2. Validación al exportar: sin protocolo, o con uno mal escrito, falla ya en
     export_to_init_data (no a mitad de simulación).
  3. SimulationConstants.update_*: cambian el valor y lo dejan en el log.

Ejecutar desde la raíz del repo:
    python3 tests/test_configuracion_protocolo.py
"""

import contextlib
import io
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from RRAM import simulation_config  # noqa: E402
from RRAM.constants_simulation import SimulationConstants  # noqa: E402
from RRAM.init_simulation import load_simulation_config  # noqa: E402
from RRAM.voltage_controller import ProtocoloVoltaje, constante, rampa, tren_pulsos  # noqa: E402

DS = 1.1 / 1000
DR = 1.4 / 1000

PROTOCOLO_RAMPAS = {
    "pp_set": [rampa(0.0, 1.1, DS)],
    "sp_set": [rampa("anterior", 0.0, DS)],
    "pp_reset": [rampa("anterior", -1.4, DR)],
    "sp_reset": [rampa("anterior", 0.0, DR)],
}
PROTOCOLO_PULSOS = {
    **PROTOCOLO_RAMPAS,
    "pp_set": tren_pulsos(1.1, 150, 0.0, 100, 3),
    "sp_set": [rampa(1.1, 0.0, DS)],
}
PROTOCOLO_COMPLIANCE = {
    **PROTOCOLO_RAMPAS,
    "pp_set": [rampa(0.0, 1.1, DS, hasta_percolacion=True, exigir=True), constante("anterior", 200)],
}

fallos = []


def check(nombre, cond, detalle=""):
    print(f"{'  OK  ' if cond else ' FALLO'} | {nombre}" + (f"  ({detalle})" if detalle else ""))
    if not cond:
        fallos.append(nombre)


def exportar(carpeta, zip_params, fixed_params=None):
    """export_to_init_data en `carpeta`, silenciando el resumen que imprime."""
    manager = simulation_config.ConfigManager()
    manager.add_zip(zip_params=zip_params, fixed_params=fixed_params)
    with contextlib.redirect_stdout(io.StringIO()):
        manager.export_to_init_data(str(carpeta))
    return manager


def falla(fn):
    try:
        fn()
        return False
    except ValueError:
        return True


# ---------------------------------------------------------------- 1
print("\n[1] ConfigManager → simulation_voltage.csv → load_simulation_config")
with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    protocolos = [PROTOCOLO_RAMPAS, PROTOCOLO_PULSOS, PROTOCOLO_COMPLIANCE]
    exportar(tmp, {"protocolo_voltaje": protocolos, "sigma_0": [4.3e8, 4.5e8, 4.7e8]})

    check("se escribe simulation_voltage.csv", (tmp / "simulation_voltage.csv").is_file())
    cabecera_params = (tmp / "simulation_parameters.csv").read_text().splitlines()[0].split(",")
    cabecera_ctes = (tmp / "simulation_constants.csv").read_text().splitlines()[0].split(",")
    restos = [c for c in cabecera_params + cabecera_ctes if c.startswith(("voltaje_final", "waveform_", "num_pasos", "v_inicial"))]
    check("ningún CSV conserva columnas de voltaje antiguas", not restos, f"{restos}")

    # load_simulation_config necesita un estado inicial por simulación.
    for i in range(len(protocolos)):
        np.savez_compressed(tmp / f"init_state_{i}.npz", actual_state=np.zeros((120, 40), dtype=int))

    for i, original in enumerate(protocolos):
        cfg = load_simulation_config(i, init_data_dir=tmp)
        esperado = ProtocoloVoltaje.desde_config(original).configuracion()
        check(f"sim {i}: el protocolo llega intacto a cfg.protocolo", cfg.protocolo.configuracion() == esperado)
    check("cada simulación conserva su propio sigma_0", load_simulation_config(2, init_data_dir=tmp).sim_ctes.sigma_0 == 4.7e8)

    rec = load_simulation_config(0, init_data_dir=tmp).protocolo.recorrido()
    check("el protocolo cargado recorre 0 → 1.1 V en pp_set", abs(rec["pp_set"]["V"][0]) < 1e-15 and abs(rec["pp_set"]["V"][-1] - 1.1) < 1e-12)

# ---------------------------------------------------------------- 2
print("\n[2] Validación al exportar")
with tempfile.TemporaryDirectory() as tmp:
    check("sin protocolo_voltaje falla al exportar", falla(lambda: exportar(tmp, {"sigma_0": [4.5e8]})))
    malo = {**PROTOCOLO_RAMPAS, "sp_set": [("rampa", {"desde": 1.1, "hasta": 0.0})]}  # falta dV
    check("un protocolo mal escrito falla al exportar", falla(lambda: exportar(tmp, {"sigma_0": [4.5e8]}, {"protocolo_voltaje": malo})))
    sin_etapa = {k: v for k, v in PROTOCOLO_RAMPAS.items() if k != "pp_reset"}
    check("un protocolo sin una etapa falla al exportar", falla(lambda: exportar(tmp, {"sigma_0": [4.5e8]}, {"protocolo_voltaje": sin_etapa})))
    exportar(tmp, {"sigma_0": [4.5e8]}, {"protocolo_voltaje": PROTOCOLO_RAMPAS})
    check("el protocolo en fixed_params sí se exporta", (Path(tmp) / "simulation_voltage.csv").is_file())

# ---------------------------------------------------------------- 3
print("\n[3] SimulationConstants.update_*: cambian el valor y lo registran en el log")
with tempfile.TemporaryDirectory() as tmp:
    exportar(tmp, {"sigma_0": [4.5e8]}, {"protocolo_voltaje": PROTOCOLO_RAMPAS})
    import csv

    with open(Path(tmp) / "simulation_constants.csv") as f:
        ctes = SimulationConstants.from_dict(next(csv.DictReader(f)))

registro = io.StringIO()
manejador = logging.StreamHandler(registro)
log_ctes = logging.getLogger("RRAM.constants_simulation")
log_ctes.addHandler(manejador)
nivel_previo = log_ctes.level
log_ctes.setLevel(logging.INFO)

casos = [
    ("update_gamma", (5.0,), "gamma"),
    ("update_recombination_energy", (1.3,), "recombination_energy"),
    ("update_generation_energy", (1.1,), "generation_energy"),
    ("update_I_0", (1e-4, "reset"), "I_0_reset"),
    ("update_pb_metal_insul", (0.01, "set"), "pb_metal_insul_set"),
    ("update_permitividad_relativa", (100.0, "reset"), "permitividad_relativa_reset"),
]
for metodo, args, atributo in casos:
    antes = getattr(ctes, atributo)
    registro.seek(0)
    registro.truncate()
    nuevas = getattr(ctes, metodo)(*args)
    texto = registro.getvalue()
    check(f"{metodo}: cambia {atributo}", getattr(nuevas, atributo) == args[0] and getattr(ctes, atributo) == antes)
    check(f"{metodo}: deja en el log '{atributo}: anterior → nuevo'", f"{atributo}: {antes} → {args[0]}" in texto, texto.strip())

check("update_I_0 con fase inválida falla", falla(lambda: ctes.update_I_0(1e-4, "otra")))

log_ctes.removeHandler(manejador)
log_ctes.setLevel(nivel_previo)

# ----------------------------------------------------------------
print(f"\n{'=' * 60}")
if fallos:
    print(f"RESULTADO: {len(fallos)} fallo(s): {fallos}")
    sys.exit(1)
print("RESULTADO: todos los checks OK")
