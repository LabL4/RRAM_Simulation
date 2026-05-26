"""
extraer_barrido_densidad.py
Extrae datos de simulaciones RRAM con estructura plana:
  <raiz>/Results/simulation_*/sim_metadata_*.json
  <raiz>/logs/log_simulacion_*.log

Uso:
    python extraer_barrido_densidad.py [directorio_raiz]
    Si no se pasa argumento, usa el directorio actual.
"""

from pathlib import Path
import json
import sys
import re


def buscar_jsons(raiz: Path) -> list[Path]:
    return sorted(raiz.glob("Results/simulation_*/sim_metadata_*.json"))


def buscar_logs(raiz: Path) -> dict[int, Path]:
    idx: dict[int, Path] = {}
    for log in raiz.glob("logs/log_simulacion_*.log"):
        m = re.search(r"log_simulacion_(\d+)", log.name)
        if m:
            idx[int(m.group(1))] = log
    return idx


def extraer_voltaje_rotura(ruta_log: Path) -> str | None:
    patron = re.compile(r"El filamento \d+ se ha roto en el voltaje\s+([-\d.]+)\s*\(V\)")
    try:
        texto = ruta_log.read_text(encoding="utf-8", errors="replace")
        m = patron.search(texto)
        return m.group(1) if m else None
    except OSError:
        return None


def extraer_datos_json(ruta_json: Path) -> dict | None:
    try:
        datos = json.loads(ruta_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    ctes = datos.get("ctes_dict", {})
    params = datos.get("params_dict", {})
    extra = datos.get("extra", {})
    vecindad = extra.get("vecindad_inicial", {})

    grosor_raw = ctes.get("grosor_filamento", None)
    grosor = grosor_raw[0] if isinstance(grosor_raw, list) else grosor_raw
    atom_size = params.get("atom_size", None)

    # Diámetro: (2 * grosor + 1) * atom_size
    if grosor is not None and atom_size is not None:
        diametro = (2 * grosor + 1) * atom_size
    else:
        diametro = None

    return {
        "diametro": diametro,
        "densidad_vacantes": params.get("densidad_vacantes", None),
        "n0": vecindad.get("vecinos_0", "N/A"),
        "n1": vecindad.get("vecinos_1", "N/A"),
        "n2": vecindad.get("vecinos_2", "N/A"),
        "n3": vecindad.get("vecinos_3", "N/A"),
        "n4": vecindad.get("vecinos_4", "N/A"),
    }


def formatear_valor(v) -> str:
    if isinstance(v, float):
        return f"{v:.4e}" if abs(v) < 1e-6 and v != 0.0 else f"{v:.4f}"
    return str(v)


def main() -> None:
    raiz = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()

    jsons = buscar_jsons(raiz)
    if not jsons:
        print(f"[ERROR] No se encontraron sim_metadata_*.json en {raiz}/Results/")
        sys.exit(1)

    logs = buscar_logs(raiz)
    print(f"[INFO] {len(jsons)} JSONs · {len(logs)} logs encontrados")

    filas: list[dict] = []
    for ruta_json in jsons:
        m = re.search(r"sim_metadata_(\d+)", ruta_json.name)
        num_sim = int(m.group(1)) if m else None

        datos = extraer_datos_json(ruta_json)
        if datos is None:
            print(f"  [WARN] No se pudo leer {ruta_json.name}")
            continue

        voltaje = None
        if num_sim is not None and num_sim in logs:
            voltaje = extraer_voltaje_rotura(logs[num_sim])

        filas.append(
            {
                "num_sim": num_sim,
                **datos,
                "v_reset": voltaje if voltaje is not None else "N/A",
            }
        )

    filas.sort(key=lambda x: x["num_sim"] or 0)

    # Escribir TSV
    columnas = ["Sim", "Diametro (m)", "Dens.Vac.", "N_0", "N_1", "N_2", "N_3", "N_4", "V_reset (V)"]
    lineas = ["\t".join(columnas)]
    for f in filas:
        lineas.append(
            "\t".join(
                [
                    str(f["num_sim"]),
                    formatear_valor(f["diametro"]),
                    formatear_valor(f["densidad_vacantes"]),
                    str(f["n0"]),
                    str(f["n1"]),
                    str(f["n2"]),
                    str(f["n3"]),
                    str(f["n4"]),
                    str(f["v_reset"]),
                ]
            )
        )

    ruta_salida = raiz / "resultados_barrido_densidad_FULL_1.txt"
    ruta_salida.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print(f"[OK] {ruta_salida}  ({len(filas)} filas)")


if __name__ == "__main__":
    main()
