"""Rutas de salida de una simulación."""

from pathlib import Path


def crear_rutas_simulacion(num_simulation: int, state: str, results_dir: str | Path = "Results") -> dict:
    """
    Crea las rutas necesarias para guardar resultados de la simulación.

    Args:
        num_simulation (int): Número índice de la simulación.
        state (str): Estado de la simulación, puede ser 'set' o 'reset'.
        results_dir (str | Path): Carpeta raíz de resultados. Por defecto "Results"
            (comportamiento histórico, relativo a Path.cwd()). Pasar una ruta
            distinta (p.ej. para pruebas de diagnóstico) aísla completamente
            los datos escritos por esta simulación sin tocar Results/ real.

    Returns:
        dict: Diccionario con las rutas Path para 'simulation', 'figures' y 'set'.

    Side-effects:
        Crea las carpetas en disco si no existen.
    """
    results_dir = Path(results_dir)
    base = results_dir if results_dir.is_absolute() else Path.cwd() / results_dir
    simulation_path = base / f"simulation_{num_simulation}"
    figures_path = simulation_path / "Figures"
    data_simulation_path = simulation_path / state

    return {
        "simulation_path": simulation_path,
        "figures_path": figures_path,
        "data_simulation_path": data_simulation_path,
    }
