"""
Persistencia de metadatos y datos de simulación.

Permite separar el ciclo en init → exec → plot:
- `exec` guarda metadata.json y Data_*.npz en `Results/simulation_{N}/`.
- `plot` lee esos archivos del disco SIN necesitar el contexto de ejecución
  (no requiere ejecutar las fases otra vez).

Estructura de archivos por simulación:

    Results/simulation_{N}/
        ├── sim_metadata_{N}.json      ← metadatos para replot + auditoría
        ├── Data_pp_set_{N}.npz
        ├── Data_sp_set_{N}.npz
        ├── Data_pp_reset_{N}.npz
        ├── Data_sp_reset_{N}.npz
        ├── Final_state_{N}_pp_set.npz
        ├── Final_state_pp_set_{N}.npz
        ├── Final_state_pp_reset_{N}.npz
        └── Final_state_sp_reset_{N}.npz
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


def metadata_filename(num_simulation: int) -> str:
    """Nombre canónico del JSON de metadatos por simulación."""
    return f"sim_metadata_{num_simulation}.json"


def serialize_dataclass(obj: Any) -> Dict[str, Any]:
    """Serializa cualquier dataclass (frozen o no) a un dict JSON-friendly."""
    if obj is None:
        return {}
    if is_dataclass(obj):
        return _to_jsonable(asdict(obj))
    if hasattr(obj, "__dict__"):
        return _to_jsonable(vars(obj))
    return {}


def _to_jsonable(value: Any) -> Any:
    """Convierte recursivamente numpy/Path/tuple a tipos JSON nativos."""
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


@dataclass
class SimulationMetadata:
    """
    Metadatos producidos por una simulación. Contiene todo lo necesario para:
    1. Replotear sin re-ejecutar (`creaciones_dict`, `roturas_dict`,
       `voltaje_percolacion`, `centros_calculados`).
    2. Auditar/reproducir la sim (`params_dict`, `ctes_dict` con copia exacta
       de los CSV usados).
    """

    num_simulation: int
    voltaje_percolacion: float
    creaciones_dict: Dict[Any, Any] = field(default_factory=dict)
    roturas_dict: Dict[Any, Any] = field(default_factory=dict)
    centros_calculados: Optional[list] = None
    cf_ranges: Optional[list] = None
    params_dict: Dict[str, Any] = field(default_factory=dict)
    ctes_dict: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _to_jsonable(asdict(self))

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SimulationMetadata":
        # JSON solo soporta keys string; restauramos a int las claves numéricas
        # de los dicts de eventos.
        d = dict(d)
        for key in ("creaciones_dict", "roturas_dict"):
            raw = d.get(key, {}) or {}
            normalised = {}
            for k, v in raw.items():
                try:
                    normalised[int(k)] = v
                except (ValueError, TypeError):
                    normalised[k] = v
            d[key] = normalised
        # Drop campos antiguos que ya no estén en el dataclass
        valid = {f for f in cls.__dataclass_fields__}
        d = {k: v for k, v in d.items() if k in valid}
        return cls(**d)


def save_metadata(meta: SimulationMetadata, simulation_path: Path) -> Path:
    """Guarda los metadatos como JSON en `simulation_path/sim_metadata_{N}.json`."""
    simulation_path = Path(simulation_path)
    simulation_path.mkdir(parents=True, exist_ok=True)
    out = simulation_path / metadata_filename(meta.num_simulation)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(meta.to_dict(), f, ensure_ascii=False, indent=2, default=_json_default)
    logger.info(f"Metadatos guardados en {out}")
    return out


def load_metadata(simulation_path: Path, num_simulation: int | None = None) -> SimulationMetadata:
    """
    Carga los metadatos de una simulación.

    Args:
        simulation_path: Carpeta `Results/simulation_{N}/`.
        num_simulation: Si se proporciona, busca `sim_metadata_{num}.json`.
            Si es None se intenta inferir del nombre de la carpeta
            (`simulation_{N}` → N) o se busca cualquier `sim_metadata_*.json`.
    """
    simulation_path = Path(simulation_path)

    candidates = []
    if num_simulation is not None:
        candidates.append(simulation_path / metadata_filename(num_simulation))
    else:
        # Inferir del nombre de la carpeta
        try:
            n = int(simulation_path.name.split("_")[-1])
            candidates.append(simulation_path / metadata_filename(n))
        except (ValueError, IndexError):
            pass
        # Glob como último recurso
        candidates.extend(sorted(simulation_path.glob("sim_metadata_*.json")))
        # Compatibilidad hacia atrás con el nombre antiguo
        candidates.append(simulation_path / "sim_metadata.json")

    for path in candidates:
        if path.is_file():
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
            return SimulationMetadata.from_dict(d)

    raise FileNotFoundError(
        f"No se encuentra sim_metadata_*.json en {simulation_path}. "
        f"¿Se ha ejecutado la fase exec? Lanza `python -m RRAM exec <num>` antes de plot."
    )


def _json_default(obj: Any) -> Any:
    """Serializador JSON que tolera tipos numpy y Path."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"Objeto no serializable a JSON: {type(obj).__name__}")
