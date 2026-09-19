"""Parámetros geométricos y temporales de la simulación RRAM."""

from dataclasses import dataclass, field, fields
from typing import Optional, get_type_hints

import numpy as np


@dataclass
class SimulationParameters:
    device_size_x: float
    device_size_y: float
    atom_size: float
    num_trampas: int
    # Paso temporal [s]. Es ENTRADA del CSV, no un valor derivado: `dt` aparece
    # dentro de la física (probabilidad de generación en Generation.py, de
    # recombinación y desplazamiento de iones en Recombination.py), así que si se
    # derivase de `num_pasos` cualquier cambio del número de pasos movería `dt` en
    # silencio y perturbaría la calibración del modelo. Fijándolo aquí, `num_pasos`
    # solo gobierna el paso de potencial y el presupuesto de pasos de cada fase.
    # CUIDADO: cambiar su valor SÍ invalida la calibración (ver MANUAL_FORMAS_DE_ONDA.md).
    paso_temporal: float
    num_pasos: int
    voltaje_final_reset: float
    voltaje_final_set: float
    init_temp: float
    densidad_vacantes: float

    # Semilla de reproducibilidad. None (defecto) = aleatorio real, igual que
    # el comportamiento histórico. Si se fija (vía --seed en la CLI), cada fase
    # (PP_set, SP_set, PP_reset, SP_reset) reinicia np.random con este mismo
    # valor al empezar, para que la corrida sea exactamente reproducible.
    seed: Optional[int] = None

    x_size: int = field(init=False)
    y_size: int = field(init=False)
    num_max_vacantes: int = field(init=False)
    # Duración NOMINAL de una fase [s] = num_pasos * paso_temporal. Es un valor
    # derivado e informativo: la duración REAL de una fase es
    # (pasos ejecutados) * paso_temporal, que con formas de onda puede ser menor
    # (la fase termina antes) o mayor (mesetas que extienden el presupuesto).
    total_simulation_time: float = field(init=False)
    paso_potencial_set: float = field(init=False)
    paso_potencial_reset: float = field(init=False)

    def __post_init__(self):
        self.x_size = int(np.ceil(self.device_size_x / self.atom_size))  # Número de "casillas" en la dimensión x
        self.y_size = int(np.ceil(self.device_size_y / self.atom_size))  # Número de "casillas" en la dimensión y
        self.num_max_vacantes = int(0.95 * (self.x_size * self.y_size))  # 95% de la matriz puede llenarse de vacantes
        self.total_simulation_time = self.num_pasos * self.paso_temporal  # Duración nominal en segundos
        self.paso_potencial_set = self.voltaje_final_set / self.num_pasos  # Paso de voltaje para la parte de set
        self.paso_potencial_reset = self.voltaje_final_reset / self.num_pasos  # Paso de voltaje para la parte de reset

    def __repr__(self):
        # Crear lista de líneas con "nombre=valor" para cada atributo
        atributos = []
        # Usar vars(self) para incluir también campos calculados en __post_init__
        for nombre, valor in vars(self).items():
            atributos.append(f"   {nombre}={valor}")
        # Formatear en varias líneas
        return "Los parámetros de la simulación son:\n" + ",\n".join(atributos) + "\n"

    @staticmethod
    def from_dict(d: dict):
        field_types = get_type_hints(SimulationParameters)
        init_fields = {f.name for f in fields(SimulationParameters) if f.init}
        kwargs = {}
        for k in init_fields:
            if k == "seed":
                # Opcional: no viene del CSV de parámetros; se fija por CLI
                # (--seed) u override explícito, nunca se exige aquí.
                raw = d.get("seed")
                kwargs["seed"] = int(raw) if raw not in (None, "") else None
                continue
            if k not in d:
                raise KeyError(
                    f"Falta la columna '{k}' en simulation_parameters.csv. "
                    f"Si es un CSV generado antes de que '{k}' fuese obligatorio, "
                    f"regenéralo desde el notebook con ConfigManager.export_to_init_data()."
                )
            kwargs[k] = field_types[k](d[k])
        return SimulationParameters(**kwargs)
