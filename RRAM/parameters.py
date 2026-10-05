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
    # recombinación y desplazamiento de iones en Recombination.py).
    # CUIDADO: cambiar su valor SÍ invalida la calibración (ver MANUAL_FORMAS_DE_ONDA.md).
    # El voltaje (formas de onda, pasos de potencial) NO está aquí: vive entero en
    # RRAM.voltage_controller.ProtocoloVoltaje.
    paso_temporal: float
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

    def __post_init__(self):
        self.x_size = int(np.ceil(self.device_size_x / self.atom_size))  # Número de "casillas" en la dimensión x
        self.y_size = int(np.ceil(self.device_size_y / self.atom_size))  # Número de "casillas" en la dimensión y
        self.num_max_vacantes = int(0.95 * (self.x_size * self.y_size))  # 95% de la matriz puede llenarse de vacantes

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
