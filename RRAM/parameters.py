"""Parámetros geométricos y temporales de la simulación RRAM."""

from dataclasses import dataclass, field, fields
from typing import get_type_hints

import numpy as np


@dataclass
class SimulationParameters:
    device_size_x: float
    device_size_y: float
    atom_size: float
    num_trampas: int
    total_simulation_time: float
    num_pasos: int
    voltaje_final_reset: float
    voltaje_final_set: float
    init_temp: float

    x_size: int = field(init=False)
    y_size: int = field(init=False)
    num_max_vacantes: int = field(init=False)
    paso_temporal: float = field(init=False)
    paso_potencial_set: float = field(init=False)
    paso_potencial_reset: float = field(init=False)

    def __post_init__(self):
        self.x_size = int(np.ceil(self.device_size_x / self.atom_size))  # Número de "casillas" en la dimensión x
        self.y_size = int(np.ceil(self.device_size_y / self.atom_size))  # Número de "casillas" en la dimensión y
        self.num_max_vacantes = int(0.95 * (self.x_size * self.y_size))  # 95% de la matriz puede llenarse de vacantes
        self.paso_temporal = self.total_simulation_time / self.num_pasos  # Paso temporal en segundos
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
            if k not in d:
                raise KeyError(f"La clave '{k}' no existe en el diccionario")
            kwargs[k] = field_types[k](d[k])
        return SimulationParameters(**kwargs)
