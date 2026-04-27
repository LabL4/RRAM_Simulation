"""Constantes físicas inmutables de la simulación RRAM (HfOx)."""

import ast
from dataclasses import dataclass, replace
from typing import Any, Dict, get_type_hints


@dataclass(frozen=True)
class SimulationConstants:
    vibration_frequency: float
    cte_red: float
    permitividad_relativa_set: float
    permitividad_relativa_reset: float
    generation_energy: float
    recombination_energy: float
    pb_metal_insul_set: float
    pb_metal_insul_reset: float
    recom_enchancement_factor: float
    long_decaimiento_concentracion: float
    ohm_resistence_set: float
    ohm_resistence_reset: float
    num_filamentos: int
    grosor_filamento: int
    gamma: float
    gamma_drift: float
    E_m: float
    I_0_set: float
    I_0_reset: float
    r_termica_no_percola: float
    conductividad_termica_dielectrico: float
    conductividad_termica_CF: float
    conductividad_termica_aislante: float
    conductividad_termica_electrodo: float
    Temperatura_electrodo: float
    factor_generar_calor: float
    pendiente_temperatura: float
    ocupacion_max_pp_set: float
    ocupacion_max_sp_set: float
    factor_vecinos_pp_set: float
    factor_libre_pp_set: float
    factor_vecinos_sp_set: float
    factor_libre_sp_set: float
    lim_voltage_percolacion: float
    compliance_voltage: float
    voltaje_gen_oxigeno_pp_1: float
    num_oxigenos_pp_reset_1: int
    voltaje_gen_oxigeno_pp_2: float
    num_oxigenos_pp_reset_2: int
    voltaje_gen_oxigeno_sp: float
    num_oxigenos_sp_reset: int

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SimulationConstants":
        """
        Crea una instancia desde un diccionario, parseando strings que contienen listas.
        Respeta los tipos definidos en el dataclass.

        Args:
            data: Diccionario con los valores de las constantes.
                Los valores pueden ser strings con formato de lista "[1,2,3]".

        Returns:
            Instancia de SimulationConstants con valores parseados.
        """

        # Obtener los tipos definidos en el dataclass
        type_hints = get_type_hints(cls)
        parsed_data = {}

        for key, value in data.items():
            if key not in type_hints:
                parsed_data[key] = value
                continue

            expected_type = type_hints[key]

            if isinstance(value, str):
                value_stripped = value.strip()

                # Detectar si el string representa una lista
                if value_stripped.startswith("[") and value_stripped.endswith("]"):
                    try:
                        parsed_data[key] = ast.literal_eval(value_stripped)
                    except (ValueError, SyntaxError):
                        parsed_data[key] = value
                else:
                    # Convertir según el tipo esperado
                    try:
                        # Manejar Union types (extraer el tipo base)
                        if hasattr(expected_type, "__origin__"):
                            # Para Union[int, List[int]], tomar int como base
                            base_types = expected_type.__args__
                            # Filtrar tipos que no sean List
                            non_list_types = [
                                t for t in base_types if not (hasattr(t, "__origin__") and t.__origin__ is list)
                            ]
                            if non_list_types:
                                expected_type = non_list_types[0]

                        if expected_type == int:
                            parsed_data[key] = int(float(value))
                        elif expected_type == float:
                            parsed_data[key] = float(value)
                        else:
                            parsed_data[key] = value
                    except (ValueError, TypeError):
                        parsed_data[key] = value
            else:
                parsed_data[key] = value

        return cls(**parsed_data)

    @property
    def propiedades_termicas(self) -> dict:
        return {
            0: {"k": self.conductividad_termica_dielectrico},
            1: {"k": self.conductividad_termica_CF},
            2: {"k": self.conductividad_termica_aislante},
            3: {"k": self.conductividad_termica_electrodo},
        }

    def update_gamma(self, nuevo_valor_gamma: float):
        # gamma no tiene distinción de set/reset en los atributos originales,
        # por lo que este se queda igual, pero asegurándonos de que la variable exista.
        return replace(self, gamma=nuevo_valor_gamma)

    def update_ohm_resistence(self, nuevo_valor: float, fase: str = "set"):
        """Actualiza la resistencia óhmica dependiendo de la fase ('set' o 'reset')."""
        if fase not in ["set", "reset"]:
            raise ValueError("El parámetro 'fase' debe ser 'set' o 'reset'.")

        # Construimos el nombre exacto del atributo: "ohm_resistence_set" o "ohm_resistence_reset"
        atributo = f"ohm_resistence_{fase}"

        # Usamos un diccionario para pasar el argumento dinámicamente a replace()
        return replace(self, **{atributo: nuevo_valor})

    def update_I_0(self, nuevo_I_0: float, fase: str = "set"):
        """Actualiza la corriente de referencia I_0 dependiendo de la fase ('set' o 'reset')."""
        if fase not in ["set", "reset"]:
            raise ValueError("El parámetro 'fase' debe ser 'set' o 'reset'.")

        atributo = f"I_0_{fase}"
        return replace(self, **{atributo: nuevo_I_0})

    def update_pb_metal_insul(self, nuevo_pb_metal_insul: float, fase: str = "set"):
        """Actualiza la barrera de potencial dependiendo de la fase ('set' o 'reset')."""
        if fase not in ["set", "reset"]:
            raise ValueError("El parámetro 'fase' debe ser 'set' o 'reset'.")

        atributo = f"pb_metal_insul_{fase}"
        return replace(self, **{atributo: nuevo_pb_metal_insul})

    def update_permitividad_relativa(self, permitividad_relativa_nuevo: float, fase: str = "set"):
        """Actualiza la permitividad relativa dependiendo de la fase ('set' o 'reset')."""
        if fase not in ["set", "reset"]:
            raise ValueError("El parámetro 'fase' debe ser 'set' o 'reset'.")
        atributo = f"permitividad_relativa_{fase}"
        return replace(self, **{atributo: permitividad_relativa_nuevo})

    def update_generation_energy(self, nueva_energia: float):
        """Actualiza la energía de generación."""
        return replace(self, generation_energy=nueva_energia)

    def update_recombination_energy(self, nueva_energia: float):
        """Actualiza la energía de generación."""
        return replace(self, recombination_energy=nueva_energia)

    def __repr__(self):
        # Crear lista de líneas con "nombre=valor" para cada atributo
        atributos = []
        # Usar vars(self) para incluir también campos calculados en __post_init__
        for nombre, valor in vars(self).items():
            atributos.append(f"   {nombre}={valor}")
        # Formatear en varias líneas
        return "Las constantes de la simulación son:\n" + ",\n".join(atributos) + "\n"
