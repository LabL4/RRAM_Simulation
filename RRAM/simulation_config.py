from typing import Dict, Any, Optional, List
import pandas as pd
import itertools
import os

# ============================================================================
# PARÁMETROS MATERIALES (HfOx)
# ============================================================================
MATERIAL_DEFAULTS = {
    "cte_red": 0.125e-9,
    "permitividad_relativa_set": 250.0,
    "permitividad_relativa_reset": 299.9991,
    "generation_energy": 1.0,
    "recombination_energy": 0.955,
    "pb_metal_insul_set": 0.2547,
    "pb_metal_insul_reset": 0.2465,
    "recom_enchancement_factor": 3e3,
    "long_decaimiento_concentracion": 1e-9,
    "ohm_resistence_set": 250.0,
    "ohm_resistence_reset": 250.0,
}

# ============================================================================
# CONSTANTES FÍSICAS
# ============================================================================
PHYSICAL_CONSTANTS = {
    "vibration_frequency": 1e13,
    "gamma": 10.0,
    "gamma_drift": 10.0,
    "E_m": 0.6,
}

# ============================================================================
# PARÁMETROS ELÉCTRICOS
# ============================================================================
ELECTRICAL_DEFAULTS = {
    "I_0_set": 1.8613e-01,
    "I_0_reset": 1.1221e-01,
}

# ============================================================================
# PARÁMETROS TÉRMICOS
# ============================================================================
THERMAL_DEFAULTS = {
    "init_temp": 300.0,
    "r_termica_no_percola": 1000.0,
    "conductividad_termica_aire": 0.02,
    "conductividad_termica_CF": 1.0,
    "conductividad_termica_aislante": 0.0,
    "conductividad_termica_electrodo": 1.0,
    "Temperatura_electrodo": 300,
}

# ============================================================================
# PARÁMETROS GEOMÉTRICOS Y DE SIMULACIÓN
# ============================================================================
SIMULATION_DEFAULTS = {
    "device_size": 5e-9,
    "atom_size": 0.125e-9,  # Se deberia llamar tamaño de red
    "num_trampas": 70,
    "total_simulation_time": 10.0,
    "num_pasos": 10000,
    "voltaje_final": 1.1,
    "voltaje_final_set": 1.1,
    "voltaje_final_reset": 1.5,
}

# ============================================================================
# PARÁMETROS SET / RESET
# ============================================================================
SET_RESET_DEFAULTS = {
    "ocupacion_max_pp_set": 0.35,
    "ocupacion_max_sp_set": 0.35,
    "factor_vecinos_pp_set": 1.2,
    "factor_libre_pp_set": 0.8,
    "factor_vecinos_sp_set": 1.0,
    "factor_libre_sp_set": 0.9,
    "lim_voltage_percolacion": 0.6,
    "compliance_voltage": 0.6,
    "voltaje_gen_oxigeno_pp_1": 0.45,
    "num_oxigenos_pp_reset_1": 1,
    "voltaje_gen_oxigeno_pp_2": 0.65,
    "num_oxigenos_pp_reset_2": 3,
    "voltaje_gen_oxigeno_sp": -0.2,
    "num_oxigenos_sp_reset": 5,
}

DEFAULT_PARAMS = {
    **MATERIAL_DEFAULTS,
    **PHYSICAL_CONSTANTS,
    **ELECTRICAL_DEFAULTS,
    **THERMAL_DEFAULTS,
    **SIMULATION_DEFAULTS,
    **SET_RESET_DEFAULTS,
}


class SimulationConfig:
    def __init__(self, sim_id: int):
        self.sim_id = sim_id
        self.params = DEFAULT_PARAMS.copy()
        self._recalculate_derived()

    def _recalculate_derived(self):
        self.params["x_size"] = int(round(self.params["device_size"] / self.params["atom_size"]))
        self.params["y_size"] = int(round(self.params["device_size"] / self.params["atom_size"]))

        self.params["conductividad_electrica_CF_set"] = 1.0 / (
            self.params["ohm_resistence_set"] * self.params["atom_size"]
        )
        self.params["conductividad_electrica_CF_reset"] = 1.0 / (
            self.params["ohm_resistence_reset"] * self.params["atom_size"]
        )

        self.params["propiedades_termicas"] = {
            0: {"k": self.params["conductividad_termica_aire"]},
            1: {"k": self.params["conductividad_termica_CF"]},
            2: {"k": self.params["conductividad_termica_aislante"]},
            3: {"k": self.params["conductividad_termica_electrodo"]},
        }

    def update(self, modifications: Dict[str, Any]):
        self.params.update(modifications)
        self._recalculate_derived()

    @classmethod
    def from_base(cls, sim_id: int, modifications: Optional[Dict[str, Any]] = None):
        instance = cls(sim_id)
        if modifications:
            instance.update(modifications)
        return instance


class ConfigManager:
    def __init__(self):
        self.simulations: List[SimulationConfig] = []

    def add_config(self, config: SimulationConfig):
        self.simulations.append(config)

    def add_sweep(self, sweep_params: Dict[str, List[Any]]):
        """Genera el producto cartesiano exacto de las listas proporcionadas."""
        keys = list(sweep_params.keys())
        values = list(sweep_params.values())

        casos_totales = 0
        for combinacion in itertools.product(*values):
            mods = dict(zip(keys, combinacion))
            sim_id = len(self.simulations)
            self.simulations.append(SimulationConfig.from_base(sim_id, mods))
            casos_totales += 1

        print(f"-> Generadas {casos_totales} combinaciones para el experimento.")

    def export_to_init_data(self, output_dir: str = "Init_data"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        cols_params = [
            "device_size",
            "atom_size",
            "x_size",
            "y_size",
            "num_trampas",
            "total_simulation_time",
            "num_pasos",
            "voltaje_final",
            "voltaje_final_set",
            "voltaje_final_reset",
            "init_temp",
        ]

        cols_ctes = [
            "cte_red",
            "permitividad_relativa_set",
            "permitividad_relativa_reset",
            "generation_energy",
            "recombination_energy",
            "pb_metal_insul_set",
            "pb_metal_insul_reset",
            "recom_enchancement_factor",
            "long_decaimiento_concentracion",
            "ohm_resistence_set",
            "ohm_resistence_reset",
            "vibration_frequency",
            "gamma",
            "gamma_drift",
            "E_m",
            "I_0_set",
            "I_0_reset",
            "r_termica_no_percola",
            "conductividad_termica_aire",
            "conductividad_termica_CF",
            "conductividad_termica_aislante",
            "conductividad_termica_electrodo",
            "Temperatura_electrodo",
            "conductividad_electrica_CF_set",
            "conductividad_electrica_CF_reset",
            "ocupacion_max_pp_set",
            "ocupacion_max_sp_set",
            "factor_vecinos_pp_set",
            "factor_libre_pp_set",
            "factor_vecinos_sp_set",
            "factor_libre_sp_set",
            "lim_voltage_percolacion",
            "compliance_voltage",
            "voltaje_gen_oxigeno_pp_1",
            "num_oxigenos_pp_reset_1",
            "voltaje_gen_oxigeno_pp_2",
            "num_oxigenos_pp_reset_2",
            "voltaje_gen_oxigeno_sp",
            "num_oxigenos_sp_reset",
        ]

        data_params = [{col: sim.params[col] for col in cols_params} for sim in self.simulations]
        data_ctes = [{col: sim.params[col] for col in cols_ctes} for sim in self.simulations]

        df_params = pd.DataFrame(data_params)
        df_ctes = pd.DataFrame(data_ctes)

        df_params.to_csv(os.path.join(output_dir, "simulation_parameters.csv"), index=False)
        df_ctes.to_csv(os.path.join(output_dir, "simulation_constants.csv"), index=False)

        print(f"✅ Exportados parámetros y constantes ({len(self.simulations)} casos) a {output_dir}/")
