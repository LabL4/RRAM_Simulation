from typing import Dict, Any, Optional, List
import pandas as pd
import itertools
import logging
import os

logger = logging.getLogger(__name__)


# ============================================================================
# PARÁMETROS MATERIALES (HfOx)
# ============================================================================
MATERIAL_DEFAULTS = {
    "cte_red": 0.25e-9,
    "permitividad_relativa_set": 299.6653,
    "permitividad_relativa_reset": 299.9641,
    "generation_energy": 1.02,
    "recombination_energy": 1.55,
    "pb_metal_insul_set": 0.001,
    "pb_metal_insul_reset": 0.0334,
    "recom_enchancement_factor": 3e3,
    "long_decaimiento_concentracion": 1e-9,
    # sigma(T) = sigma_0 / (1 + alpha_T * (T - T_0)).
    # sigma_0 = 1 / (R_ref * atom_size) = 1 / (4.3 * 0.25e-9): a T_0 reproduce
    # exactamente R = 4.3 Ohm, la resistencia de celda usada históricamente.
    "sigma_0": 930232558.1395348,
    "alpha_T": 2e-3,
    "num_filamentos": 2,
    "grosor_filamento": [0, 1],
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
    "I_0_set": 0.0010731,
    "I_0_reset": 0.002505,
}

# ============================================================================
# PARÁMETROS TÉRMICOS
# ============================================================================
THERMAL_DEFAULTS = {
    "init_temp": 300.0,
    "r_termica_no_percola": 8000,  # 10000.0,
    "conductividad_termica_dielectrico": 2.3,
    "conductividad_termica_CF": 15.0,
    "conductividad_termica_aislante": 0.000001,
    "conductividad_termica_electrodo": 5.0,
    "Temperatura_electrodo": 300,
    "factor_generar_calor": 5.5e-5,  # 6e-5,
    "pendiente_temperatura": -28,  # -24.7,
}


# ============================================================================
# PARÁMETROS GEOMÉTRICOS Y DE SIMULACIÓN
# ============================================================================
SIMULATION_DEFAULTS = {
    "device_size_x": 10e-9,  # Ancho entre electrodos, debe corresponder a los dispositivos medidos
    "device_size_y": 40e-9,
    "atom_size": 0.25e-9,  # Se deberia llamar tamaño de red
    "num_trampas": 150,
    "total_simulation_time": 10.0,
    "num_pasos": 10000,
    "voltaje_final": 1.1,
    "voltaje_final_set": 1.1,
    "voltaje_final_reset": 1.4,
    "densidad_vacantes": 4.0,  # vacantes / nm²
    "centros_filamento": None,
}

# ============================================================================
# PARÁMETROS SET / RESET
# ============================================================================
SET_RESET_DEFAULTS = {
    "ocupacion_max_pp_set": 0.8,
    "ocupacion_max_sp_set": 0.45,
    "factor_vecinos_pp_set": 1.0,
    "factor_libre_pp_set": 1.0,
    "factor_vecinos_sp_set": 1.0,
    "factor_libre_sp_set": 0.9,
    "lim_voltage_percolacion": 1.4,
    "compliance_voltage": 0.6,
    "voltaje_gen_oxigeno_pp_1": 1.1,
    "num_oxigenos_pp_reset_1": 2,
    "voltaje_gen_oxigeno_pp_2": 1.2,
    "num_oxigenos_pp_reset_2": 10,
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


# Esta clase no es mas que un contenedor vacío, inicialmente se creó para tener un lugar donde crear nuevos oarámetros a partir de los
# base pero ya se hace en las clases SimulationParameters y SimulationConstants, por lo que esta clase ya no tiene sentido para no tardar en eliminarla, se deja como un contenedor de parámetros que no hace nada.
class SimulationConfig:
    def __init__(self, sim_id: int):
        self.sim_id = sim_id
        # Solo copiamos los valores por defecto. NO calculamos nada derivado.
        self.params = DEFAULT_PARAMS.copy()

    def update(self, modifications: Dict[str, Any]):
        self.params.update(modifications)
        # Ya no llamamos a recalculate_derived porque NO existe.

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

    def add_zip(self, zip_params: Dict[str, List[Any]], fixed_params: Optional[Dict[str, Any]] = None):
        """
        Genera simulaciones emparejando los valores por posición (zip), no por
        producto cartesiano. Todas las listas deben tener la misma longitud.

        Args:
            zip_params: Diccionario de parámetros a emparejar por posición.
                Ej: {"grosor_filamento": [[15,15],[15,10]], "num_trampas": [197, 100]}
            fixed_params: Parámetros fijos adicionales aplicados a todas las sims.
                Ej: {"device_size_y": 15e-9}

        Raises:
            ValueError: Si las listas no tienen la misma longitud.
        """
        longitudes = {k: len(v) for k, v in zip_params.items()}
        if len(set(longitudes.values())) > 1:
            raise ValueError(
                f"Todas las listas en zip_params deben tener la misma longitud. Longitudes encontradas: {longitudes}"
            )

        keys = list(zip_params.keys())
        casos_totales = 0

        for valores in zip(*zip_params.values()):
            mods = dict(zip(keys, valores))
            if fixed_params:
                mods.update(fixed_params)
            sim_id = len(self.simulations)
            self.simulations.append(SimulationConfig.from_base(sim_id, mods))
            casos_totales += 1

        logger.info(f"-> Generadas {casos_totales} combinaciones por zip.")

    def add_sweep(self, sweep_params: Dict[str, List[Any]]):
        # ... (Este método se queda igual) ...
        keys = list(sweep_params.keys())
        values = list(sweep_params.values())
        casos_totales = 0
        for combinacion in itertools.product(*values):
            mods = dict(zip(keys, combinacion))
            sim_id = len(self.simulations)
            self.simulations.append(SimulationConfig.from_base(sim_id, mods))
            casos_totales += 1
        logger.info(f"-> Generadas {casos_totales} combinaciones.")

    def export_to_init_data(self, output_dir: str = "Init_data"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # === LIMPIEZA: SOLO EXPORTAMOS LOS INPUTS ===
        # Quitamos x_size, y_size, etc. porque se calcularán solos al cargar.
        cols_params = [
            "device_size_x",
            "device_size_y",
            "atom_size",
            "num_trampas",
            "total_simulation_time",
            "num_pasos",
            "voltaje_final",
            "voltaje_final_set",
            "voltaje_final_reset",
            "init_temp",
            "densidad_vacantes",
        ]

        # Quitamos conductividades calculadas y diccionarios complejos
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
            "sigma_0",
            "alpha_T",
            "num_filamentos",
            "grosor_filamento",
            "vibration_frequency",
            "gamma",
            "gamma_drift",
            "E_m",
            "I_0_set",
            "I_0_reset",
            "r_termica_no_percola",
            "conductividad_termica_dielectrico",
            "conductividad_termica_CF",
            "conductividad_termica_aislante",
            "conductividad_termica_electrodo",
            "Temperatura_electrodo",
            "factor_generar_calor",
            "pendiente_temperatura",
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
            "centros_filamento",  # ← añadir al final
        ]

        data_params = [{col: sim.params[col] for col in cols_params} for sim in self.simulations]
        data_ctes = [{col: sim.params[col] for col in cols_ctes} for sim in self.simulations]

        df_params = pd.DataFrame(data_params)
        df_ctes = pd.DataFrame(data_ctes)

        df_params.to_csv(os.path.join(output_dir, "simulation_parameters.csv"), index=False)
        df_ctes.to_csv(os.path.join(output_dir, "simulation_constants.csv"), index=False)

        logger.info(f"✅ Exportados parámetros y constantes ({len(self.simulations)} casos) a {output_dir}/")
