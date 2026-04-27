# Excepciones personalizadas para el módulo RRAM

from RRAM import Representate
import numpy as np
import pickle


class NoPercolationException(Exception):
    def __init__(self, message="El sistema no ha percolado al alcanzar el voltaje final"):
        self.message = message
        super().__init__(self.message)


# Excepción para cuando se ha llenado el espacio de simulación
class MaxVacantesException(Exception):
    def __init__(self, k=None, voltage=None):
        if k is not None and voltage is not None:
            message = (
                "El espacio de simulación se ha llenado de vacantes.\n"
                f"Se ha llenado el 90% del espacio de la matriz en la iteración: {k} "
                f"que corresponde al voltaje: {voltage}"
            )
        else:
            message = "El espacio de simulación se ha llenado de vacantes"
        super().__init__(message)


# Excepción para cuando el voltaje de percolación es demasiado alto
class HighPercolationVoltageException(Exception):
    def __init__(self, voltage_percola=None, data_path=None, actual_state=None):
        if voltage_percola is not None:
            message = f"El voltaje de percolación es demasiado alto.\nEl voltaje de percolación es: {voltage_percola}"

            # Guardo el estado de la simulación empleando npz si no data path y actual state none
            if data_path and actual_state is not None:
                np.save(data_path, actual_state)
        else:
            message = "El voltaje de percolación es demasiado alto"
        self.message = message
        super().__init__(self.message)


# Excepción para cuando la resistencia de la parte óhmica es baja y no se reproduce la seguna parte del set
class LowResistanceException(Exception):
    def __init__(self, valor_resistencia=None):
        if valor_resistencia is not None:
            message = (
                "La resistencia del sistema no reproduce la curva de la segunda parte del set\n"
                f"El valor de la resistencia alcanzada es: {valor_resistencia}"
            )
        else:
            message = "La resistencia del sistema no reproduce la curva de la segunda parte del set"
        self.message = message
        super().__init__(self.message)


# Excepción para cuando la resistencia es cero y se ha usado corriente óhmica, donde supuestamente ya hay filamento y no debera ser cero
class NullResistanceException(Exception):
    """Excepción para manejar casos de resistencia nula y detener la simulación"""

    def __init__(
        self,
        simulation_path,
        voltage,
        num_simulation,
        actual_state,
    ):
        self.filename = str(simulation_path) + f"/state_{voltage}_null_resistance.npz"
        self.simulation_path = simulation_path
        self.num_simulation = num_simulation + 1
        self.actual_state = actual_state

        print("Null resistance matrix in ", self.filename)

        # Guardar estado
        np.save(self.filename, self.actual_state)

        super().__init__("Se detectó una resistencia nula. La simulación se detendrá.")


# Excepción para cuando no se han formado todos los filamentos conductores (CF)
class FilamentosNoFormadosException(Exception):
    """Excepción para manejar casos donde no se forman todos los filamentos esperados"""

    def __init__(
        self,
        simulation_path,
        num_simulation,
        actual_state,
        CF_formados,
        CF_esperados,
    ):
        self.filename = str(simulation_path) + f"/simulation_{num_simulation}_not_all_CF_formed.npz"
        self.simulation_path = simulation_path
        self.num_simulation = num_simulation + 1
        self.actual_state = actual_state

        # Generar representación visual

        # Guardar estado
        np.save(self.filename, self.actual_state)

        super().__init__(f"Se esperaba que se formaran {CF_esperados} CF, y se han formado {CF_formados} CF.")


# Excepción para cuando no se han formado todos los filamentos conductores (CF)
class FilamentosNoDestruidosException(Exception):
    """Excepción para manejar casos donde no se destruyen todos los filamentos esperados"""

    def __init__(
        self,
        simulation_path,
        figures_path,
        voltage,
        num_simulation,
        actual_state,
        CF_destruidos,
        CF_esperados,
    ):
        self.filename = str(simulation_path) + f"/state_not_all_CF_destroyed.npz"
        self.voltage = voltage
        self.simulation_path = simulation_path
        self.num_simulation = num_simulation + 1
        self.actual_state = actual_state
        self.figures_path = str(figures_path) + f"/state_not_all_CF_destroyed_{self.num_simulation}.png"

        print(f"Se esperaba que se destruyeran {CF_esperados} y se han destruido {CF_destruidos}.")

        # Generar representación visual
        Representate.RepresentateState(self.actual_state, round(self.voltage, 3), self.figures_path)

        # Guardar estado
        with open(self.filename, "wb") as f:
            pickle.dump({"actual_state": self.actual_state}, f)

        super().__init__(f"Se esperaba que se destruyeran {CF_esperados} y se han destruido {CF_destruidos}.")
