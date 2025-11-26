# Excepciones personalizadas para el módulo RRAM

from RRAM import Representate
import pickle


class NoPercolationException(Exception):
    def __init__(
        self, message="El sistema no ha percolado al alcanzar el voltaje final"
    ):
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
    def __init__(self, voltage_percola=None):
        if voltage_percola is not None:
            message = (
                "El voltaje de percolación es demasiado alto.\n"
                f"El voltaje de percolación es: {voltage_percola}"
            )
        else:
            message = "El voltaje de percolación es demasiado alto"
        self.message = message
        super().__init__(self.message)


# Excepción para cuando la resistencia es cero y se ha usado corriente óhmica, donde supuestamente ya hay filamento y no debera ser cero
class NullResistanceException(Exception):
    """Excepción para manejar casos de resistencia nula y detener la simulación"""

    def __init__(
        self,
        simulation_path,
        figures_path,
        voltage,
        num_simulation,
        actual_state,
    ):
        self.filename = str(simulation_path) + f"/state_{voltage}_null_resistance.pkl"
        self.voltage = voltage
        self.simulation_path = simulation_path
        self.num_simulation = num_simulation + 1
        self.actual_state = actual_state
        self.figures_path = (
            str(figures_path) + f"/NULL_resistance_{self.num_simulation}.png"
        )

        print("Null resistance matrix in ", self.filename)

        # Generar representación visual
        Representate.RepresentateState(
            self.actual_state, round(self.voltage, 3), self.figures_path
        )

        # Guardar estado
        with open(self.filename, "wb") as f:
            pickle.dump({"actual_state": self.actual_state}, f)

        super().__init__("Se detectó una resistencia nula. La simulación se detendrá.")
