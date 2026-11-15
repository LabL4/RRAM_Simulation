# Excepciones personalizadas para el módulo RRAM

from pathlib import Path
import logging

logger = logging.getLogger(__name__)


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


class ExcepcionConEstado(Exception):
    """
    Base de las excepciones de simulación que dejan un estado de diagnóstico.

    Solo TRANSPORTA los datos del fallo (estado, voltaje, rutas destino). No escribe nada en
    disco ni dibuja: de eso se encarga `run_cycle`, que es quien captura la excepción
    (ver `run_cycle._guardar_estado_fallo`). `num_simulation` es el número ya "guardado"
    (N+1 respecto al índice del CSV), tal como llega de las fases: no se vuelve a sumar 1.
    """

    #: Ruta donde `run_cycle` guarda `actual_state` (np.savez_compressed). None = no se guarda.
    ruta_estado_fallo: Path | None = None
    #: Ruta de la figura del estado fallido. None = no se dibuja.
    ruta_figura_fallo: Path | None = None

    def __init__(self, mensaje, simulation_path=None, num_simulation=None, actual_state=None, voltage=None, estado_fase=None):
        self.simulation_path = None if simulation_path is None else Path(simulation_path)
        self.num_simulation = num_simulation
        self.actual_state = actual_state
        self.voltage = voltage
        #: Variables con las que terminó la fase (mismo dict que devuelve si acaba bien). `run_cycle`
        #: las guarda como phase_state_N_<fase>_fallida y las incluye en la metadata parcial.
        self.estado_fase = estado_fase
        super().__init__(mensaje)


class NoPercolationException(ExcepcionConEstado):
    """La fase de formación terminó sin que el sistema percolase."""

    def __init__(self, message="El sistema no ha percolado al alcanzar el voltaje final", estado_fase=None):
        self.message = message
        super().__init__(message, estado_fase=estado_fase)


class PercolacionSinFilamentoException(ExcepcionConEstado):
    """
    El sistema percola pero ningún filamento cuenta como formado.

    No debería ocurrir: la celda de contacto con el electrodo izquierdo siempre cae en la banda de
    algún filamento (`topologia_filamentos`). Si ocurre, el estado o las bandas son incoherentes.
    """

    def __init__(self, simulation_path, voltage, num_simulation, actual_state):
        super().__init__(
            "El sistema percola pero ningún filamento está formado: estado o bandas de filamento incoherentes.",
            simulation_path=simulation_path,
            num_simulation=num_simulation,
            actual_state=actual_state,
            voltage=voltage,
        )
        self.ruta_estado_fallo = self.simulation_path / f"simulation_{num_simulation}_percolacion_sin_filamento.npz"


# Excepción para cuando la resistencia es cero y se ha usado corriente óhmica, donde supuestamente ya hay filamento y no debera ser cero
class NullResistanceException(ExcepcionConEstado):
    """Excepción para manejar casos de resistencia nula y detener la simulación"""

    def __init__(self, simulation_path, voltage, num_simulation, actual_state):
        super().__init__(
            "Se detectó una resistencia nula. La simulación se detendrá.",
            simulation_path=simulation_path,
            num_simulation=num_simulation,
            actual_state=actual_state,
            voltage=voltage,
        )
        self.ruta_estado_fallo = self.simulation_path / f"state_{voltage}_null_resistance.npz"


# Excepción para cuando no se han formado todos los filamentos conductores (CF)
class FilamentosNoFormadosException(ExcepcionConEstado):
    """Excepción para manejar casos donde no se forman todos los filamentos esperados"""

    def __init__(self, simulation_path, num_simulation, actual_state, CF_formados, CF_esperados, estado_fase=None):
        super().__init__(
            f"Se esperaba que se formaran {CF_esperados} CF, y se han formado {CF_formados} CF.",
            simulation_path=simulation_path,
            num_simulation=num_simulation,
            actual_state=actual_state,
            estado_fase=estado_fase,
        )
        self.CF_formados = CF_formados
        self.CF_esperados = CF_esperados
        self.ruta_estado_fallo = self.simulation_path / f"simulation_{num_simulation}_not_all_CF_formed.npz"


# Excepción para cuando no se han destruido todos los filamentos conductores (CF)
class FilamentosNoDestruidosException(ExcepcionConEstado):
    """Excepción para manejar casos donde no se destruyen todos los filamentos esperados"""

    def __init__(
        self, simulation_path, figures_path, voltage, num_simulation, actual_state, CF_destruidos, CF_esperados, estado_fase=None
    ):
        super().__init__(
            f"Se esperaba que se destruyeran {CF_esperados} y se han destruido {CF_destruidos}.",
            simulation_path=simulation_path,
            num_simulation=num_simulation,
            actual_state=actual_state,
            voltage=voltage,
            estado_fase=estado_fase,
        )
        self.CF_destruidos = CF_destruidos
        self.CF_esperados = CF_esperados
        self.ruta_estado_fallo = self.simulation_path / "state_not_all_CF_destroyed.npz"
        self.ruta_figura_fallo = Path(figures_path) / f"state_not_all_CF_destroyed_{num_simulation}.png"
        logger.info(f"Se esperaba que se destruyeran {CF_esperados} y se han destruido {CF_destruidos}.")


# Excepción para cuando el reset no tiene de dónde sacar el límite de iones de oxígeno
class LimiteOxigenosNoDefinidoException(Exception):
    def __init__(self):
        message = (
            "No hay límite de iones de oxígeno para el reset: el estado del set no trae "
            "'vacantes_generadas_set' y 'max_oxigenos_reset' no está fijado en simulation_config.py."
        )
        logger.error(message)
        super().__init__(message)
