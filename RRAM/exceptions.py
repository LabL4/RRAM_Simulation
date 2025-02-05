
# Excepciones personalizadas para el módulo RRAM

import logging

# Configurar el logging
logging.basicConfig(filename='simulation.log', level=logging.ERROR,
                    format='% (asctime)s - %(levelname)s - %(message)s')


# Excepción para cuando no ha percolado el sistema
class NoPercolationException(Exception):
    def __init__(self, message="El sistema no ha percolado al alcanzar el voltaje final"):
        self.message = message
        super().__init__(self.message)


# Excepción para cuando se ha llenado el espacio de simulación
class MaxVacantesExcception(Exception):
    def __init__(self, message="El espacio de simulación se ha llenado de vacantes"):
        self.message = message
        super().__init__(self.message)
