"""Tiempo efectivo acumulativo: mientras el voltaje no cambia, t_ef = n * dt."""


class AcumuladorTiempo:
    """
    Cuenta los pasos consecutivos con el mismo voltaje y devuelve el tiempo efectivo
    que multiplica a las tasas de generación y recombinación.

    - V igual al del paso anterior: n += 1.
    - V distinto: n = 1 (se reinicia la acumulación).
    - V = 0 (reposo): no acumula; devuelve dt y deja n = 0.
    La comparación de V es por igualdad exacta.
    """

    def __init__(self, paso_temporal: float):
        self.paso_temporal = paso_temporal
        self._v_prev: float | None = None
        self._n = 0

    def reiniciar(self) -> None:
        """Pone la cuenta a 0 sin olvidar el voltaje previo: el siguiente paso con el mismo V vale 1·dt."""
        self._n = 0

    def actualizar(self, voltage: float) -> float:
        if voltage == 0:
            self._n = 0
        elif voltage == self._v_prev:
            self._n += 1
        else:
            self._n = 1
        self._v_prev = voltage
        return self.paso_temporal if self._n == 0 else self._n * self.paso_temporal
