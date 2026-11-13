"""Tiempo efectivo acumulativo: mientras el voltaje no cambia, t_ef = n * dt."""

import numpy as np


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


class RelojesIones:
    """
    Reloj propio de cada ion de oxígeno: matriz entera con los pasos acumulados por celda
    (el tiempo de un ion es reloj · dt). Invariante: reloj == 0 donde oxygen_state == 0;
    la presencia de ion se mira SIEMPRE en oxygen_state (un ion recién movido o generado
    también tiene reloj 0).

    - V = 0 o V distinto del paso anterior: todos los relojes a 0.
    - V distinto (≠ 0): además, tras reiniciar, cada ion existente suma 1.
    - V igual al anterior: cada ion existente suma 1.
    El reinicio individual de un ion que se mueve lo hace el núcleo de movimiento.
    La comparación de V es por igualdad exacta.
    """

    def __init__(self, reloj_inicial: np.ndarray, v_prev: float | None = None):
        self.reloj = np.array(reloj_inicial, dtype=np.int32)  # copia: no comparte memoria con quien lo pasa
        self.v_prev = v_prev

    def actualizar(self, voltage: float, oxygen_state: np.ndarray) -> None:
        """Paso de reloj ANTES de generar: solo cuentan los iones que ya existían."""
        if voltage == 0 or voltage != self.v_prev:
            self.reloj.fill(0)
        if voltage != 0:
            self.reloj += oxygen_state  # oxygen_state es 0/1: suma 1 solo donde hay ion
        self.v_prev = voltage

    def limpiar(self, oxygen_state: np.ndarray) -> None:
        """Borra el reloj de toda celda sin ion (p. ej. iones recombinados)."""
        self.reloj *= oxygen_state

    def estado(self) -> tuple[np.ndarray, float | None]:
        """(reloj, v_prev) para que la fase siguiente herede los relojes."""
        return self.reloj, self.v_prev
