"""
Controlador de la forma de onda de voltaje aplicada en una fase.

Sustituye al vector rígido `vector_ddp = np.arange(...)` de las fases: en vez
de precalcular la rampa completa, el bucle de la fase pide en cada paso el
voltaje siguiente con `controller.next(medidas)`, y el controlador decide en
función de las medidas del paso anterior (corriente, vacantes, filamentos,
percolación) si sigue en rampa, congela el voltaje, o encadena otro segmento.

Configuración
-------------
Una forma de onda es una lista de segmentos ``(modo, opciones)``:

    [("rampa", {"hasta_I": 1e-4}), ("constante", {})]

Modos:
    - ``"rampa"``: el voltaje avanza `paso_potencial` por paso (con signo
      `sentido`), partiendo del último voltaje aplicado.
    - ``"constante"``: mantiene el último voltaje aplicado, u ``opciones["V"]``
      si se da explícito. ``opciones["pasos"]`` limita su duración: agotados,
      el controlador queda `terminado` (fin de fase).

Condiciones de transición (válidas en cualquier modo; la primera que se cumple
salta al siguiente segmento, evaluadas sobre las medidas del paso ANTERIOR):
    - ``hasta_I``:           |I_total| >= valor [A]
    - ``hasta_V``:           |V aplicado| >= valor [V]
    - ``hasta_vacantes``:    nº total de vacantes >= valor
    - ``hasta_filamentos``:  nº de filamentos creados >= valor
    - ``hasta_percolacion``: True → dispara al percolar el sistema

`segmentos=None` es el MODO LEGACY: una única rampa sin condiciones, que
reproduce exactamente `np.arange(v_inicial, ..., paso)` indexado por k
(V(k) = v_inicial + k * paso, misma aritmética flotante que np.arange).

El viaje de la configuración es el mismo que el resto de constantes: columna
`waveform_pp_set` en `simulation_constants.csv` (string parseado con
ast.literal_eval por `SimulationConstants.from_dict`) → dataclass → fase.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

MODOS_VALIDOS = ("rampa", "constante")

#: Condiciones de transición admitidas en las opciones de un segmento.
CONDICIONES_VALIDAS = (
    "hasta_I",
    "hasta_V",
    "hasta_vacantes",
    "hasta_filamentos",
    "hasta_percolacion",
)

#: Opciones de segmento que NO son condiciones de transición.
OPCIONES_VALIDAS = ("V", "pasos")


@dataclass
class Medidas:
    """Medidas del paso ANTERIOR con las que el controlador decide el voltaje."""

    I_total: float
    V_anterior: float
    n_vacantes: int
    n_filamentos: int
    percola: bool
    k: int


def parsear_segmentos(raw: Any) -> Optional[List[Tuple[str, dict]]]:
    """
    Normaliza y valida la configuración de segmentos venida del CSV.

    Acepta None (modo legacy), o una lista de pares ``(modo, opciones)`` donde
    los pares pueden llegar como tuplas o listas (ast.literal_eval devuelve
    cualquiera de las dos según cómo se escribiera el CSV).

    Raises:
        ValueError: si la estructura, un modo o una clave de opciones no son válidos.
    """
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)) or len(raw) == 0:
        raise ValueError(f"waveform: se esperaba una lista de segmentos (modo, opciones), llegó: {raw!r}")

    segmentos: List[Tuple[str, dict]] = []
    for i, seg in enumerate(raw):
        if not isinstance(seg, (list, tuple)) or len(seg) not in (1, 2):
            raise ValueError(f"waveform: el segmento {i} debe ser (modo, opciones), llegó: {seg!r}")
        modo = seg[0]
        opciones = dict(seg[1]) if len(seg) == 2 and seg[1] is not None else {}
        if modo not in MODOS_VALIDOS:
            raise ValueError(f"waveform: modo desconocido {modo!r} en el segmento {i}. Válidos: {MODOS_VALIDOS}")
        desconocidas = [c for c in opciones if c not in CONDICIONES_VALIDAS + OPCIONES_VALIDAS]
        if desconocidas:
            raise ValueError(
                f"waveform: claves desconocidas {desconocidas} en el segmento {i}. "
                f"Condiciones: {CONDICIONES_VALIDAS}; opciones: {OPCIONES_VALIDAS}"
            )
        segmentos.append((modo, opciones))
    return segmentos


@dataclass
class _Transicion:
    segmento: int  # índice del segmento que se ABANDONA
    k: int  # paso en el que se disparó
    V: float  # voltaje aplicado en ese paso
    condicion: str  # condición que la disparó


class VoltageController:
    """
    Máquina de segmentos que devuelve el voltaje de cada paso de una fase.

    Args:
        segmentos: Lista (modo, opciones) ya parseada por `parsear_segmentos`,
            o None / crudo del CSV (se parsea aquí). None = rampa legacy.
        paso_potencial: Incremento de voltaje por paso en modo rampa [V].
        v_inicial: Voltaje del primer paso (k=0) [V].
        sentido: +1 rampa hacia voltajes crecientes (SET), -1 decrecientes.
    """

    def __init__(
        self,
        segmentos: Any = None,
        paso_potencial: float = 0.0,
        v_inicial: float = 0.0,
        sentido: int = +1,
        v_objetivo: Optional[float] = None,
    ):
        self.segmentos = parsear_segmentos(segmentos)
        if self.segmentos is None:
            # Modo legacy: una rampa infinita sin condiciones (la fase corta por
            # su propia condición voltage >= voltaje_final, como siempre).
            self.segmentos = [("rampa", {})]
            self.legacy = True
        else:
            self.legacy = False

        self.paso_potencial = paso_potencial
        self.v_inicial = v_inicial
        self.sentido = sentido
        self.v_objetivo = v_objetivo

        self._idx = 0  # segmento activo
        self._pasos_en_segmento = 0  # pasos ya consumidos en el segmento activo
        self._v_base_segmento = v_inicial  # V con el que arrancó el segmento activo
        # Desplazamiento del primer paso de una rampa. El PRIMER segmento arranca en
        # v_inicial (offset 0), pero una rampa que RELEVA a otro segmento debe avanzar
        # ya en su primer paso (offset 1): si no, repetiría el voltaje que dejó el
        # segmento anterior y una meseta de N pasos saldría de N+1 pasos planos,
        # robándole un paso de potencial a la rampa.
        self._offset_rampa = 0
        self._V = v_inicial  # último voltaje devuelto
        self.terminado = False
        self.transiciones: List[_Transicion] = []

    # ------------------------------------------------------------------ API

    def en_rampa(self) -> bool:
        """True si el segmento activo es una rampa (o el modo legacy)."""
        return self.segmentos[self._idx][0] == "rampa"

    def objetivo_alcanzado(self, V: float) -> bool:
        """
        True si la rampa ha llegado (o pasado) el voltaje objetivo de la fase.

        La comparación se hace con el signo del sentido de avance, de modo que
        sirve para las cuatro combinaciones sin casos especiales:
        subida a +V_set, bajada a 0, bajada a -V_reset y subida a 0.
        """
        if self.v_objetivo is None:
            return False
        # Tolerancia para absorber el error de coma flotante acumulado por la rampa:
        # tras cientos de pasos, un objetivo de 0 V se alcanza como -1e-17 y una
        # comparación estricta daría False, haciendo que la rampa se pasase un paso
        # del objetivo. La tolerancia es millonésimas de paso, así que nunca puede
        # adelantar una decisión real.
        tol = abs(self.paso_potencial) * 1e-6
        return self.sentido * (V - self.v_objetivo) >= -tol

    def presupuesto(self) -> int:
        """
        COTA SUPERIOR del número de pasos que puede durar la fase.

        Los arrays de datos se reservan antes del bucle, así que hace falta saber
        cuántas filas como máximo se van a producir. Se recorre la forma de onda
        suponiendo que cada rampa llega tan lejos como podría:

          - rampa sin destino fijo  -> los pasos necesarios para llegar a v_objetivo
          - rampa con 'hasta_V'     -> los necesarios para llegar a ese umbral
          - rampa con 'pasos'       -> esos pasos
          - meseta con 'pasos'      -> esos pasos (el voltaje no avanza, salvo 'V' explícito)
          - meseta sin 'pasos'      -> 0: no tiene duración propia, absorbe el sobrante
                                       que dejen las rampas que cortaron antes de tiempo

        Las condiciones estocásticas (hasta_I, hasta_percolacion, hasta_vacantes,
        hasta_filamentos) solo pueden ACORTAR un tramo, nunca alargarlo, así que el
        recorrido es siempre cota superior. Pasarse es inofensivo (las filas
        sobrantes se recortan); quedarse corto dejaría la rampa sin llegar a su
        voltaje final, por eso hay que recorrer los segmentos en vez de sumar en
        plano: una meseta con 'V' explícito puede mover el voltaje HACIA ATRÁS y
        obligar a una rampa posterior a recorrer dos veces el mismo tramo.
        """
        if self.v_objetivo is None:
            raise ValueError("presupuesto() requiere v_objetivo en el constructor.")

        V = float(self.v_inicial)
        total = 0

        for modo, opciones in self.segmentos:
            if modo == "constante":
                if "V" in opciones:
                    V = float(opciones["V"])
                total += int(opciones.get("pasos", 0))
                continue

            # rampa
            if "pasos" in opciones:
                n = int(opciones["pasos"])
                V = V + n * self.paso_potencial * self.sentido
            else:
                destino = self._umbral_con_signo(opciones["hasta_V"]) if "hasta_V" in opciones else self.v_objetivo
                n = int(math.ceil(abs(destino - V) / self.paso_potencial))
                V = destino
            total += n

        return total

    def _umbral_con_signo(self, valor: float) -> float:
        """
        Convierte un umbral de voltaje escrito como MAGNITUD en el CSV al valor
        con signo que corresponde al rango de la fase.

        El usuario escribe `hasta_V: 0.8` sin preocuparse del signo; en una fase de
        RESET eso significa -0.8 V. El signo se toma del extremo no nulo del rango
        (v_inicial o v_objetivo), no del sentido de avance: en SP_set el voltaje es
        positivo aunque la rampa baje.
        """
        referencia = self.v_inicial if self.v_inicial != 0 else (self.v_objetivo or 0.0)
        signo = -1.0 if referencia < 0 else 1.0
        return abs(float(valor)) * signo

    def next(self, medidas: Medidas) -> float:
        """
        Devuelve el voltaje a aplicar en el paso `medidas.k`.

        Evalúa primero las condiciones del segmento activo con las medidas del
        paso anterior; si alguna se cumple, salta al siguiente segmento (o
        marca `terminado` si era el último / se agotaron sus `pasos`).
        """
        modo, opciones = self.segmentos[self._idx]

        # 1. ¿Toca transición? (no en el primer paso: aún no hay medidas reales)
        if medidas.k > 0:
            condicion = self._condicion_cumplida(opciones, medidas)
            agotado = "pasos" in opciones and self._pasos_en_segmento >= int(opciones["pasos"])
            if condicion is not None or agotado:
                self.transiciones.append(
                    _Transicion(
                        segmento=self._idx,
                        k=medidas.k,
                        V=self._V,
                        condicion=condicion if condicion is not None else "pasos",
                    )
                )
                if self._idx + 1 < len(self.segmentos):
                    self._idx += 1
                    self._pasos_en_segmento = 0
                    self._v_base_segmento = self._V
                    modo, opciones = self.segmentos[self._idx]
                    # Una rampa que releva a otro segmento avanza ya en su primer paso.
                    self._offset_rampa = 1
                    logger.info(
                        f"waveform: transición al segmento {self._idx} ({modo}) en k={medidas.k}, "
                        f"V={self._V:.5f} V, condición={self.transiciones[-1].condicion}"
                    )
                else:
                    # Último segmento agotado: la fase debe terminar.
                    self.terminado = True
                    logger.info(
                        f"waveform: último segmento agotado en k={medidas.k} "
                        f"(condición={self.transiciones[-1].condicion}). Fin de la forma de onda."
                    )
                    return self._V

        # 2. Voltaje del paso según el modo activo.
        if modo == "rampa":
            # Misma aritmética que np.arange: base + n * paso (no acumulación),
            # para que el modo legacy sea bit a bit idéntico al vector_ddp.
            n = self._pasos_en_segmento + self._offset_rampa
            self._V = self._v_base_segmento + n * self.paso_potencial * self.sentido
        else:  # constante
            self._V = float(opciones["V"]) if "V" in opciones else self._v_base_segmento

        self._pasos_en_segmento += 1
        return self._V

    def estado(self) -> dict:
        """Estado JSON-serializable para metadata y phase_state (auditoría/reanudación)."""
        return {
            "legacy": self.legacy,
            "segmentos": [[modo, opciones] for modo, opciones in self.segmentos],
            "segmento_activo": self._idx,
            "V_actual": float(self._V),
            "terminado": self.terminado,
            "transiciones": [
                {"segmento": t.segmento, "k": t.k, "V": float(t.V), "condicion": t.condicion}
                for t in self.transiciones
            ],
        }

    # ------------------------------------------------------------- interno

    def _condicion_cumplida(self, opciones: dict, medidas: Medidas) -> Optional[str]:
        """Devuelve el nombre de la primera condición cumplida, o None."""
        if "hasta_I" in opciones and abs(medidas.I_total) >= float(opciones["hasta_I"]):
            return "hasta_I"
        if "hasta_V" in opciones:
            # Comparación con el sentido de avance: en una rampa que SUBE dispara al
            # superar el umbral, y en una que BAJA al caer por debajo. Comparar
            # magnitudes (abs >= abs) solo funcionaría en las rampas que se alejan de
            # cero y dispararía desde el primer paso en SP_set / SP_reset.
            umbral = self._umbral_con_signo(opciones["hasta_V"])
            if self.sentido * (medidas.V_anterior - umbral) >= 0:
                return "hasta_V"
        if "hasta_vacantes" in opciones and medidas.n_vacantes >= int(opciones["hasta_vacantes"]):
            return "hasta_vacantes"
        if "hasta_filamentos" in opciones and medidas.n_filamentos >= int(opciones["hasta_filamentos"]):
            return "hasta_filamentos"
        if opciones.get("hasta_percolacion") and medidas.percola:
            return "hasta_percolacion"
        return None
