"""
Protocolo de voltaje de la simulación: TODO lo que decide el voltaje aplicado.

Este módulo es la única fuente de verdad del voltaje. Las fases no leen nada de
voltaje de `params` ni de `sim_ctes`: piden a `ProtocoloVoltaje` un controlador
para su etapa y, en cada paso, le piden el voltaje con `controller.next(medidas)`.

Principio: NADA se calcula de forma implícita. Cada etapa se describe con una
lista de segmentos en la que el usuario escribe cada voltaje, cada paso y cada
duración. No hay pasos derivados de `num_pasos`, ni "un paso por debajo" en las
fronteras entre etapas, ni signos inferidos: el RESET se escribe en negativo.

Segmentos
---------
Cada segmento es un par ``(modo, opciones)``:

    ("rampa",     {"desde": 0.0, "hasta": 1.1, "dV": 1.1e-4})
    ("constante", {"V": 1.1, "pasos": 500})

``rampa`` (obligatorios ``desde``, ``hasta``, ``dV``):
    - ``desde``: voltaje del primer punto [V], o ``"anterior"``.
    - ``hasta``: voltaje final [V]. La rampa termina al llegar a él; si ``dV`` no
      divide el tramo, el último punto se recorta a ``hasta`` exacto.
    - ``dV``:    tamaño del paso [V], siempre > 0. El sentido sale de comparar
      ``desde`` con ``hasta``.
    - ``pasos`` (opcional): máximo de puntos del segmento.

``constante`` (obligatorios ``V`` y ``pasos``):
    - ``V``:     voltaje de la meseta [V], o ``"anterior"``.
    - ``pasos``: número de puntos de la meseta.

``"anterior"`` es la única palabra especial: "donde quedó el segmento anterior"
(o, en el primer segmento de una etapa, donde terminó la etapa anterior).
    - En una meseta, la meseta mantiene ese voltaje.
    - En una rampa, la rampa CONTINÚA desde él: su primer punto es un ``dV`` más
      allá, para no repetir el punto que ya dejó el segmento anterior.
Se necesita cuando una condición congela el voltaje en un valor que no se
conoce de antemano (compliance, percolación).

Condiciones de salida anticipada (opcionales, en cualquier modo; la primera
que se cumple abandona el segmento; se evalúan con las medidas del PASO
ANTERIOR):
    - ``hasta_I``:           |I_total| >= valor [A]
    - ``hasta_vacantes``:    nº total de vacantes >= valor
    - ``hasta_filamentos``:  nº de filamentos >= valor
    - ``hasta_percolacion``: True → el sistema percola
    - ``exigir``: True → si el segmento termina por su final natural (``hasta``
      o ``pasos``) sin que se cumpla ninguna condición, la etapa se ABORTA con
      `CondicionExigidaNoCumplida`.

La etapa termina cuando termina su último segmento.

Configuración
-------------
`ProtocoloVoltaje` agrupa las cuatro etapas (``pp_set``, ``sp_set``,
``pp_reset``, ``sp_reset``), todas obligatorias. Viaja por una columna propia,
``protocolo_voltaje``, del CSV ``Init_data/simulation_voltage.csv``, como un
diccionario en texto (``ast.literal_eval``).
"""

from __future__ import annotations

import ast
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

#: Etapas del ciclo, en orden.
FASES = ("pp_set", "sp_set", "pp_reset", "sp_reset")

#: Palabra especial: "donde quedó el segmento o la etapa anterior".
ANTERIOR = "anterior"

MODOS_VALIDOS = ("rampa", "constante")

#: Condiciones de salida anticipada admitidas en las opciones de un segmento.
CONDICIONES_VALIDAS = ("hasta_I", "hasta_vacantes", "hasta_filamentos", "hasta_percolacion")

#: Claves admitidas por modo (además de las condiciones y `exigir`).
CLAVES_POR_MODO = {
    "rampa": {"obligatorias": ("desde", "hasta", "dV"), "opcionales": ("pasos",)},
    "constante": {"obligatorias": ("V", "pasos"), "opcionales": ()},
}

#: Transiciones que se escriben en el log a nivel INFO; el resto va a DEBUG
#: (un tren de cientos de pulsos no debe inundar el log).
MAX_TRANSICIONES_LOG = 10

#: Transiciones que se guardan en la metadata (las primeras y las últimas).
MAX_TRANSICIONES_METADATA = 200


class CondicionExigidaNoCumplida(RuntimeError):
    """Un segmento con ``exigir: True`` terminó sin que se cumpliera su condición."""


@dataclass
class Medidas:
    """Medidas del paso ANTERIOR con las que el controlador decide el voltaje."""

    I_total: float
    n_vacantes: int
    n_filamentos: int
    percola: bool
    k: int


# ======================================================================
# Validación de segmentos
# ======================================================================


def _es_numero(valor: Any) -> bool:
    return isinstance(valor, (int, float)) and not isinstance(valor, bool) and math.isfinite(valor)


def _es_voltaje(valor: Any) -> bool:
    return valor == ANTERIOR or _es_numero(valor)


def parsear_segmentos(raw: Any, fase: str = "?") -> List[Tuple[str, dict]]:
    """
    Normaliza y valida la lista de segmentos de una etapa.

    Los pares pueden llegar como tuplas o listas (``ast.literal_eval`` devuelve
    cualquiera de las dos).

    Raises:
        ValueError: con un mensaje que dice qué segmento y qué campo fallan.
    """
    if not isinstance(raw, (list, tuple)) or len(raw) == 0:
        raise ValueError(f"{fase}: se esperaba una lista no vacía de segmentos (modo, opciones), llegó: {raw!r}")

    segmentos: List[Tuple[str, dict]] = []
    for i, seg in enumerate(raw):
        donde = f"{fase}, segmento {i}"
        if not isinstance(seg, (list, tuple)) or len(seg) != 2 or not isinstance(seg[1], dict):
            raise ValueError(f"{donde}: debe ser un par (modo, {{opciones}}), llegó: {seg!r}")
        modo, opciones = seg[0], dict(seg[1])
        if modo not in MODOS_VALIDOS:
            raise ValueError(f"{donde}: modo desconocido {modo!r}. Válidos: {MODOS_VALIDOS}")

        claves = CLAVES_POR_MODO[modo]
        admitidas = claves["obligatorias"] + claves["opcionales"] + CONDICIONES_VALIDAS + ("exigir",)
        desconocidas = [c for c in opciones if c not in admitidas]
        if desconocidas:
            raise ValueError(f"{donde} ({modo}): claves no admitidas {desconocidas}. Admitidas: {admitidas}")
        faltan = [c for c in claves["obligatorias"] if c not in opciones]
        if faltan:
            raise ValueError(f"{donde} ({modo}): faltan campos obligatorios {faltan}")

        if modo == "rampa":
            if not _es_voltaje(opciones["desde"]):
                raise ValueError(f"{donde}: 'desde' debe ser un número o {ANTERIOR!r}, llegó {opciones['desde']!r}")
            if not _es_numero(opciones["hasta"]):
                raise ValueError(f"{donde}: 'hasta' debe ser un número, llegó {opciones['hasta']!r}")
            if not _es_numero(opciones["dV"]) or opciones["dV"] <= 0:
                raise ValueError(f"{donde}: 'dV' debe ser un número > 0 (el sentido lo dan desde/hasta), llegó {opciones['dV']!r}")
            if _es_numero(opciones["desde"]) and opciones["desde"] == opciones["hasta"]:
                raise ValueError(f"{donde}: 'desde' y 'hasta' son iguales ({opciones['hasta']}); para mantener un voltaje usa 'constante'")
        else:
            if not _es_voltaje(opciones["V"]):
                raise ValueError(f"{donde}: 'V' debe ser un número o {ANTERIOR!r}, llegó {opciones['V']!r}")

        if "pasos" in opciones and (not isinstance(opciones["pasos"], int) or isinstance(opciones["pasos"], bool) or opciones["pasos"] < 1):
            raise ValueError(f"{donde}: 'pasos' debe ser un entero >= 1, llegó {opciones['pasos']!r}")

        for c in ("hasta_I", "hasta_vacantes", "hasta_filamentos"):
            if c in opciones and (not _es_numero(opciones[c]) or opciones[c] <= 0):
                raise ValueError(f"{donde}: '{c}' debe ser un número > 0, llegó {opciones[c]!r}")
        if "hasta_percolacion" in opciones and opciones["hasta_percolacion"] is not True:
            raise ValueError(f"{donde}: 'hasta_percolacion' solo admite True (quítalo si no lo quieres), llegó {opciones['hasta_percolacion']!r}")
        if "exigir" in opciones:
            if opciones["exigir"] is not True:
                raise ValueError(f"{donde}: 'exigir' solo admite True (quítalo si no lo quieres), llegó {opciones['exigir']!r}")
            if not any(c in opciones for c in CONDICIONES_VALIDAS):
                raise ValueError(f"{donde}: 'exigir' necesita al menos una condición {CONDICIONES_VALIDAS}")

        segmentos.append((modo, opciones))
    return segmentos


# ======================================================================
# Constructores de segmentos (azúcar para el notebook; siguen siendo explícitos)
# ======================================================================


def rampa(desde: float | str, hasta: float, dV: float, **opciones) -> Tuple[str, dict]:
    """``("rampa", {"desde": desde, "hasta": hasta, "dV": dV, **opciones})``."""
    return ("rampa", {"desde": desde, "hasta": hasta, "dV": dV, **opciones})


def constante(V: float | str, pasos: int, **opciones) -> Tuple[str, dict]:
    """``("constante", {"V": V, "pasos": pasos, **opciones})``."""
    return ("constante", {"V": V, "pasos": pasos, **opciones})


def tren_pulsos(V_alto: float, pasos_alto: int, V_bajo: float, pasos_bajo: int, n: int) -> List[Tuple[str, dict]]:
    """`n` pulsos: meseta a `V_alto` durante `pasos_alto` y a `V_bajo` durante `pasos_bajo`."""
    return [constante(V_alto, pasos_alto), constante(V_bajo, pasos_bajo)] * n


# ======================================================================
# Análisis estático: cuántos puntos puede durar una etapa y dónde puede acabar
# ======================================================================


def _num_puntos_rampa(v_ini: float, hasta: float, dV: float) -> int:
    """Puntos de una rampa cuyo primer punto es `v_ini` y el último `hasta` (recortado)."""
    distancia = abs(hasta - v_ini)
    # Tolerancia: 1.1/1e-4 no da un entero exacto en coma flotante.
    return int(math.ceil(distancia / dV - 1e-9)) + 1


def _analizar_etapa(segmentos: List[Tuple[str, dict]], entrada: Optional[Tuple[float, float]]) -> Tuple[int, Tuple[float, float]]:
    """
    Cota superior de puntos de una etapa e intervalo de voltajes en que puede acabar.

    `entrada` es el intervalo (lo, hi) en que pudo acabar la etapa anterior
    (None si no la hay). Se propaga un intervalo porque una condición puede
    cortar un segmento en cualquier punto de su recorrido, y entonces el
    siguiente segmento con ``"anterior"`` arranca en un valor desconocido. La
    cota de cada segmento se toma con el peor caso de ese intervalo; la suma es
    cota superior porque cada segmento está acotado por sí mismo,
    independientemente de cómo acabaran los anteriores.
    """
    total = 0
    actual = entrada  # intervalo (lo, hi) del último voltaje aplicado

    for modo, op in segmentos:
        corta = any(c in op for c in CONDICIONES_VALIDAS)
        if modo == "constante":
            if op["V"] == ANTERIOR:
                if actual is None:
                    raise ValueError("'anterior' sin segmento ni etapa anterior")
                v_seg = actual
            else:
                v_seg = (float(op["V"]), float(op["V"]))
            total += int(op["pasos"])
            actual = v_seg
            continue

        # rampa
        hasta, dV = float(op["hasta"]), float(op["dV"])
        if op["desde"] == ANTERIOR:
            if actual is None:
                raise ValueError("'anterior' sin segmento ni etapa anterior")
            # Continúa: su primer punto es un dV más allá del anterior.
            peor = max(abs(hasta - actual[0]), abs(hasta - actual[1]))
            n = max(0, int(math.ceil(peor / dV - 1e-9)))
            inicio = actual
        else:
            desde = float(op["desde"])
            n = _num_puntos_rampa(desde, hasta, dV)
            inicio = (desde, desde)
        if "pasos" in op:
            n = min(n, int(op["pasos"]))
        total += n

        if corta or "pasos" in op:
            # Puede acabar en cualquier punto entre su inicio y `hasta`.
            actual = (min(inicio[0], hasta), max(inicio[1], hasta))
        else:
            actual = (hasta, hasta)

    return total, actual  # type: ignore[return-value]


# ======================================================================
# Controlador de una etapa
# ======================================================================


@dataclass
class _Transicion:
    segmento: int  # índice del segmento que se ABANDONA
    k: int  # paso en el que se disparó (el primer punto del segmento siguiente)
    V: float  # último voltaje aplicado en el segmento abandonado
    condicion: str  # condición que la disparó, o "hasta" / "pasos" (final natural)


class VoltageController:
    """
    Máquina de segmentos que devuelve el voltaje de cada paso de UNA etapa.

    No se construye a mano en las fases: se pide a `ProtocoloVoltaje.controlador`.

    Args:
        fase: Nombre de la etapa (para logs, errores y metadata).
        segmentos: Lista de segmentos (cruda o ya parseada).
        v_previo: Último voltaje aplicado por la etapa anterior, o None si no la
            hay. Resuelve un ``"anterior"`` en el primer segmento y se usa para el
            informe de enganche; nunca para calcular nada más.
        previsualizacion: True desactiva ``exigir`` y los mensajes de log (la
            previsualización recorre la forma de onda sin física).
    """

    def __init__(self, fase: str, segmentos: Any, v_previo: Optional[float] = None, previsualizacion: bool = False):
        self.fase = fase
        self.segmentos = parsear_segmentos(segmentos, fase)
        self.v_previo = None if v_previo is None else float(v_previo)
        self.previsualizacion = previsualizacion

        if self.segmentos[0][1].get("desde", self.segmentos[0][1].get("V")) == ANTERIOR and self.v_previo is None:
            raise ValueError(f"{fase}: el primer segmento usa {ANTERIOR!r} pero no hay etapa anterior de la que continuar")

        entrada = None if self.v_previo is None else (self.v_previo, self.v_previo)
        self._presupuesto, _ = _analizar_etapa(self.segmentos, entrada)

        self._idx = 0
        self.terminado = False
        self.completada = False
        self.fin_por: Optional[str] = None
        self.transiciones: List[_Transicion] = []
        self.anterior_resuelto: List[Dict[str, Any]] = []
        self.puntos = 0
        self.V_inicio: Optional[float] = None
        self._V: Optional[float] = self.v_previo  # último voltaje aplicado
        self._activar_segmento(k=0)

    # ------------------------------------------------------------------ API

    def presupuesto(self) -> int:
        """
        COTA SUPERIOR del número de puntos (filas) de la etapa.

        Sirve para reservar los arrays de datos antes del bucle. Las condiciones
        de salida anticipada solo pueden acortar un segmento, nunca alargarlo.
        La fase recorre ``range(presupuesto + 1)``: la iteración extra es la que
        recibe ``terminado``.
        """
        return self._presupuesto

    def next(self, medidas: Medidas) -> float:
        """
        Devuelve el voltaje a aplicar en el paso `medidas.k`.

        Primero decide, con las medidas del paso anterior, si el segmento activo
        termina (condición cumplida o final natural). Si era el último, marca
        `terminado` y la fase debe cortar ANTES de la física.
        """
        if self.terminado:
            return float(self._V)  # type: ignore[arg-type]

        if medidas.k > 0:
            modo, op = self.segmentos[self._idx]
            condicion = self._condicion_cumplida(op, medidas)
            if condicion is None and self._fin_natural:
                condicion = "hasta" if (modo == "rampa" and self._llego_a_hasta) else "pasos"
                if op.get("exigir") and not self.previsualizacion:
                    raise CondicionExigidaNoCumplida(
                        f"{self.fase}, segmento {self._idx} ({modo}): terminó por '{condicion}' en k={medidas.k} "
                        f"(V={self._V:.5f} V) sin cumplirse "
                        f"{[c for c in CONDICIONES_VALIDAS if c in op]}, marcadas como exigidas."
                    )
            if condicion is not None:
                self._registrar_transicion(condicion, medidas.k)
                if self._idx + 1 >= len(self.segmentos):
                    self._terminar(condicion)
                    return float(self._V)  # type: ignore[arg-type]
                self._idx += 1
                self._activar_segmento(medidas.k)

        # Un segmento puede no tener ningún punto (rampa 'anterior' que ya está en
        # su 'hasta'): se encadena al siguiente sin producir voltaje.
        while self._sin_puntos:
            self._registrar_transicion("hasta", medidas.k)
            if self._idx + 1 >= len(self.segmentos):
                self._terminar("hasta")
                return float(self._V)  # type: ignore[arg-type]
            self._idx += 1
            self._activar_segmento(medidas.k)

        V = self._voltaje_del_paso()
        self._V = V
        self.puntos += 1
        if self.V_inicio is None:
            self.V_inicio = V
            self._informe_enganche()
        return V

    def estado(self) -> dict:
        """Estado JSON-serializable de la etapa para metadata y phase_state."""
        trans = [{"segmento": t.segmento, "k": t.k, "V": float(t.V), "condicion": t.condicion} for t in self.transiciones]
        n_trans = len(trans)
        if n_trans > MAX_TRANSICIONES_METADATA:
            mitad = MAX_TRANSICIONES_METADATA // 2
            trans = trans[:mitad] + trans[-mitad:]
        return {
            "fase": self.fase,
            "segmentos": [[modo, op] for modo, op in self.segmentos],
            "completada": self.completada,
            "fin_por": self.fin_por,
            "V_inicio": self.V_inicio,
            "V_fin": None if self._V is None or self.puntos == 0 else float(self._V),
            "puntos": self.puntos,
            "presupuesto": self._presupuesto,
            "segmento_activo": self._idx,
            "enganche": self._enganche(),
            "anterior_resuelto": self.anterior_resuelto,
            "n_transiciones": n_trans,
            "transiciones": trans,
        }

    # ------------------------------------------------------------- interno

    def _activar_segmento(self, k: int) -> None:
        """Prepara el segmento `self._idx`: resuelve 'anterior' y fija su recorrido."""
        modo, op = self.segmentos[self._idx]
        self._pasos_seg = 0
        self._fin_natural = False
        self._llego_a_hasta = False
        self._sin_puntos = False

        if modo == "constante":
            if op["V"] == ANTERIOR:
                self._V_meseta = float(self._V)  # type: ignore[arg-type]
                self.anterior_resuelto.append({"segmento": self._idx, "k": k, "V": self._V_meseta})
            else:
                self._V_meseta = float(op["V"])
            return

        hasta, dV = float(op["hasta"]), float(op["dV"])
        if op["desde"] == ANTERIOR:
            anterior = float(self._V)  # type: ignore[arg-type]
            self.anterior_resuelto.append({"segmento": self._idx, "k": k, "V": anterior})
            if abs(hasta - anterior) <= dV * 1e-6:
                self._sin_puntos = True
                self._llego_a_hasta = True
                return
            self._sentido = 1.0 if hasta > anterior else -1.0
            # Continúa: el primer punto es un dV más allá del anterior.
            self._V_base = anterior + self._sentido * dV
        else:
            self._V_base = float(op["desde"])
            self._sentido = 1.0 if hasta > self._V_base else -1.0

    def _voltaje_del_paso(self) -> float:
        modo, op = self.segmentos[self._idx]
        n = self._pasos_seg
        self._pasos_seg += 1

        if modo == "constante":
            V = self._V_meseta
            if self._pasos_seg >= int(op["pasos"]):
                self._fin_natural = True
            return V

        hasta, dV = float(op["hasta"]), float(op["dV"])
        # base + n·dV (no acumulación): misma aritmética que np.arange.
        V = self._V_base + n * dV * self._sentido
        tol = dV * 1e-6
        if self._sentido * (V - hasta) >= -tol:
            # Llega o se pasa: si dV no divide el tramo, se recorta a 'hasta' exacto.
            if self._sentido * (V - hasta) > tol:
                V = hasta
            self._fin_natural = True
            self._llego_a_hasta = True
        elif "pasos" in op and self._pasos_seg >= int(op["pasos"]):
            self._fin_natural = True
        return V

    def _condicion_cumplida(self, op: dict, m: Medidas) -> Optional[str]:
        """Nombre de la primera condición cumplida, o None."""
        if "hasta_I" in op and abs(m.I_total) >= float(op["hasta_I"]):
            return "hasta_I"
        if "hasta_vacantes" in op and m.n_vacantes >= op["hasta_vacantes"]:
            return "hasta_vacantes"
        if "hasta_filamentos" in op and m.n_filamentos >= op["hasta_filamentos"]:
            return "hasta_filamentos"
        if op.get("hasta_percolacion") and m.percola:
            return "hasta_percolacion"
        return None

    def _registrar_transicion(self, condicion: str, k: int) -> None:
        modo = self.segmentos[self._idx][0]
        self.transiciones.append(_Transicion(segmento=self._idx, k=k, V=float(self._V), condicion=condicion))  # type: ignore[arg-type]
        n = len(self.transiciones)
        if self.previsualizacion:
            return
        msg = f"{self.fase}: fin del segmento {self._idx} ({modo}) en k={k}, V={self._V:.5f} V, por {condicion}"
        if n <= MAX_TRANSICIONES_LOG:
            logger.info(msg)
        else:
            if n == MAX_TRANSICIONES_LOG + 1:
                logger.info(f"{self.fase}: más de {MAX_TRANSICIONES_LOG} transiciones; el resto se registra en DEBUG")
            logger.debug(msg)

    def _terminar(self, condicion: str) -> None:
        self.terminado = True
        self.completada = True
        self.fin_por = condicion
        if self.puntos == 0:
            raise ValueError(f"{self.fase}: la forma de onda terminó sin producir ningún punto")
        if self.previsualizacion:
            return
        logger.info(
            f"{self.fase} terminada: {self.puntos} puntos, V {self.V_inicio:.5f} → {self._V:.5f} V, "
            f"{len(self.transiciones)} transiciones, fin por {condicion}"
        )

    def _enganche(self) -> Optional[dict]:
        if self.v_previo is None or self.V_inicio is None:
            return None
        salto = self.V_inicio - self.v_previo
        primero = self.segmentos[0][1]
        continua = primero.get("desde", primero.get("V")) == ANTERIOR
        return {
            "V_fin_anterior": self.v_previo,
            "V_inicio": self.V_inicio,
            "salto": salto,
            "continua_anterior": continua,
            "punto_repetido": (not continua or self.segmentos[0][0] == "constante") and abs(salto) < 1e-12,
        }

    def _informe_enganche(self) -> None:
        if self.previsualizacion:
            return
        e = self._enganche()
        if e is None:
            logger.info(f"{self.fase}: empieza en {self.V_inicio:.5f} V (sin etapa anterior)")
            return
        texto = f"enganche → {self.fase}: la etapa anterior acabó en {e['V_fin_anterior']:.5f} V, {self.fase} empieza en {e['V_inicio']:.5f} V"
        if e["continua_anterior"] and self.segmentos[0][0] == "rampa":
            logger.info(f"{texto} (continúa desde la anterior)")
        elif e["punto_repetido"]:
            logger.info(f"{texto} (punto repetido: ese voltaje aparece dos veces en los datos)")
        elif abs(e["salto"]) > 1e-12:
            logger.warning(f"{texto}: SALTO de {e['salto']:+.5f} V")
        else:
            logger.info(texto)


# ======================================================================
# Protocolo: las cuatro etapas
# ======================================================================


@dataclass
class ProtocoloVoltaje:
    """
    Protocolo de voltaje completo del ciclo: una lista de segmentos por etapa.

    Todas las etapas son obligatorias. Ver el docstring del módulo para la
    sintaxis de los segmentos.
    """

    pp_set: List[Tuple[str, dict]]
    sp_set: List[Tuple[str, dict]]
    pp_reset: List[Tuple[str, dict]]
    sp_reset: List[Tuple[str, dict]]
    _controladores: Dict[str, VoltageController] = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self):
        for fase in FASES:
            setattr(self, fase, parsear_segmentos(getattr(self, fase), fase))
        primero = self.pp_set[0][1]
        if primero.get("desde", primero.get("V")) == ANTERIOR:
            raise ValueError(f"pp_set, segmento 0: no puede usar {ANTERIOR!r}, no hay etapa anterior")

    # ----------------------------------------------------- configuración

    @classmethod
    def desde_config(cls, raw: Any) -> "ProtocoloVoltaje":
        """Construye el protocolo desde un dict o desde su texto (celda del CSV)."""
        if isinstance(raw, ProtocoloVoltaje):
            return raw
        if isinstance(raw, str):
            try:
                raw = ast.literal_eval(raw.strip())
            except (ValueError, SyntaxError) as e:
                raise ValueError(f"protocolo_voltaje: el texto no es un diccionario válido: {e}") from e
        if not isinstance(raw, dict):
            raise ValueError(f"protocolo_voltaje: se esperaba un dict con las etapas {FASES}, llegó {type(raw).__name__}")
        faltan = [f for f in FASES if f not in raw]
        sobran = [f for f in raw if f not in FASES]
        if faltan or sobran:
            raise ValueError(f"protocolo_voltaje: etapas que faltan {faltan}, claves desconocidas {sobran}. Obligatorias: {FASES}")
        return cls(**{f: raw[f] for f in FASES})

    def configuracion(self) -> Dict[str, list]:
        """Las cuatro etapas tal como se escribieron (tuplas → listas, apto para JSON)."""
        return {f: [[modo, dict(op)] for modo, op in getattr(self, f)] for f in FASES}

    def a_texto(self) -> str:
        """Texto para la celda del CSV (se lee con `desde_config`)."""
        return repr({f: [(modo, dict(op)) for modo, op in getattr(self, f)] for f in FASES})

    # ------------------------------------------------------- ejecución

    def controlador(self, fase: str, previo: Optional[dict] = None) -> VoltageController:
        """
        Controlador de la etapa `fase`.

        `previo` es el dict de estado que devolvió la etapa anterior (el que
        contiene la clave ``"voltaje"``), o None en PP_set. De él solo se lee el
        último voltaje aplicado (``previo["voltaje"]["V_fin"]``).
        """
        if fase not in FASES:
            raise ValueError(f"fase desconocida {fase!r}. Válidas: {FASES}")
        # Un estado preparado a mano (p. ej. Preparar_pp_reset.ipynb) puede no traer
        # su voltaje: entonces no hay enganche y un 'anterior' en el primer
        # segmento da error al construir el controlador.
        v_previo = ((previo or {}).get("voltaje") or {}).get("V_fin")
        controller = VoltageController(fase, getattr(self, fase), v_previo=v_previo)
        self._controladores[fase] = controller
        logger.info(f"{fase}: forma de onda\n{_texto_etapa(fase, controller.segmentos, controller.presupuesto())}")
        return controller

    def informe(self) -> dict:
        """Bloque ``protocolo_voltaje`` de la metadata: lo pedido y lo que pasó."""
        return {
            "configuracion": self.configuracion(),
            "ejecucion": {f: c.estado() for f, c in self._controladores.items()},
        }

    # ------------------------------------------------ previsualización

    def recorrido(self) -> Dict[str, dict]:
        """
        Recorrido de voltaje de cada etapa SIN física: se supone que ninguna
        condición de salida anticipada se cumple (el camino más largo).

        Devuelve por etapa: ``V`` (lista de voltajes), ``cortes`` (índices de
        los puntos donde empieza un segmento que puede cortarse antes) y
        ``enganche`` (dict o None).
        """
        salida: Dict[str, dict] = {}
        previo: Optional[dict] = None
        for fase in FASES:
            v_previo = None if previo is None else previo["V_fin"]
            c = VoltageController(fase, getattr(self, fase), v_previo=v_previo, previsualizacion=True)
            nulas = Medidas(I_total=0.0, n_vacantes=0, n_filamentos=0, percola=False, k=0)
            V: List[float] = []
            cortes: List[int] = []
            idx_prev = -1
            for k in range(c.presupuesto() + 1):
                nulas.k = k
                v = c.next(nulas)
                if c.terminado:
                    break
                if c._idx != idx_prev:
                    idx_prev = c._idx
                    if any(x in c.segmentos[c._idx][1] for x in CONDICIONES_VALIDAS):
                        cortes.append(len(V))
                V.append(v)
            previo = c.estado()
            salida[fase] = {"V": V, "cortes": cortes, "enganche": previo["enganche"], "segmentos": c.segmentos}
        return salida

    def resumen(self) -> str:
        """Texto de las cuatro etapas y sus enganches (sin física)."""
        rec = self.recorrido()
        lineas = []
        for fase in FASES:
            r = rec[fase]
            lineas.append(_texto_etapa(fase, r["segmentos"], len(r["V"])))
            if r["V"]:
                lineas.append(f"    recorrido (camino más largo): {r['V'][0]:+.5f} V → {r['V'][-1]:+.5f} V")
            e = r["enganche"]
            if e is not None:
                if e["continua_anterior"] and r["segmentos"][0][0] == "rampa":
                    tipo = "continúa desde la anterior"
                elif e["punto_repetido"]:
                    tipo = "punto repetido"
                elif abs(e["salto"]) > 1e-12:
                    tipo = f"⚠ SALTO de {e['salto']:+.5f} V"
                else:
                    tipo = "engancha"
                lineas.append(f"    enganche: anterior acaba en {e['V_fin_anterior']:+.5f} V, empieza en {e['V_inicio']:+.5f} V → {tipo}")
            if r["cortes"]:
                lineas.append("    (hay segmentos con condiciones: pueden cortar antes y cambiar los voltajes siguientes)")
        return "\n".join(lineas)


def _texto_etapa(fase: str, segmentos: List[Tuple[str, dict]], puntos: int) -> str:
    """Una línea por segmento, para logs y previsualización."""
    lineas = [f"  {fase} ({len(segmentos)} segmentos, hasta {puntos} puntos)"]
    for i, (modo, op) in enumerate(segmentos):
        extra = {c: op[c] for c in CONDICIONES_VALIDAS + ("exigir",) if c in op}
        extra_txt = f"  {extra}" if extra else ""
        if modo == "rampa":
            pasos = f", máx. {op['pasos']} pasos" if "pasos" in op else ""
            lineas.append(f"    [{i}] rampa     {op['desde']} → {op['hasta']} V, dV = {op['dV']:g} V{pasos}{extra_txt}")
        else:
            lineas.append(f"    [{i}] constante {op['V']} V durante {op['pasos']} pasos{extra_txt}")
        if len(segmentos) > 12 and i == 5:
            lineas.append(f"    ... ({len(segmentos) - 8} segmentos más) ...")
            break
    if len(segmentos) > 12:
        for j in range(len(segmentos) - 2, len(segmentos)):
            modo, op = segmentos[j]
            desc = f"{op['desde']} → {op['hasta']} V, dV = {op['dV']:g} V" if modo == "rampa" else f"{op['V']} V durante {op['pasos']} pasos"
            lineas.append(f"    [{j}] {modo:<9} {desc}")
    return "\n".join(lineas)


# ======================================================================
# Previsualización desde el notebook
# ======================================================================


def _protocolos_de(fuente: Any) -> List[Tuple[str, ProtocoloVoltaje]]:
    """(etiqueta, protocolo) desde un ConfigManager, un protocolo o una lista."""
    if isinstance(fuente, ProtocoloVoltaje):
        return [("protocolo", fuente)]
    if hasattr(fuente, "simulations"):
        out = []
        for i, sim in enumerate(fuente.simulations):
            if "protocolo_voltaje" not in sim.params:
                raise ValueError(f"sim {i}: falta 'protocolo_voltaje'")
            out.append((f"sim {i}", ProtocoloVoltaje.desde_config(sim.params["protocolo_voltaje"])))
        return out
    return [(f"protocolo {i}", ProtocoloVoltaje.desde_config(p)) for i, p in enumerate(fuente)]


def previsualizar(fuente: Any, sims: Optional[List[int]] = None, max_figuras: int = 12, figura: bool = True):
    """
    Muestra, SIN simular, el voltaje que va a aplicar cada simulación.

    Args:
        fuente: Un `ConfigManager`, un `ProtocoloVoltaje`, o una lista de protocolos
            (dicts o textos).
        sims: Índices a mostrar (None = todas).
        max_figuras: Con más simulaciones que esto solo se imprime el texto,
            salvo que se pidan índices concretos con `sims`.
        figura: False = solo texto.

    Returns:
        La figura de matplotlib, o None si no se dibuja.
    """
    protocolos = _protocolos_de(fuente)
    if sims is not None:
        protocolos = [protocolos[i] for i in sims]

    for etiqueta, p in protocolos:
        print(f"=== {etiqueta} ===")
        print(p.resumen())

    if not figura or not protocolos or (sims is None and len(protocolos) > max_figuras):
        if figura and protocolos and sims is None and len(protocolos) > max_figuras:
            print(f"\n{len(protocolos)} simulaciones: solo texto. Usa sims=[...] para dibujar algunas.")
        return None

    import matplotlib.pyplot as plt  # solo aquí: el controlador no depende de matplotlib

    # Figura de diagnóstico: sin LaTeX (el paquete puede haberlo activado globalmente).
    with plt.rc_context({"text.usetex": False}):
        n = len(protocolos)
        cols = 1 if n == 1 else 2
        filas = math.ceil(n / cols)
        fig, axes = plt.subplots(filas, cols, figsize=(7 * cols, 3.2 * filas), squeeze=False)
        colores = {"pp_set": "tab:blue", "sp_set": "tab:cyan", "pp_reset": "tab:red", "sp_reset": "tab:orange"}

        for ax, (etiqueta, p) in zip(axes.flat, protocolos):
            rec = p.recorrido()
            x0 = 0
            for fase in FASES:
                r = rec[fase]
                xs = list(range(x0, x0 + len(r["V"])))
                ax.plot(xs, r["V"], color=colores[fase], label=fase, linewidth=1.5)
                for c in r["cortes"]:
                    ax.plot(xs[c], r["V"][c], marker="v", color="black", markersize=6)
                e = r["enganche"]
                if e is not None and abs(e["salto"]) > 1e-12 and not (e["continua_anterior"] and r["segmentos"][0][0] == "rampa"):
                    ax.axvline(x0, color="red", linestyle="--", linewidth=1)
                x0 += len(r["V"])
            ax.set_title(etiqueta)
            ax.set_xlabel("paso")
            ax.set_ylabel("V (V)")
            ax.axhline(0, color="gray", linewidth=0.5)
        for ax in list(axes.flat)[n:]:
            ax.set_visible(False)
        axes.flat[0].legend(loc="best", fontsize=8)
        fig.suptitle("Protocolo de voltaje (camino más largo; triángulo = puede cortar antes; línea roja = salto)", fontsize=10)
        fig.tight_layout()
    return fig
