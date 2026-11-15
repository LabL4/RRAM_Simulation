"""
Guardado de una etapa: filas de datos (Data_<etapa>_N.npz) y estados intermedios
(Estado_<etapa>_sim_N_paso_k.npz).

Antes cada fase construía su cabecera, reservaba y recortaba sus arrays, y repetía el guardado de
estados dos veces (cada `pasos_guardar_estado` pasos y al final si el último paso no caía en la
cadencia). Aquí:

- `RegistroDatos` reserva las filas, las escribe, las recorta y las guarda, también si la fase se
  para a mitad (`volcar_si_falla`), de modo que nunca se pierden las filas ya calculadas.
- `GrabadorEstados` decide CUÁNDO se guarda un estado; la fase solo dice QUÉ se guarda, con una
  función que se evalúa únicamente cuando toca guardar.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from . import utils
from .esquema_datos import columnas_datos

logger = logging.getLogger(__name__)

#: Tipo con el que se guarda cada matriz de un estado intermedio (las no listadas, tal cual).
#: - Estados 0/1 en int8: con signo, para que las restas entre estados den -1/0/1.
#: - Temperaturas, resistencias y muro en float32 (error relativo ~6e-8: ~3e-5 K a 500 K).
#: - Las probabilidades se quedan en float64: bajan de 1e-38 y en float32 serían 0.
#: - El reloj de los iones se queda en int32: en una meseta larga pasaría de 65535 pasos.
TIPOS_ESTADO = {
    "actual_state": np.int8,
    "cf_clean_matrix": np.int8,
    "oxygen_state": np.int8,
    "temperatura": np.float32,
    "temperatura_sin_amortiguar": np.float32,
    "mapa_resistencias": np.float32,
    "matriz_para_plot_muro": np.float32,
}


def compactar(matrices: dict) -> dict:
    """Convierte cada matriz a su tipo de `TIPOS_ESTADO`. None y los arrays vacíos se dejan igual."""
    out = {}
    for nombre, valor in matrices.items():
        tipo = TIPOS_ESTADO.get(nombre)
        if tipo is None or valor is None:
            out[nombre] = valor
        else:
            arr = np.asarray(valor)
            out[nombre] = arr if arr.size == 0 else arr.astype(tipo)
    return out


class RegistroDatos:
    """
    Filas de datos de una etapa.

    `presupuesto` es una COTA SUPERIOR de pasos (la da el controlador de voltaje): se reservan
    presupuesto + 1 filas y al guardar se recorta a las realmente escritas. No se puede recortar
    por NaN, porque las filas sin percolación llevan NaN legítimo en R_total, I_fils y R_fils.
    """

    def __init__(
        self,
        ruta: Path,
        presupuesto: int,
        N: int,
        filas_electrodo: int,
        con_t_ef: bool = False,
        series_extra: Optional[Dict[str, str]] = None,
    ):
        self.ruta = Path(ruta)
        self.columnas = columnas_datos(N, con_t_ef)
        self._datos = np.zeros((presupuesto + 1, len(self.columnas)), dtype=np.float64)
        # Perfil de temperatura del electrodo usado en cada paso (escalar repetido sin FVM)
        self._perfil_electrodo = np.zeros((presupuesto + 1, filas_electrodo), dtype=np.float64)
        #: Series de 3 columnas (paso, V, valor) con su cabecera, p. ej. vacantes y resistencia en PP_set.
        self.cabeceras_extra = dict(series_extra or {})
        self._extra = {nombre: np.zeros((presupuesto + 1, 3), dtype=np.float64) for nombre in self.cabeceras_extra}
        #: Filas realmente escritas.
        self.n_filas = 0

    def escribir(self, k: int, fila, perfil_electrodo, **extra) -> None:
        self._datos[k] = fila
        self._perfil_electrodo[k] = perfil_electrodo
        for nombre, valor in extra.items():
            self._extra[nombre][k] = valor
        self.n_filas = k + 1

    @property
    def filas(self) -> np.ndarray:
        """`datos_sim` recortado a las filas escritas."""
        return self._datos[: self.n_filas]

    def guardar(self) -> None:
        n = self.n_filas
        utils.guardar_datos(
            save_path_data=self.ruta,
            headers={"datos_sim": ",".join(self.columnas), **self.cabeceras_extra},
            datos_sim=self._datos[:n],
            perfil_electrodo=self._perfil_electrodo[:n],
            **{nombre: serie[:n] for nombre, serie in self._extra.items()},
        )

    @contextmanager
    def volcar_si_falla(self):
        """Si el bloque lanza una excepción, guarda las filas ya escritas y la relanza."""
        try:
            yield
        except Exception as error:
            try:
                self.guardar()
                logger.warning(
                    f"La etapa se ha parado ({type(error).__name__}); guardadas sus {self.n_filas} filas de datos en "
                    f"{self.ruta.with_suffix('.npz').name}."
                )
            except Exception as e2:  # noqa: BLE001 - el volcado no debe tapar el error original
                logger.warning(f"No se pudieron guardar las filas de datos de la etapa: {type(e2).__name__}: {e2}")
            raise


class GrabadorEstados:
    """
    Estados intermedios de una etapa: cada `cada` pasos, en los pasos con un evento (percolación,
    creación o rotura de un filamento: `marcar_evento`) y, al cerrar, el último paso si no se guardó.
    `cada <= 0` desactiva todo el guardado (opción --sin-estados).
    `pasos_guardados` es el índice de lo guardado; acaba en la metadata.

    Los nombres de fichero y las claves son los de siempre (`utils.guardar_estado_intermedio`); cada
    matriz se guarda con el tipo compacto de `TIPOS_ESTADO`.
    """

    def __init__(self, ruta_destino: Path, etapa: str, num_simulation: int, cada: int):
        self.ruta_destino = Path(ruta_destino)
        self.etapa = etapa
        self.num_simulation = num_simulation
        self.cada = int(cada)
        self.pasos_guardados: list = []
        self._evento = False

    @property
    def guardados(self) -> int:
        return len(self.pasos_guardados)

    def toca(self, k: int) -> bool:
        return self.cada > 0 and k % self.cada == 0

    def marcar_evento(self) -> None:
        """El paso en curso tiene un evento: se guardará al final del paso aunque no toque por cadencia."""
        self._evento = True

    def paso(self, k: int, contenido: Callable[[], dict]) -> None:
        """Guarda el estado del paso `k` si toca o si hubo un evento. `contenido` solo se evalúa si se guarda."""
        if self.toca(k) or (self.cada > 0 and self._evento):
            self._guardar(k, contenido())
        self._evento = False

    def cerrar(self, k: int, contenido: Callable[[], dict]) -> None:
        """Estado final: se guarda si el último `k` no cayó en la cadencia (si cayó, ya está guardado)."""
        if self.cada > 0 and not self.toca(k) and k not in self.pasos_guardados:
            self._guardar(k, contenido())

    def _guardar(self, k: int, matrices: dict) -> None:
        utils.guardar_estado_intermedio(
            ruta_destino=self.ruta_destino, etapa=self.etapa, num_simulation=self.num_simulation, k=k, **compactar(matrices)
        )
        self.pasos_guardados.append(int(k))
