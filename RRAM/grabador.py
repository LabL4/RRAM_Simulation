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
        guardar_datos(
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

    Los nombres de fichero y las claves son los de siempre (`guardar_estado_intermedio`); cada
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
        guardar_estado_intermedio(
            ruta_destino=self.ruta_destino, etapa=self.etapa, num_simulation=self.num_simulation, k=k, **compactar(matrices)
        )
        self.pasos_guardados.append(int(k))


# ---------------------------------------------------------------------------
# Esquema de columnas de `datos_sim` (los `Data_<fase>_<N>.npz`)
# ---------------------------------------------------------------------------
# Única definición del orden de columnas: la usa el grabador para escribir (y para la cabecera
# `header_datos_sim` de cada fichero) y cualquier lector puede importarla. Si cambia el orden o el
# significado de una columna, subir `VERSION_ESQUEMA`.


VERSION_ESQUEMA = 1


def columnas_datos(N: int, con_t_ef: bool = False) -> List[str]:
    """
    Columnas de `datos_sim`: [t, V, I_total, R_total] + I_1..I_N + R_1..R_N + T_1..T_N (+ t_ef en el RESET).
    El orden por bloques es el de `PasoSimulacion.fila`.
    """
    cols = ["t[s]", "V[V]", "I_total[A]", "R_total[Ohm]"]
    cols += [f"I_{i}[A]" for i in range(1, N + 1)]
    cols += [f"R_{i}[Ohm]" for i in range(1, N + 1)]
    cols += [f"T_{i}[K]" for i in range(1, N + 1)]
    if con_t_ef:
        cols += ["t_ef[s]"]
    return cols


def indices_columnas(N: int, con_t_ef: bool = False) -> Dict[str, int]:
    """Posición de cada columna por nombre, p. ej. `indices_columnas(2)["R_total[Ohm]"] == 3`."""
    return {nombre: i for i, nombre in enumerate(columnas_datos(N, con_t_ef))}


# ---------------------------------------------------------------------------
# Rutas de salida y escritura de ficheros
# ---------------------------------------------------------------------------


def crear_rutas_simulacion(num_simulation: int, state: str, results_dir: str | Path = "Results") -> dict:
    """
    Crea las rutas necesarias para guardar resultados de la simulación.

    Args:
        num_simulation (int): Número índice de la simulación.
        state (str): Estado de la simulación, puede ser 'set' o 'reset'.
        results_dir (str | Path): Carpeta raíz de resultados. Por defecto "Results"
            (comportamiento histórico, relativo a Path.cwd()). Pasar una ruta
            distinta (p.ej. para pruebas de diagnóstico) aísla completamente
            los datos escritos por esta simulación sin tocar Results/ real.

    Returns:
        dict: Diccionario con las rutas Path para 'simulation', 'figures' y 'set'.

    Side-effects:
        Crea las carpetas en disco si no existen.
    """
    results_dir = Path(results_dir)
    base = results_dir if results_dir.is_absolute() else Path.cwd() / results_dir
    simulation_path = base / f"simulation_{num_simulation}"
    figures_path = simulation_path / "Figures"
    data_simulation_path = simulation_path / state

    return {
        "simulation_path": simulation_path,
        "figures_path": figures_path,
        "data_simulation_path": data_simulation_path,
    }


def guardar_datos(save_path_data: Path, headers: Optional[dict] = None, **vectores) -> None:
    """
    Guarda múltiples vectores, matrices o datos escalares en un archivo .npz comprimido.
    Asocia dinámicamente encabezados a los datos si se proporcionan.

    Args:
        ruta_destino (Path): Ruta del archivo donde se guardarán los datos.
        headers (dict, opcional): Diccionario con los encabezados. Las claves deben
                                  coincidir con los nombres dados en **vectores.
        **vectores: Datos a guardar pasados como argumentos con nombre (ej: datos_iv=matriz).
    """
    if headers is None:
        headers = {}

    datos_seguros = {}

    for nombre, vector in vectores.items():
        if vector is not None:
            datos_seguros[nombre] = vector

            # Si el usuario ha proporcionado un encabezado para este vector específico,
            # lo guardamos en el .npz añadiendo el prefijo 'header_' al nombre original.
            if nombre in headers:
                # Lo convertimos a array de NumPy de 1 elemento para guardarlo de forma nativa
                datos_seguros[f"header_{nombre}"] = np.array([headers[nombre]])
        else:
            # Si se pasa un None, guardamos un array vacío para mantener la estructura de claves
            datos_seguros[nombre] = np.array([])

    # Guardamos todo el paquete de forma consolidada y comprimida
    np.savez_compressed(save_path_data.with_suffix(".npz"), **datos_seguros)


def guardar_estado_intermedio(ruta_destino: Path, etapa: str, num_simulation: int, k: int, **matrices) -> None:
    """
    Guarda todas las matrices relevantes de un paso de simulación en un único archivo .npz comprimido.
    Garantiza que todas las claves pasadas existan en el archivo final.
    """

    # 1. Reemplazamos los valores None por un array vacío de NumPy.
    # Esto es mejor que guardar 'None' a secas, porque NumPy prefiere trabajar con arrays.
    matrices_seguras = {}
    for nombre, matriz in matrices.items():
        if matriz is None:
            matrices_seguras[nombre] = np.array([])  # Array vacío indicando que no hay datos
        else:
            matrices_seguras[nombre] = matriz

    # 2. Construimos el nombre del archivo estandarizado
    nombre_archivo = ruta_destino / f"Estado_{etapa}_sim_{num_simulation}_paso_{k}.npz"

    # 3. Guardamos de forma comprimida asegurando que todas las claves se escriben
    np.savez_compressed(nombre_archivo, **matrices_seguras)
