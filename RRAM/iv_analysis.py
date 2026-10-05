"""Postprocesado y representación de curvas I-V de la simulación RRAM."""

from pathlib import Path

import numpy as np

from . import utils
import logging

from . import Representate

logger = logging.getLogger(__name__)

#: Por debajo de este valor absoluto de intensidad, el punto se descarta antes
#: de representar/marcar: es ruido de fondo (offset del solver, filamento aún
#: sin percolar) y distorsiona la escala logarítmica del eje Y.
INTENSIDAD_MINIMA_DEFAULT = 1e-7

#: Puntos marcados de la curva I-V (figura I-V_marcado_{N}). Es la lista por
#: defecto: se usa si no se pasa otra (argumento `puntos_iv` o `--puntos-iv
#: archivo.json` en la CLI) y siempre se escribe en el log la que se aplicó.
#:
#: Cada punto:
#:   - "etapa": pp_set | sp_set | pp_reset | sp_reset  (curva donde se busca)
#:   - "en": cómo se localiza, uno de:
#:       ("voltaje", V)        el punto de la etapa más cercano a ese voltaje [V]
#:       ("evento", nombre)    la fila exacta (paso k) en que ocurrió el evento:
#:                             "percolacion", "creacion_<i>" o "rotura_<i>"
#:                             (i empieza en 0). El evento debe haber ocurrido
#:                             en la misma etapa del punto; si no, se omite.
#:       ("fin_etapa",)        el último punto de la etapa
#:   - "desplazamiento": (dx, factor_y) de la etiqueta respecto al punto
#:
#: "c" usa fin_etapa (y no un voltaje fijo) para seguir al protocolo de voltaje.
PUNTOS_IV = {
    "a": {"etapa": "pp_set", "en": ("voltaje", 1e-7), "desplazamiento": (0.025, 1.0)},
    "b": {"etapa": "pp_set", "en": ("evento", "percolacion"), "desplazamiento": (0.005, 0.27)},
    "c": {"etapa": "pp_set", "en": ("fin_etapa",), "desplazamiento": (0.02, 0.35)},
    "d": {"etapa": "pp_reset", "en": ("voltaje", -0.44), "desplazamiento": (0.02, 1.0)},
    "e": {"etapa": "pp_reset", "en": ("evento", "rotura_0"), "desplazamiento": (-0.11, 0.66)},
    "f": {"etapa": "pp_reset", "en": ("voltaje", -1.1), "desplazamiento": (0.025, 0.25)},
    "g": {"etapa": "sp_reset", "en": ("voltaje", -2e-7), "desplazamiento": (-0.12, 1.0)},
}

_ETAPAS_IV = ("pp_set", "sp_set", "pp_reset", "sp_reset")


def validar_puntos_iv(puntos_iv: dict) -> dict:
    """
    Comprueba la tabla de puntos marcados y la normaliza (listas de JSON → tuplas).

    Raises:
        ValueError: con el punto y el campo que fallan.
    """
    if not isinstance(puntos_iv, dict) or not puntos_iv:
        raise ValueError(f"puntos_iv: se esperaba un dict no vacío {{etiqueta: punto}}, llegó {puntos_iv!r}")
    normalizados = {}
    for etiqueta, punto in puntos_iv.items():
        donde = f"puntos_iv[{etiqueta!r}]"
        if not isinstance(punto, dict) or set(punto) != {"etapa", "en", "desplazamiento"}:
            raise ValueError(
                f"{donde}: debe tener exactamente las claves 'etapa', 'en' y 'desplazamiento', llegó {punto!r}"
            )
        if punto["etapa"] not in _ETAPAS_IV:
            raise ValueError(f"{donde}: etapa {punto['etapa']!r} desconocida. Válidas: {_ETAPAS_IV}")
        en = tuple(punto["en"]) if isinstance(punto["en"], (list, tuple)) else ()
        if not en or en[0] not in ("voltaje", "evento", "fin_etapa"):
            raise ValueError(
                f"{donde}: 'en' debe ser ('voltaje', V), ('evento', nombre) o ('fin_etapa',), llegó {punto['en']!r}"
            )
        if en[0] == "voltaje" and (len(en) != 2 or not isinstance(en[1], (int, float)) or isinstance(en[1], bool)):
            raise ValueError(f"{donde}: ('voltaje', V) necesita un número, llegó {punto['en']!r}")
        if en[0] == "evento" and (len(en) != 2 or not _es_evento_valido(en[1])):
            raise ValueError(
                f"{donde}: evento desconocido {punto['en']!r}. Válidos: 'percolacion', 'creacion_<i>', 'rotura_<i>'"
            )
        if en[0] == "fin_etapa" and len(en) != 1:
            raise ValueError(f"{donde}: ('fin_etapa',) no lleva valor, llegó {punto['en']!r}")
        desp = punto["desplazamiento"]
        if not isinstance(desp, (list, tuple)) or len(desp) != 2 or not all(isinstance(x, (int, float)) for x in desp):
            raise ValueError(f"{donde}: 'desplazamiento' debe ser (dx, factor_y), llegó {desp!r}")
        normalizados[etiqueta] = {"etapa": punto["etapa"], "en": en, "desplazamiento": tuple(desp)}
    return normalizados


def _es_evento_valido(nombre) -> bool:
    if nombre == "percolacion":
        return True
    for prefijo in ("creacion_", "rotura_"):
        if isinstance(nombre, str) and nombre.startswith(prefijo) and nombre[len(prefijo) :].isdigit():
            return True
    return False


def _evento(nombre: str, paso_percolacion, creaciones_dict: dict, roturas_dict: dict):
    """(etapa, k) de un evento, o None si no ocurrió en esta simulación."""
    if nombre == "percolacion":
        return None if paso_percolacion is None else ("pp_set", int(paso_percolacion))
    registro = creaciones_dict if nombre.startswith("creacion_") else roturas_dict
    evento = registro.get(int(nombre.split("_")[1]))
    return None if evento is None else (evento["etapa"], int(evento["k"]))


def localizar_puntos_iv(
    puntos_iv: dict,
    data: dict,
    data_sin_filtrar: dict,
    paso_percolacion: int | None,
    creaciones_dict: dict,
    roturas_dict: dict,
    intensidad_minima: float,
) -> dict:
    """
    Coordenadas (V, |I|) de cada punto de la tabla `puntos_iv` ya validada.

    Args:
        data: Datos de cada etapa ya filtrados de ruido (curva que se dibuja).
        data_sin_filtrar: Los mismos datos sin filtrar; los eventos se buscan
            aquí por su fila k.

    Returns:
        {etiqueta: (V, |I|)} con los puntos que se han podido localizar.
    """
    # Cada punto se busca solo en la curva de SU etapa. Si esa etapa no tiene
    # datos (simulación a medias) o el evento no ocurrió, el punto se omite y el
    # resto se sigue marcando.
    puntos_totales: dict = {}
    for etiqueta, punto in puntos_iv.items():
        curva = data[punto["etapa"]]
        if curva.shape[0] == 0:
            logger.info(f"Punto {etiqueta}: sin datos de {punto['etapa']}; se omite.")
            continue
        tipo = punto["en"][0]
        if tipo == "fin_etapa":
            puntos_totales[etiqueta] = (float(curva[-1, 1]), float(abs(curva[-1, 2])))
        elif tipo == "voltaje":
            puntos_totales.update(
                utils.obtener_puntos_en_curva(curva[:, 1], abs(curva[:, 2]), {etiqueta: punto["en"][1]})
            )
        else:
            nombre = punto["en"][1]
            evento = _evento(nombre, paso_percolacion, creaciones_dict, roturas_dict)
            if evento is None:
                logger.info(f"Punto {etiqueta}: el evento {nombre!r} no ocurrió; se omite.")
                continue
            etapa_evento, k = evento
            if etapa_evento != punto["etapa"]:
                logger.warning(
                    f"Punto {etiqueta}: el evento {nombre!r} ocurrió en {etapa_evento}, no en {punto['etapa']}; se omite."
                )
                continue
            if not 0 <= k < data_sin_filtrar[etapa_evento].shape[0]:
                logger.warning(
                    f"Punto {etiqueta}: la fila {k} del evento {nombre!r} no existe en {etapa_evento}; se omite."
                )
                continue
            fila = data_sin_filtrar[etapa_evento][k]
            if abs(fila[2]) < intensidad_minima:
                logger.info(
                    f"Punto {etiqueta}: la fila {k} del evento {nombre!r} es ruido (|I| < {intensidad_minima:.1e} A); se omite."
                )
                continue
            puntos_totales[etiqueta] = (float(fila[1]), float(abs(fila[2])))

    return puntos_totales


def simulation_IV(
    num_simulation: int,
    figures_path: Path,
    simulation_path: Path,
    puntos_iv: dict,
    paso_percolacion: int | None,
    creaciones_dict: dict,
    roturas_dict: dict,
    marcado: bool = False,
    intensidad_minima: float = INTENSIDAD_MINIMA_DEFAULT,
    mostrar_experimental: bool = True,
):
    """
    Genera UNA figura: la curva I-V sin marcar (``marcado=False``, default) o
    la curva con los puntos a-g marcados (``marcado=True``). Antes esta
    función generaba ambas figuras en la misma llamada; ahora cada subcomando
    de la CLI (`plot` / `plot_marcado`) pide explícitamente la que necesita.

    Args:
        puntos_iv: Tabla de puntos marcados (ver `PUNTOS_IV`). Solo se usa con
            ``marcado=True``.
        marcado: Si True, dibuja `I-V_marcado_{N}` (curva + puntos a-g). Si
            False, dibuja solo `I-V_{N}` (curva sin marcar).
        intensidad_minima: Umbral absoluto de intensidad (A). Cualquier punto
            con |I| por debajo de este valor se descarta de TODAS las fases
            antes de unir curvas y buscar los puntos marcados, para no
            representar ruido de fondo cerca de I=0 en la escala log.
    """
    # region Representar datos
    # Los nombres de fichero (I-V_{N}, I-V_marcado_{N}) los construye cada
    # función de plot a partir de la CARPETA figures_path que le pasamos.
    # Definir nombres base y tipos
    prefixes = ["pp", "sp"]
    stages = ["set", "reset"]

    # Diccionario para guardar los datos cargados en memoria
    data = {}

    # Cargar archivos de forma automatizada
    for prefix in prefixes:
        for stage in stages:
            # ACTUALIZACIÓN 1: Nombre de archivo ajustado a tu nuevo formato
            name = f"Data_{prefix}_{stage}_{num_simulation}.npz"
            key = f"{prefix}_{stage}"

            try:
                # Cargamos el archivo .npz
                archivo_npz = np.load(simulation_path / name)

                # ACTUALIZACIÓN 2: Extraemos solo la matriz "datos_sim" a la memoria
                # Si en el futuro guardas vectores sueltos (ej: voltaje=v), aquí usarías archivo_npz["voltaje"]
                data[key] = archivo_npz["datos_sim"]

                # Cerramos el archivo npz (buena práctica de manejo de I/O)
                archivo_npz.close()
            except FileNotFoundError:
                logger.info(f"Advertencia: No se encontró el archivo {name}")
                # Podrías inicializar un array vacío o manejar el error según convenga
                data[key] = np.zeros((0, 3))

    # Copia sin filtrar: los puntos por evento se localizan por su fila k, que
    # indexa los datos originales (el filtro de ruido quita filas y desplazaría
    # los índices).
    data_sin_filtrar = {key: arr for key, arr in data.items()}

    # Filtrado de ruido: descarta filas con |I| < intensidad_minima ANTES de
    # unir curvas y de buscar los puntos marcados, para que ni la curva ni los
    # marcadores puedan caer en esa zona de ruido.
    for key, arr in data.items():
        if arr.shape[0] == 0:
            continue
        mask = np.abs(arr[:, 2]) >= intensidad_minima
        n_descartados = arr.shape[0] - int(mask.sum())
        if n_descartados > 0:
            logger.info(
                f"{key}: {n_descartados}/{arr.shape[0]} puntos descartados por |I| < {intensidad_minima:.1e} A."
            )
        data[key] = arr[mask]

    # Unir las partes PP y SP para el SET
    # Nota: Ya que hemos extraído 'datos_sim', podemos acceder a las columnas directamente
    # Columna 1 = Voltaje, Columna 2 = Intensidad
    i_set = np.concatenate([abs(data["pp_set"][:, 2]), abs(data["sp_set"][:, 2])])
    v_set = np.concatenate([data["pp_set"][:, 1], data["sp_set"][:, 1]])

    # Unir las partes PP y SP para el RESET
    i_reset = np.concatenate([abs(data["pp_reset"][:, 2]), abs(data["sp_reset"][:, 2])])
    v_reset = np.concatenate([data["pp_reset"][:, 1], data["sp_reset"][:, 1]])

    if not marcado:
        # `plot_IV` acepta arrays vacíos para las fases que falten y
        # simplemente no dibuja esa rama.
        Representate.plot_IV(
            v_set,
            i_set,
            v_reset,
            i_reset,
            num_simulation - 1,
            titulo_figura="",
            figures_path=str(figures_path),
            mostrar_experimental=mostrar_experimental,
        )
        return None

    # ----- marcado=True: solo la curva con puntos marcados -----
    puntos_iv = validar_puntos_iv(puntos_iv)
    logger.info("Tabla de puntos marcados aplicada:")
    for etiqueta, punto in puntos_iv.items():
        logger.info(f"  {etiqueta}: etapa={punto['etapa']} en={punto['en']} desplazamiento={punto['desplazamiento']}")

    puntos_totales = localizar_puntos_iv(
        puntos_iv, data, data_sin_filtrar, paso_percolacion, creaciones_dict, roturas_dict, intensidad_minima
    )

    logger.info("Puntos en la curva I-V:")
    for label, (v, i) in puntos_totales.items():
        logger.info(f"  Punto {label}: V = {v:.6f} V, I = {i:.6e} A")

    if not puntos_totales:
        logger.warning(
            f"Sim {num_simulation}: no hay ningún punto marcado calculable (sin datos en las "
            f"etapas de la tabla o sin los eventos pedidos). Se omite I-V_marcado."
        )
        return None

    Representate.plot_IV_marcado(
        v_set,
        i_set,
        v_reset,
        i_reset,
        num_simulation - 1,
        puntos_totales,
        {etiqueta: punto["desplazamiento"] for etiqueta, punto in puntos_iv.items()},
        figures_path=str(figures_path),
        mostrar_experimental=mostrar_experimental,
    )

    return None
