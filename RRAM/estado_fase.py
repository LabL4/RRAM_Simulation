"""
Estado con el que termina una fase y arranca la siguiente.

Antes era un diccionario de unas 20 claves, con nombres distintos para lo mismo según la fase
(`CF_centros`/`centros_calculados`, `sistema_percola`/`percola`, `current_final`/`intensidad_final`,
`Temperatura_final`/`temperatura_final`) y claves que nadie leía (`k_maxima`, `CF_destruido_index`,
`voltaje_max_reset`, `tiempo_sp_reset`). Aquí cada dato tiene un campo con un nombre.

El formato en disco (Init_data/phase_state_N_<fase>.npz/.json) NO cambia: `a_dict` escribe las
claves de siempre y `desde_dict` acepta todas las variantes antiguas, de modo que los estados ya
guardados y los que preparan los notebooks (Preparar_pp_reset*, crear_estado_densidad) siguen
valiendo. Las claves que no corresponden a ningún campo se conservan en `extra` (p. ej.
`vecindad_inicial`, que ponen los notebooks y acaba en la metadata).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

#: Etapas del ciclo, en orden. Definición única: el resto del paquete la importa de aquí.
FASES = ("pp_set", "sp_set", "pp_reset", "sp_reset")

#: Claves que se escribían y ya no lee nadie: se descartan al cargar.
CLAVES_OBSOLETAS = {"k_maxima", "CF_destruido_index", "voltaje_max_reset", "tiempo_sp_reset"}

#: Campo -> clave en disco (la que se escribe). Las variantes antiguas se aceptan al leer (ALIAS).
CLAVE_EN_DISCO = {
    "actual_state": "actual_state",
    "temperatura_final": "Temperatura_final",
    "perfil_electrodo": "perfil_electrodo",
    "voltaje": "voltaje",
    "centros": "centros_calculados",
    "percola": "sistema_percola",
    "T_max_fils": "T_max_fils",
    "estados_guardados": "estados_guardados",
    "cf_clean_matrix": "cf_clean_matrix",
    "voltaje_percolacion": "voltaje_percolacion",
    "paso_percolacion": "paso_percolacion",
    "corriente_final": "current_final",
    "ocupacion_percola": "ocupacion_percola",
    "creaciones_dict": "creaciones_dict",
    "vacantes_generadas_set": "vacantes_generadas_set",
    "oxygen_state": "oxygen_state",
    "reloj_iones": "reloj_iones",
    "v_prev_reloj": "v_prev_reloj",
    "CF_destruido": "CF_destruido",
    "voltage_CF_destruido": "voltage_CF_destruido",
    "roturas_dict": "roturas_dict",
    "max_oxigenos_reset": "max_oxigenos_reset",
}

#: Variantes antiguas de cada campo, en orden de preferencia tras la clave principal.
ALIAS = {
    "temperatura_final": ("temperatura_final",),
    "centros": ("CF_centros",),
    "percola": ("percola",),
    "corriente_final": ("intensidad_final",),
}


@dataclass(eq=False)
class EstadoFase:
    #: Fase que produjo el estado: pp_set, sp_set, pp_reset o sp_reset.
    fase: str
    actual_state: Any
    #: Temperatura final tal como la guardan las fases: mapa FVM (Ny, Nx+2) o escalar.
    temperatura_final: Any
    #: Instante absoluto que le tocaría a la fila siguiente (en disco: tiempo_<fase>).
    tiempo_fin: float
    #: Configuración con la que se simuló. No se guarda: se reconstruye del CSV al cargar.
    params: Any = None
    sim_ctes: Any = None
    #: Perfil (o escalar) del electrodo con el que arranca la fase siguiente; None = la constante.
    perfil_electrodo: Any = None
    #: Estado del controlador de voltaje (`VoltageController.estado()`); la siguiente lee su V_fin.
    voltaje: Optional[dict] = None
    #: Fila central de cada filamento.
    centros: Optional[list] = None
    percola: bool = False
    #: Pico de temperatura de cada filamento en la fase (solo pasos FVM).
    T_max_fils: Optional[list] = None
    #: Pasos con estado intermedio guardado.
    estados_guardados: List[int] = field(default_factory=list)

    # --- SET
    cf_clean_matrix: Any = None
    voltaje_percolacion: Optional[float] = None
    paso_percolacion: Optional[int] = None
    corriente_final: Optional[float] = None
    ocupacion_percola: Any = None
    creaciones_dict: Dict[Any, Any] = field(default_factory=dict)
    #: Vacantes generadas en el SET (sin las iniciales); limita los oxígenos del RESET.
    vacantes_generadas_set: Optional[int] = None

    # --- RESET
    oxygen_state: Any = None
    reloj_iones: Any = None
    v_prev_reloj: Optional[float] = None
    CF_destruido: Any = None
    voltage_CF_destruido: Any = None
    roturas_dict: Dict[Any, Any] = field(default_factory=dict)
    max_oxigenos_reset: Optional[int] = None

    #: Claves en disco que no corresponden a ningún campo (se conservan al volver a guardar).
    extra: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ disco

    def a_dict(self) -> dict:
        """Diccionario con las claves de siempre (lo que escribe `persistence.save_phase_state`)."""
        d = {"params": self.params, "sim_ctes": self.sim_ctes}
        for campo, clave in CLAVE_EN_DISCO.items():
            valor = getattr(self, campo)
            if valor is None or (campo in ("creaciones_dict", "roturas_dict") and not valor):
                continue  # sin dato: no se escribe (los diccionarios de eventos vacíos tampoco)
            d[clave] = valor
        d[f"tiempo_{self.fase}"] = self.tiempo_fin
        d.update(self.extra)
        return d

    @classmethod
    def desde_dict(cls, d: dict, fase: str) -> "EstadoFase":
        """Desde el dict de disco o de un notebook, con cualquiera de las variantes antiguas de las claves."""
        d = dict(d)
        usadas = {"params", "sim_ctes"} | CLAVES_OBSOLETAS
        valores = {}
        for campo, clave in CLAVE_EN_DISCO.items():
            for candidata in (clave,) + ALIAS.get(campo, ()):
                if candidata in d:
                    valores.setdefault(campo, d[candidata])
                    usadas.add(candidata)
        clave_tiempo = f"tiempo_{fase}"
        if clave_tiempo not in d:
            raise KeyError(f"El estado de {fase} no trae '{clave_tiempo}' (instante de traspaso a la fase siguiente).")
        usadas.add(clave_tiempo)
        for obligatoria in ("actual_state", "temperatura_final"):
            if obligatoria not in valores:
                raise KeyError(f"El estado de {fase} no trae '{CLAVE_EN_DISCO[obligatoria]}'.")
        extra = {k: v for k, v in d.items() if k not in usadas and not k.startswith("tiempo_")}
        for campo in ("creaciones_dict", "roturas_dict"):
            if valores.get(campo) is None:
                valores.pop(campo, None)
        if valores.get("estados_guardados") is None:
            valores.pop("estados_guardados", None)
        return cls(fase=fase, tiempo_fin=d[clave_tiempo], params=d.get("params"), sim_ctes=d.get("sim_ctes"), extra=extra, **valores)


def como_estado_fase(estado, fase: str) -> EstadoFase:
    """Acepta un `EstadoFase` o el diccionario antiguo (notebooks, estados guardados) y devuelve un `EstadoFase`."""
    return estado if isinstance(estado, EstadoFase) else EstadoFase.desde_dict(estado, fase)
