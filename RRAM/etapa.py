"""
Bucle de una etapa del ciclo (PP_set, SP_set, PP_reset, SP_reset).

Las cuatro etapas comparten el mismo paso:

    1. el controlador decide el voltaje con las medidas del paso anterior
    2. campo eléctrico y topología de los filamentos
    3. conducción y temperatura: Ohm + FVM si percola, Poole-Frenkel + Joule si no
    4. proceso estocástico: generación de vacantes (SET) o recombinación con oxígeno (RESET)
    5. fila de datos y, si toca, estado intermedio

`Etapa.ejecutar` es ese bucle, escrito una sola vez, y `Etapa._cerrar` el final común (datos,
último estado, estado final, `EstadoFase` y validación). Lo que cambia de una etapa a otra son
métodos que cada una redefine; los propios de SET y de RESET están en `phases_set.EtapaSet` y
`phases_reset.EtapaReset`, y los de cada fase en sus cuatro subclases.

Colaboradores: `VoltageController` (voltaje), `CacheTopologia` (filamentos), `fisica_paso` (rama
eléctrica, muro y FVM), `EstadoTermico` (temperatura, amortiguación y electrodo), los procesos de
`procesos_estocasticos`, `RegistroDatos` y `GrabadorEstados` (disco) y `PasoSimulacion` (lo que
calcula cada paso).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np

from . import fisica_paso, utils
from .acumulador_tiempo import AcumuladorTiempo
from .estado_fase import EstadoFase
from .estado_termico import EstadoTermico
from .grabador import GrabadorEstados, RegistroDatos
from .paso_simulacion import PasoSimulacion
from .topologia_filamentos import CacheTopologia, TopologiaFilamentos
from .voltage_controller import Medidas, ProtocoloVoltaje

logger = logging.getLogger(__name__)


@dataclass
class ContextoPaso:
    """Datos de entrada de un paso, comunes a todas las etapas."""

    k: int
    voltage: float
    #: Tiempo efectivo acumulado a voltaje constante (multiplica las tasas de generación y recombinación).
    tiempo_efectivo: float
    #: Vacantes al empezar el paso.
    total_vacantes: Any
    #: Campo eléctrico local de cada fila.
    E_filas: Optional[np.ndarray] = None
    #: Campo uniforme V/L (solo lo usa el RESET).
    E_uniforme: Optional[float] = None


class Etapa:
    """Una etapa del ciclo. Las subclases fijan estos atributos y redefinen los métodos del final."""

    #: Nombre de la fase en ficheros, semillas y protocolo de voltaje: pp_set, sp_set, pp_reset, sp_reset.
    FASE = ""
    #: Nombre para los mensajes de log.
    NOMBRE = ""
    #: Subcarpeta de los estados intermedios: "set" o "reset".
    SUBCARPETA = ""
    #: La fila de datos lleva el tiempo efectivo como última columna.
    CON_T_EF = False
    #: Series de 3 columnas que acompañan a los datos: {nombre: cabecera}.
    SERIES_EXTRA: Optional[dict] = None

    def __init__(
        self,
        *,
        num_simulation: int,
        params,
        sim_ctes,
        CF_ranges: List[tuple],
        centros: list,
        actual_state: np.ndarray,
        protocolo: ProtocoloVoltaje,
        cada_estados: int,
        previo: Optional[EstadoFase] = None,
        usar_muro: bool = True,
        results_dir="Results",
        tiempo_origen: float = 0.0,
        corriente_inicial: float = 0.0,
        percola_inicial: bool = False,
    ):
        #: Generador aleatorio de la fase, compartido por sus procesos estocásticos.
        self.rng = utils.crear_rng(params, self.FASE)
        self.num_simulation = num_simulation
        self.params = params
        #: Constantes de la simulación. PP_set las cambia al formarse cada filamento.
        self.sim_ctes = sim_ctes
        self.CF_ranges = CF_ranges
        self.N = len(CF_ranges)
        self.centros = centros
        self.actual_state = actual_state
        self.usar_muro = usar_muro

        self.rutas = utils.crear_rutas_simulacion(num_simulation=num_simulation, state=self.SUBCARPETA, results_dir=results_dir)
        for carpeta in ("simulation_path", "figures_path", "data_simulation_path"):
            self.rutas[carpeta].mkdir(parents=True, exist_ok=True)

        # Todo el voltaje lo decide el protocolo; de la etapa anterior solo se toma su V final
        # (para 'anterior' y para el informe de enganche). El presupuesto es la cota superior de pasos.
        self.controller = protocolo.controlador(self.FASE, previo=previo)
        self.presupuesto = self.controller.presupuesto()
        #: Último voltaje aplicado (antes del primer paso, el final de la etapa anterior).
        self.voltage = self.controller.v_previo
        #: Corriente y percolación del último paso: lo primero que lee el controlador en el siguiente.
        self.current = corriente_inicial
        self.percola = percola_inicial
        #: Instante absoluto en que empieza la etapa.
        self.tiempo_origen = tiempo_origen

        # Temperatura del paso, la del paso anterior y la de los electrodos
        if previo is None:
            self.termico = EstadoTermico.inicial(params, sim_ctes, centros=centros, N=self.N)
        else:
            self.termico = EstadoTermico.heredado(previo, params, sim_ctes, centros=centros, N=self.N)
        #: Lo que calcula el paso en curso. Tras el bucle queda el del último paso simulado.
        self.paso = PasoSimulacion()
        #: Máximo histórico de temperatura de cada filamento (solo pasos que resuelven el FVM).
        self.T_max_fils: List[Optional[float]] = [None] * self.N
        #: Filas intermedias entre filamentos y distancias a ellas: geometría del muro térmico.
        self.filas_intermedias = None
        self.dist_casillas = None

        self.registro = RegistroDatos(
            self.rutas["simulation_path"] / f"Data_{self.FASE}_{num_simulation}",
            self.presupuesto,
            self.N,
            actual_state.shape[0],
            con_t_ef=self.CON_T_EF,
            series_extra=self.SERIES_EXTRA,
        )
        self.grabador = GrabadorEstados(self.rutas["data_simulation_path"], self.FASE, num_simulation, cada_estados)
        # Topología de los filamentos: se reanaliza solo cuando el estado cambia
        self.cache_topologia = CacheTopologia(CF_ranges)
        self.acumulador_tiempo = AcumuladorTiempo(params.paso_temporal)

    # ================================================================== bucle

    def ejecutar(self) -> EstadoFase:
        """Simula la etapa y devuelve su `EstadoFase`. Si se para a mitad, guarda las filas ya calculadas."""
        logger.info(f"Simulacion {self.num_simulation} - {self.NOMBRE}")
        k = 0
        with self.registro.volcar_si_falla():
            for k in range(0, self.presupuesto + 1):
                T_electrodo_paso = self.termico.T_electrodo  # la que usa el solver en este paso (antes de actualizarla)
                total_vacantes = np.sum(self.actual_state)
                self.comprobar_antes_del_paso(k, total_vacantes)

                # El controlador decide con las medidas del paso ANTERIOR (mismo convenio que la temperatura)
                voltage = self.controller.next(
                    Medidas(
                        I_total=float(self.current),
                        n_vacantes=int(total_vacantes),
                        n_filamentos=self.filamentos_para_el_controlador(),
                        percola=self.percola,
                        k=k,
                    )
                )
                tiempo_efectivo = self.acumulador_tiempo.actualizar(voltage)
                self.voltage = voltage

                # La forma de onda ha terminado (su último segmento acabó en el paso anterior).
                # Se corta ANTES de la física para no añadir un punto de más.
                if self.controller.terminado:
                    break

                # Todo lo que calcula este paso empieza vacío: lo que no calcule no puede arrastrarse del anterior
                self.paso = PasoSimulacion()
                ctx = ContextoPaso(k=k, voltage=voltage, tiempo_efectivo=tiempo_efectivo, total_vacantes=total_vacantes)
                self.empezar_paso(ctx)
                ctx.E_filas, ctx.E_uniforme = self.campo_electrico(voltage)

                # Conducción y temperatura, según haya o no un camino entre electrodos
                topo = self.cache_topologia.analizar(self.actual_state)
                self.seguir_filamentos(ctx, topo)
                if topo.percola:
                    self.rama_percolante(ctx, topo)
                else:
                    self.rama_no_percolante(ctx)
                self.percola = topo.percola

                # Generación o recombinación con la temperatura de este paso
                self.actual_state = self.actualizar_estado(ctx)

                # Fila de datos y, si toca, estado intermedio
                tiempo = self.params.paso_temporal * k + self.tiempo_origen
                self.registro.escribir(
                    k,
                    self.paso.fila(tiempo, voltage, self.current, *self.columnas_extra(ctx)),
                    T_electrodo_paso,
                    **self.series_extra(ctx),
                )
                self.grabador.paso(k, self.estado_a_guardar)
        return self._cerrar(k)

    def _cerrar(self, k: int) -> EstadoFase:
        """
        Cierra la etapa ENTERA antes de validarla: si la simulación se para por no formar o no destruir
        los filamentos, queda guardado todo lo calculado (datos, último estado, estado final y, desde
        la excepción, las variables de la fase).
        """
        self.registro.guardar()
        # Traspaso a la etapa siguiente: instante absoluto que le tocaría a la fila siguiente
        tiempo_fin = self.tiempo_origen + self.params.paso_temporal * self.registro.n_filas
        logger.info(
            f"{self.NOMBRE} termina: {self.registro.n_filas} filas, V final {self.registro.filas[-1, 1]:.5f} V, "
            f"t traspaso {tiempo_fin:.5f} s"
        )
        # Estado del último paso, si no cayó en la cadencia de guardado
        self.grabador.cerrar(k, self.estado_a_guardar)
        np.savez(self.ruta_estado_final(), self.actual_state)
        logger.info(f"Temperatura máxima por filamento en {self.FASE}: {self.T_max_fils} K")

        estado = EstadoFase(
            fase=self.FASE,
            actual_state=self.actual_state,
            temperatura_final=self.termico.actual.legado(),
            tiempo_fin=tiempo_fin,
            params=self.params,
            sim_ctes=self.sim_ctes,
            # Perfil (o escalar) del electrodo con el que arranca la etapa siguiente
            perfil_electrodo=self.termico.T_electrodo,
            # Estado del voltaje: la etapa siguiente lee de aquí su V_fin
            voltaje=self.controller.estado(),
            centros=self.centros,
            percola=self.percola,
            T_max_fils=self.T_max_fils,
            estados_guardados=self.grabador.pasos_guardados,
            **self.campos_estado_final(),
        )
        self.validar_final(estado)
        return estado

    # ================================================================== física común

    def conduccion_ohmica(self, ctx: ContextoPaso, cf_matrix: np.ndarray, *, con_muro: bool, perfiles_siempre: bool = True):
        """
        Paso percolante: corriente de Ohm sobre `cf_matrix` y mapa térmico FVM, amortiguado.

        `con_muro` coloca el muro térmico entre filamentos. Los perfiles de la temperatura anterior se
        extraen si hay muro o si `perfiles_siempre` (exige que la temperatura anterior sea un mapa).
        Devuelve el resultado óhmico.
        """
        ohm = fisica_paso.resolver_rama_electrica(
            ctx.voltage, cf_matrix, self.termico.anterior, self.params, self.sim_ctes, self.CF_ranges,
            simulation_path=self.rutas["simulation_path"], num_simulation=self.num_simulation, actual_state=self.actual_state,
        )
        self.paso.registrar_ohm(ohm)
        self.current = ohm.I_total

        if con_muro or perfiles_siempre:
            self.paso.perfiles = self.termico.perfiles_anteriores()
        if con_muro:
            self.paso.matriz_muro = fisica_paso.construir_muro_termico(
                cf_matrix, self.paso.perfiles, self.filas_intermedias, self.dist_casillas, self.params, self.sim_ctes
            )
        T_fvm = fisica_paso.resolver_temperatura_fvm(
            cf_matrix, self.paso.R_local, self.paso.I_fils, self.termico.T_electrodo, self.params, self.sim_ctes,
            self.CF_ranges, matriz_muro=self.paso.matriz_muro,
        )
        # Se guarda el mapa FVM sin amortiguar para poder compararlo con el amortiguado
        self.paso.temperatura_sin_amortiguar = T_fvm
        self.paso.T_fils = self.termico.registrar_fvm(T_fvm, self.sim_ctes)
        self.T_max_fils = utils.acumular_T_max_filamentos(self.T_max_fils, self.paso.T_fils)
        return ohm

    # ================================================================== lo que define cada etapa

    def comprobar_antes_del_paso(self, k: int, total_vacantes) -> None:
        """Comprobaciones previas al voltaje del paso (SET: malla llena de vacantes)."""

    def filamentos_para_el_controlador(self) -> int:
        """Número de filamentos que recibe el controlador de voltaje (condición `hasta_filamentos`)."""
        raise NotImplementedError

    def empezar_paso(self, ctx: ContextoPaso) -> None:
        """Lo primero de un paso ya confirmado (RESET: reloj de los iones)."""

    def campo_electrico(self, voltage: float):
        """Devuelve (campo local por filas, campo uniforme o None)."""
        raise NotImplementedError

    def seguir_filamentos(self, ctx: ContextoPaso, topo: TopologiaFilamentos) -> None:
        """Seguimiento de filamentos previo a la conducción, percole o no (PP_reset: roturas)."""

    def rama_percolante(self, ctx: ContextoPaso, topo: TopologiaFilamentos) -> None:
        """Hay camino entre electrodos: corriente de Ohm y mapa térmico."""
        raise NotImplementedError

    def rama_no_percolante(self, ctx: ContextoPaso) -> None:
        """No hay camino: corriente de Poole-Frenkel y temperatura de Joule."""
        raise NotImplementedError

    def actualizar_estado(self, ctx: ContextoPaso) -> np.ndarray:
        """Proceso estocástico del paso. Devuelve el nuevo estado de vacantes."""
        raise NotImplementedError

    def columnas_extra(self, ctx: ContextoPaso) -> tuple:
        """Columnas tras las de filamento en la fila de datos (RESET: tiempo efectivo)."""
        return ()

    def series_extra(self, ctx: ContextoPaso) -> dict:
        """Valores de `SERIES_EXTRA` en este paso."""
        return {}

    def estado_a_guardar(self) -> dict:
        """Matrices de un estado intermedio. Solo se evalúa cuando toca guardar."""
        raise NotImplementedError

    def ruta_estado_final(self):
        return self.rutas["simulation_path"] / f"Final_state_{self.FASE}_{self.num_simulation}.npz"

    def campos_estado_final(self) -> dict:
        """Campos de `EstadoFase` propios de la etapa."""
        return {}

    def validar_final(self, estado: EstadoFase) -> None:
        """Lanza la excepción que corresponda si la etapa no cumplió su objetivo. `estado` viaja en ella."""
