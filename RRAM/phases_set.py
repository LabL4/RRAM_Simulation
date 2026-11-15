"""
Etapas del SET: PP_set (formación de los filamentos) y SP_set (meseta con los filamentos formados).

El bucle del paso está en `etapa.Etapa`. Aquí está lo propio del SET (`EtapaSet`: generación de
vacantes, campo local, Poole-Frenkel del SET) y lo que distingue a cada una de las dos etapas.
`PP_set` y `SP_set` son las funciones de siempre: construyen la etapa y la ejecutan.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import numpy as np

from . import CurrentSolver, ElectricField, Generation, Temperature, exceptions, utils
from .constants_simulation import SimulationConstants
from .estado_fase import EstadoFase, como_estado_fase
from .etapa import ContextoPaso, Etapa
from .filament_tracking import actualizar_parametros_por_filamento, procesar_filamentos_creados
from .parameters import SimulationParameters
from .procesos_estocasticos import ProcesoGeneracion
from .topologia_filamentos import TopologiaFilamentos
from .voltage_controller import ProtocoloVoltaje

logger = logging.getLogger(__name__)


class EtapaSet(Etapa):
    """Lo común a las dos etapas del SET. Cada una crea su `self.generacion` y su `self.cf_clean_matrix`."""

    SUBCARPETA = "set"

    #: Proceso de generación de vacantes de la etapa.
    generacion: ProcesoGeneracion
    #: Matriz de filamentos completos con la que se calculan resistencias y calor.
    cf_clean_matrix = None

    def comprobar_antes_del_paso(self, k: int, total_vacantes) -> None:
        if total_vacantes > int(self.params.num_max_vacantes):
            # La malla se ha llenado de vacantes: la simulación no es válida
            raise exceptions.MaxVacantesException(k=k - 1, voltage=self.voltage)

    def campo_electrico(self, voltage: float):
        E_filas = ElectricField.GapElectricFieldVector(voltage, self.actual_state, self.params.device_size_x, self.params.atom_size)
        return E_filas, None

    def muro_activo(self) -> bool:
        """Muro térmico entre filamentos: con `usar_muro` y geometría de muro definida."""
        return self.usar_muro and self.filas_intermedias is not None and self.dist_casillas is not None

    def poole_frenkel(self, temperatura, campo) -> float:
        """Corriente sin percolación, con las constantes del SET."""
        return CurrentSolver.Poole_Frenkel(
            temperatura,
            campo,
            pb_metal_insul=self.sim_ctes.pb_metal_insul_set,
            permitividad_relativa=self.sim_ctes.permitividad_relativa_set,
            I_0=self.sim_ctes.I_0_set,
        ) * (self.params.device_size_y)

    def temperatura_joule(self, ctx: ContextoPaso):
        """Joule escalar con la corriente del paso anterior."""
        return Temperature.Temperature_Joule(
            ctx.voltage, self.current, T_0=self.params.init_temp, r_termica=self.sim_ctes.r_termica_no_percola
        )

    def actualizar_estado(self, ctx: ContextoPaso) -> np.ndarray:
        estado, self.paso.probabilidad_generacion = self.generacion.actualizar(
            self.actual_state, ctx.total_vacantes, self.params, self.sim_ctes, ctx.E_filas,
            self.termico.actual.interior, ctx.k, ctx.tiempo_efectivo,
        )
        return estado

    def estado_a_guardar(self) -> dict:
        return dict(
            actual_state=self.actual_state,
            cf_clean_matrix=self.cf_clean_matrix,
            temperatura=self.termico.actual.legado(),
            matriz_para_plot_muro=self.paso.matriz_muro_para_plot(self.centros),
            mapa_resistencias=self.paso.R_local,
            **Temperature.extra_sin_amortiguar(self.sim_ctes, self.paso.temperatura_sin_amortiguar),
        )


class EtapaPPSet(EtapaSet):
    """
    Primera parte del SET: se generan vacantes hasta que percolan y forman los filamentos esperados.

    Parte sin percolación (Poole-Frenkel + Joule). Al percolar el primer filamento pasa a Ohm + FVM.
    Termina con error si no percola (`NoPercolationException`) o no se forman todos los filamentos
    (`FilamentosNoFormadosException`); en ambos casos la etapa se cierra antes y su estado viaja en la excepción.
    """

    FASE = "pp_set"
    NOMBRE = "PP_set"
    SERIES_EXTRA = {"vacantes": "paso, Voltaje [V], Total Vacantes", "resistencia": "paso, Voltaje [V], Resistencia [Ohm]"}

    def __init__(
        self,
        num_simulation: int,
        params: SimulationParameters,
        sim_ctes: SimulationConstants,
        CF_ranges: List[tuple],
        CF_creado: np.ndarray,
        protocolo: ProtocoloVoltaje,
        cada_estados: int,
        CF_centros: List[int] | None = None,
        actual_state: np.ndarray | None = None,
        usar_muro: bool = True,
        results_dir="Results",
    ):
        # Estado inicial: prioridad al argumento explícito, fallback al disco
        if actual_state is None:
            actual_state = utils.cargar_estado(Path.cwd() / f"Init_data/init_state_{num_simulation - 1}")
        else:
            actual_state = actual_state.copy()
        super().__init__(
            num_simulation=num_simulation, params=params, sim_ctes=sim_ctes, CF_ranges=CF_ranges, centros=CF_centros,
            actual_state=actual_state, protocolo=protocolo, cada_estados=cada_estados, usar_muro=usar_muro,
            results_dir=results_dir,
        )
        # Estado antes de generar, para contar las vacantes generadas en la etapa
        self.estado_inicial = actual_state.copy()
        logger.info(f"El número de vacantes iniciales es {actual_state.sum()}")
        logger.info(f"Los centros de cada filamento son: {CF_centros}")
        logger.info(f"El grosor de los filamentos es de {sim_ctes.grosor_filamento} filas")
        logger.info(
            f"Factor vecinos y factor libre: {sim_ctes.factor_vecinos_pp_set} y {sim_ctes.factor_libre_pp_set} "
            f"(vecinos: {sim_ctes.neighbor_mode_pp_set}). Gamma: {sim_ctes.gamma}"
        )

        # Seguimiento de la percolación y de los filamentos
        self.CF_creado = CF_creado
        self.voltage_CF_creado = np.full(self.N, 0.0)
        self.creaciones_dict: dict = {}  # mismo formato que roturas_dict
        self.filamentos_formados = 0
        self.all_CFs_created = False
        self.voltaje_percolacion = None
        self.paso_percolacion = None  # fila de Data_pp_set en que percola por primera vez
        self.ocupacion_percola = None  # vacantes al percolar por primera vez
        # Al inicio la corriente es de Poole-Frenkel: la resistencia óhmica se registra como 0
        self.resistencia = 0.0

        # Generación limitada a la banda de cada filamento esperado y a la ocupación máxima
        self.generacion = ProcesoGeneracion(
            sim_ctes.factor_vecinos_pp_set,
            sim_ctes.factor_libre_pp_set,
            int(sim_ctes.ocupacion_max_pp_set * params.num_max_vacantes),
            sim_ctes.neighbor_mode_pp_set,
            mascara=Generation.create_custom_mask(state=actual_state, centros_CF=CF_centros, grosor_CF=sim_ctes.grosor_filamento),
            aviso_cupo=f"Se ha alcanzado la ocupación máxima del {sim_ctes.ocupacion_max_pp_set * 100}% en la primera parte del set",
            rng=self.rng,
        )

        # Muros estáticos: sus filas dependen solo de la geometría esperada (los centros), así que se
        # calculan ya. El muro se coloca en cuanto el FVM entra con el PRIMER filamento. Con un solo
        # filamento no hay interfaces: se quedan en None y no hay muro.
        if self.N >= 2:
            self.filas_intermedias, self.dist_casillas = Temperature.calcular_filas_intermedias(CF_centros)  # type: ignore

    def filamentos_para_el_controlador(self) -> int:
        return int(np.sum(self.CF_creado))

    def seguir_filamentos(self, ctx: ContextoPaso, topo: TopologiaFilamentos) -> None:
        if not topo.percola:
            return
        if not self.percola:  # primera percolación (el paso anterior no percolaba)
            self.grabador.marcar_evento()
            self.voltaje_percolacion = ctx.voltage
            self.paso_percolacion = ctx.k
            self.ocupacion_percola = np.sum(self.actual_state)
            logger.info(
                f"El sistema ha percolado en la iteración {ctx.k}, a {round(ctx.voltage, 5)} V, con "
                f"{int(self.ocupacion_percola)} vacantes "
                f"({round(self.ocupacion_percola / self.params.num_max_vacantes, 4) * 100} % de ocupación)"
            )
            # La temperatura anterior pasa de escalar (Joule) a mapa: la necesita el mapa de resistencias
            self.termico.anterior_a_mapa_si_escalar(self.actual_state.shape)

        # Filamentos nuevos
        if any(~self.CF_creado):
            creados_antes = int(np.sum(self.CF_creado))
            procesar_filamentos_creados(
                imagen_path=self.rutas["figures_path"],
                data_save_path=self.rutas["data_simulation_path"],
                existentes=topo.existe,
                CF_creado=self.CF_creado,
                voltage=ctx.voltage,
                voltage_CF_creado=self.voltage_CF_creado,
                actual_state=self.actual_state,
                num_simulation=self.num_simulation,
                k=ctx.k,
                creaciones_dict=self.creaciones_dict,
                etapa=self.FASE,
            )
            if int(np.sum(self.CF_creado)) > creados_antes:
                self.grabador.marcar_evento()  # filamento nuevo: se guarda este paso

        # Al formarse filamentos cambian las constantes (gamma) y, con todos formados, la geometría del muro
        formados = sum(self.CF_creado)
        if formados > self.filamentos_formados:
            self.sim_ctes, self.all_CFs_created, filas, distancias = actualizar_parametros_por_filamento(
                formados, self.filamentos_formados, self.CF_ranges, self.centros, self.sim_ctes, self.all_CFs_created
            )
            if filas is not None:
                self.filas_intermedias, self.dist_casillas = filas, distancias
            self.filamentos_formados = formados

    def rama_percolante(self, ctx: ContextoPaso, topo: TopologiaFilamentos) -> None:
        self.cf_clean_matrix = topo.matriz_filamentos
        # Si percola hay siempre al menos un filamento formado: la celda de contacto con el electrodo
        # izquierdo cae en la banda de alguno. Si no, el estado es incoherente.
        if not any(topo.existe):
            raise exceptions.PercolacionSinFilamentoException(
                simulation_path=self.rutas["simulation_path"],
                voltage=ctx.voltage,
                num_simulation=self.num_simulation,
                actual_state=self.actual_state,
            )
        # En cuanto percola el PRIMER filamento se resuelve el FVM, sin esperar a que estén todos
        ohm = self.conduccion_ohmica(ctx, self.cf_clean_matrix, con_muro=self.muro_activo())
        self.resistencia = ohm.R_total

    def rama_no_percolante(self, ctx: ContextoPaso) -> None:
        campo_medio = np.mean(ctx.E_filas).item()
        # Temperatura de Joule, amortiguada contra la escalar del paso anterior
        T_joule = self.temperatura_joule(ctx)
        self.paso.temperatura_sin_amortiguar = T_joule
        self.paso.T_fils = self.termico.registrar_joule(T_joule, self.sim_ctes)
        self.paso.sin_percolacion(self.N)
        # Con el cupo de vacantes lleno la corriente se queda en su último valor
        if not self.generacion.cupo_alcanzado:
            self.current = self.poole_frenkel(self.termico.actual.escalar, campo_medio)

    def series_extra(self, ctx: ContextoPaso) -> dict:
        return dict(
            vacantes=np.array([ctx.k, ctx.voltage, ctx.total_vacantes]),
            resistencia=np.array([ctx.k, ctx.voltage, self.resistencia]),
        )

    def estado_a_guardar(self) -> dict:
        estado = super().estado_a_guardar()
        estado["probabilidad_matrix"] = self.paso.probabilidad_generacion
        return estado

    def ruta_estado_final(self):
        return self.rutas["simulation_path"] / f"Final_state_{self.num_simulation}_pp_set.npz"

    def campos_estado_final(self) -> dict:
        vacantes_generadas = Generation.contar_vacantes_generadas(self.estado_inicial, self.actual_state)
        logger.info(f"Vacantes generadas en PP_set: {vacantes_generadas}")
        return dict(
            cf_clean_matrix=self.cf_clean_matrix,
            voltaje_percolacion=self.voltaje_percolacion,
            paso_percolacion=self.paso_percolacion,
            corriente_final=self.current,
            ocupacion_percola=self.ocupacion_percola,
            creaciones_dict=self.creaciones_dict,
            # Vacantes generadas en PP_set (sin las iniciales); SP_set le suma las suyas
            vacantes_generadas_set=vacantes_generadas,
        )

    def validar_final(self, estado: EstadoFase) -> None:
        if not self.percola:
            raise exceptions.NoPercolationException(estado_fase=estado)
        # Si no se forman los filamentos esperados se descarta la simulación
        if self.filamentos_formados < self.N:
            raise exceptions.FilamentosNoFormadosException(
                simulation_path=self.rutas["simulation_path"],
                num_simulation=self.num_simulation,
                actual_state=self.actual_state,
                CF_formados=self.filamentos_formados,
                CF_esperados=self.N,
                estado_fase=estado,
            )


class EtapaSPSet(EtapaSet):
    """
    Segunda parte del SET: los filamentos ya están formados y su geometría (la que dejó PP_set) no
    cambia. Sigue generando vacantes hasta `ocupacion_max_sp_set` por encima de las que había al percolar.
    """

    FASE = "sp_set"
    NOMBRE = "SP_set"

    def __init__(
        self,
        previo: EstadoFase,
        num_simulation: int,
        CF_ranges: List[tuple],
        protocolo: ProtocoloVoltaje,
        cada_estados: int,
        usar_muro: bool = True,
        results_dir="Results",
    ):
        previo = como_estado_fase(previo, "pp_set")  # un dict antiguo también vale
        super().__init__(
            num_simulation=num_simulation, params=previo.params, sim_ctes=previo.sim_ctes, CF_ranges=CF_ranges,
            centros=previo.centros, actual_state=previo.actual_state, protocolo=protocolo, cada_estados=cada_estados,
            previo=previo, usar_muro=usar_muro, results_dir=results_dir, tiempo_origen=previo.tiempo_fin,
            corriente_inicial=previo.corriente_final, percola_inicial=previo.percola,
        )
        self.cf_clean_matrix = previo.cf_clean_matrix
        # Estado antes de generar y vacantes generadas en PP_set (None en estados antiguos sin el dato)
        self.estado_inicial = self.actual_state.copy()
        self.vacantes_generadas_pp_set = previo.vacantes_generadas_set
        logger.info(f"Vacantes al inicio del SP set: {np.sum(self.actual_state)}; al percolar: {previo.ocupacion_percola}")

        sim_ctes, params = self.sim_ctes, self.params
        self.generacion = ProcesoGeneracion(
            sim_ctes.factor_vecinos_sp_set,
            sim_ctes.factor_libre_sp_set,
            previo.ocupacion_percola + int(sim_ctes.ocupacion_max_sp_set * params.num_max_vacantes),
            sim_ctes.neighbor_mode_sp_set,
            aviso_cupo=f"Se ha alcanzado la ocupación máxima del {sim_ctes.ocupacion_max_sp_set * 100}% en la segunda parte del set",
            rng=self.rng,
        )

        self.filas_intermedias, self.dist_casillas = Temperature.calcular_filas_intermedias(self.centros)
        logger.info(
            f"Centros de los filamentos: {self.centros}; filas intermedias: {self.filas_intermedias}; "
            f"distancias: {self.dist_casillas}"
        )

    def filamentos_para_el_controlador(self) -> int:
        return self.N

    def rama_percolante(self, ctx: ContextoPaso, topo: TopologiaFilamentos) -> None:
        # Con la geometría que dejó PP_set (SP_set no la recalcula)
        self.conduccion_ohmica(ctx, self.cf_clean_matrix, con_muro=self.muro_activo())

    def rama_no_percolante(self, ctx: ContextoPaso) -> None:
        # NO se amortigua: en SP_set no hay recombinación, así que el filamento que percoló en PP_set
        # no puede destruirse y esta rama no se alcanza. Si se alcanzara, la temperatura anterior es
        # una matriz y el resultado ya no sería escalar.
        T_joule = self.temperatura_joule(ctx)
        self.paso.T_fils = self.termico.registrar_joule(T_joule, self.sim_ctes, amortiguar=False, actualizar_anterior=False)
        self.current = self.poole_frenkel(self.termico.actual.escalar, np.mean(ctx.E_filas).item())
        self.paso.sin_percolacion(self.N)

    def campos_estado_final(self) -> dict:
        generadas = Generation.contar_vacantes_generadas(self.estado_inicial, self.actual_state)
        if self.vacantes_generadas_pp_set is None:
            total = None
            logger.warning(
                f"El estado de PP_set no trae 'vacantes_generadas_set': no se puede dar el total del set "
                f"(SP_set ha generado {generadas}). El reset necesitará 'max_oxigenos_reset' en la configuración."
            )
        else:
            total = int(self.vacantes_generadas_pp_set) + generadas
            logger.info(
                f"Vacantes generadas en el set: {total} (PP_set {self.vacantes_generadas_pp_set} + SP_set {generadas})"
            )
        # Vacantes generadas en todo el set (sin las iniciales): limita los oxígenos del reset
        return dict(vacantes_generadas_set=total)


def PP_set(
    num_simulation: int,
    params: SimulationParameters,
    sim_ctes: SimulationConstants,
    CF_ranges: List[tuple],
    CF_creado: np.ndarray,
    protocolo: ProtocoloVoltaje,
    num_pasos_guardar_estado: int,
    CF_centros: List[int] | None = None,
    actual_state: np.ndarray | None = None,
    usar_muro: bool = True,
    results_dir: str = "Results",
) -> EstadoFase:
    """
    Primera parte del SET (ver `EtapaPPSet`). Devuelve el `EstadoFase` con el que arranca `SP_set`.

    `CF_creado` (un booleano por filamento) se actualiza in situ. `actual_state` es el estado inicial;
    si es None se carga `Init_data/init_state_{num_simulation - 1}`. `num_pasos_guardar_estado` es la
    cadencia de los estados intermedios (0 = no se guardan).

    Lanza `MaxVacantesException`, `NullResistanceException`, `NoPercolationException`,
    `FilamentosNoFormadosException` o `PercolacionSinFilamentoException`.
    """
    return EtapaPPSet(
        num_simulation, params, sim_ctes, CF_ranges, CF_creado, protocolo, num_pasos_guardar_estado,
        CF_centros=CF_centros, actual_state=actual_state, usar_muro=usar_muro, results_dir=results_dir,
    ).ejecutar()


def SP_set(
    final_state_pp_set: EstadoFase,
    num_simulation: int,
    CF_ranges: List[tuple],
    protocolo: ProtocoloVoltaje,
    num_pasos_guardar_estado: int,
    usar_muro: bool = True,
    results_dir: str = "Results",
) -> EstadoFase:
    """Segunda parte del SET (ver `EtapaSPSet`), desde el estado de `PP_set`. Devuelve el de `PP_reset`."""
    return EtapaSPSet(
        final_state_pp_set, num_simulation, CF_ranges, protocolo, num_pasos_guardar_estado,
        usar_muro=usar_muro, results_dir=results_dir,
    ).ejecutar()
