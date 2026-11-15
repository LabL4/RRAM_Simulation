"""
Etapas del RESET: PP_reset (rotura de los filamentos) y SP_reset (con los filamentos ya destruidos).

El bucle del paso está en `etapa.Etapa`. Aquí está lo propio del RESET (`EtapaReset`: iones de
oxígeno y recombinación, campo uniforme, Poole-Frenkel del RESET, tiempo efectivo en los datos) y
lo que distingue a cada una de las dos etapas. `PP_reset` y `SP_reset` son las funciones de
siempre: construyen la etapa y la ejecutan.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np

from . import CurrentSolver, ElectricField, Recombination, Temperature, exceptions
from .acumulador_tiempo import RelojesIones
from .estado_fase import EstadoFase, como_estado_fase
from .etapa import ContextoPaso, Etapa
from .filament_tracking import procesar_filamentos_destruidos
from .procesos_estocasticos import ProcesoRecombinacion
from .topologia_filamentos import TopologiaFilamentos
from .voltage_controller import ProtocoloVoltaje

logger = logging.getLogger(__name__)


class EtapaReset(Etapa):
    """
    Lo común a las dos etapas del RESET. Cada una crea su `self.recombinacion` y fija `CF_destruido`,
    `voltage_CF_destruido` y `roturas_dict`.
    """

    SUBCARPETA = "reset"
    CON_T_EF = True

    #: Proceso de los iones de oxígeno de la etapa.
    recombinacion: ProcesoRecombinacion
    #: Un booleano por filamento, voltaje de rotura de cada uno y registro de las roturas.
    CF_destruido: np.ndarray
    voltage_CF_destruido: np.ndarray
    roturas_dict: dict

    def filamentos_para_el_controlador(self) -> int:
        return int(np.sum(~self.CF_destruido))

    def empezar_paso(self, ctx: ContextoPaso) -> None:
        # Reloj de cada ion: después del corte de la forma de onda (el paso terminado repite el
        # último V y no se simula) y antes de generar (los iones nuevos empiezan en 0).
        self.recombinacion.empezar_paso(ctx.voltage)

    def campo_electrico(self, voltage: float):
        E_uniforme = abs(ElectricField.SimpleElectricField(voltage, self.params.device_size_x))
        # Campo local por filas: lo recibe el movimiento de los iones. Hoy la velocidad aplicada sale
        # de los umbrales de voltaje, pero es la entrada del modelo físico que se quiere recuperar.
        E_filas = np.abs(
            ElectricField.GapElectricFieldVector(voltage, self.actual_state, self.params.device_size_x, self.params.atom_size)
        )
        return E_filas, E_uniforme

    def joule_sin_percolar(self, ctx: ContextoPaso, temperatura_para_corriente) -> None:
        """
        Paso sin percolación: Poole-Frenkel con `temperatura_para_corriente` y, con esa corriente,
        temperatura de Joule amortiguada.

        El paso en que se pasa del mapa FVM al escalar de Joule (rotura del filamento) NO se
        amortigua: la temperatura anterior sigue siendo el mapa y no hay un valor de Joule con el que
        comparar. A partir del paso siguiente la anterior es escalar y ya se amortigua.
        """
        # TODO: usar la media del campo local por filas, np.mean(ctx.E_filas), como en el SET, en
        # lugar del campo uniforme. Pendiente de decidir: cambia la corriente del tramo sin percolación.
        self.current = abs(
            CurrentSolver.Poole_Frenkel(
                temperatura_para_corriente,
                ctx.E_uniforme,
                pb_metal_insul=self.sim_ctes.pb_metal_insul_reset,
                permitividad_relativa=self.sim_ctes.permitividad_relativa_reset,
                I_0=self.sim_ctes.I_0_reset,
            )
            * (self.params.device_size_y)
        )
        T_joule = self.temperatura_joule(ctx)
        self.paso.temperatura_sin_amortiguar = T_joule
        self.paso.T_fils = self.termico.registrar_joule(T_joule, self.sim_ctes)
        self.paso.sin_percolacion(self.N)

    def temperatura_joule(self, ctx: ContextoPaso):
        """Joule escalar con la corriente que haya en `self.current` en ese momento."""
        return Temperature.Temperature_Joule(
            ctx.voltage, self.current, T_0=self.params.init_temp, r_termica=self.sim_ctes.r_termica_no_percola
        )

    def actualizar_estado(self, ctx: ContextoPaso) -> np.ndarray:
        # La recombinación indexa con las coordenadas de actual_state: recibe la temperatura sin columnas de electrodo
        return self.recombinacion.actualizar(
            self.actual_state, ctx.voltage, ctx.E_filas, self.termico.actual.interior, self.params, self.sim_ctes,
            ctx.tiempo_efectivo,
        )

    def columnas_extra(self, ctx: ContextoPaso) -> tuple:
        return (ctx.tiempo_efectivo,)

    def estado_a_guardar(self) -> dict:
        return dict(
            actual_state=self.actual_state,
            oxygen_state=self.recombinacion.oxygen_state,
            probabilidad_recombinacion=self.recombinacion.probabilidad(),
            reloj_iones=self.recombinacion.relojes.reloj,
            cf_clean_matrix=self.paso.cf_clean_matrix,
            temperatura=self.termico.actual.legado(),
            matriz_para_plot_muro=self.paso.matriz_muro_para_plot(self.centros),
            mapa_resistencias=self.paso.R_local,
            **Temperature.extra_sin_amortiguar(self.sim_ctes, self.paso.temperatura_sin_amortiguar),
        )

    def campos_estado_final(self) -> dict:
        reloj, v_prev = self.recombinacion.relojes.estado()
        return dict(
            oxygen_state=self.recombinacion.oxygen_state,
            # Relojes por ion (pasos acumulados) y último V aplicado: la etapa siguiente los hereda
            reloj_iones=reloj,
            v_prev_reloj=v_prev,
            CF_destruido=self.CF_destruido,
            voltage_CF_destruido=self.voltage_CF_destruido,
            roturas_dict=self.roturas_dict,
            # Máximo de iones de oxígeno simultáneos
            max_oxigenos_reset=self.recombinacion.max_oxigenos,
        )

    def _informar_oxigeno(self) -> None:
        for umbral, n in self.recombinacion.oxygen_config.items():
            logger.info(f"  generación de oxígeno por encima de {umbral} V: hasta {n} por paso")
        for umbral, v in self.recombinacion.velocity_thresholds.items():
            logger.info(f"  velocidad de los iones por encima de {umbral} V: {v} m/s")


class EtapaPPReset(EtapaReset):
    """
    Primera parte del RESET: los iones de oxígeno recombinan vacantes hasta romper los filamentos.

    Parte percolando (Ohm + FVM) y pasa a Poole-Frenkel + Joule al romperse. Termina con
    `FilamentosNoDestruidosException` si no se rompen todos; la etapa se cierra antes y su estado
    viaja en la excepción.
    """

    FASE = "pp_reset"
    NOMBRE = "PP_reset"

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
        previo = como_estado_fase(previo, "sp_set")  # un dict antiguo también vale
        super().__init__(
            num_simulation=num_simulation, params=previo.params, sim_ctes=previo.sim_ctes, CF_ranges=CF_ranges,
            centros=previo.centros, actual_state=previo.actual_state, protocolo=protocolo, cada_estados=cada_estados,
            previo=previo, usar_muro=usar_muro, results_dir=results_dir, tiempo_origen=previo.tiempo_fin,
            # El dispositivo llega del SET en LRS, así que percola al entrar; aún no hay corriente
            corriente_inicial=0.0, percola_inicial=True,
        )
        sim_ctes = self.sim_ctes
        self.filas_intermedias, self.dist_casillas = Temperature.calcular_filas_intermedias(self.centros)

        self.CF_destruido = np.full(self.N, False, dtype=bool)
        self.voltage_CF_destruido = np.full(self.N, 0.0)
        self.roturas_dict: dict = {}

        oxygen_state = np.zeros_like(self.actual_state, dtype=np.int8)
        self.recombinacion = ProcesoRecombinacion(
            oxygen_state,
            RelojesIones(np.zeros(oxygen_state.shape, dtype=np.int32)),  # reloj propio de cada ion
            # CUIDADO: los umbrales tienen que ir ordenados de mayor a menor
            oxygen_config={
                float(sim_ctes.voltaje_gen_oxigeno_pp_2): int(sim_ctes.num_oxigenos_pp_reset_2),
                float(sim_ctes.voltaje_gen_oxigeno_pp_1): int(sim_ctes.num_oxigenos_pp_reset_1),
            },
            velocity_thresholds={
                float(sim_ctes.voltaje_gen_oxigeno_pp_2): sim_ctes.velocidad_iones_pp_2,
                float(sim_ctes.voltaje_gen_oxigeno_pp_1): sim_ctes.velocidad_iones_pp_1,
            },
            # Filas de la interfaz donde se pueden generar oxígenos: banda de los filamentos (centro ± grosor)
            filas_permitidas=Recombination.filas_interfaz_filamento(self.actual_state.shape[0], self.centros, sim_ctes.grosor_filamento),
            # Máximo de iones simultáneos: manual (configuración) o vacantes generadas en el set
            max_oxigenos=Recombination.resolver_limite_oxigenos(previo.vacantes_generadas_set, sim_ctes.max_oxigenos_reset),
            rng=self.rng,
        )
        self._informar_oxigeno()

    def seguir_filamentos(self, ctx: ContextoPaso, topo: TopologiaFilamentos) -> None:
        if not any(~self.CF_destruido):  # ya están todos rotos
            return
        destruidos_antes = int(np.sum(self.CF_destruido))
        procesar_filamentos_destruidos(
            imagen_path=self.rutas["figures_path"],
            data_save_path=self.rutas["data_simulation_path"],
            existentes=topo.existe,
            CF_destruido=self.CF_destruido,
            voltage=ctx.voltage,
            voltage_CF_destruido=self.voltage_CF_destruido,
            actual_state=self.actual_state,
            num_simulation=self.num_simulation,
            roturas_dict=self.roturas_dict,
            k=ctx.k,
            etapa=self.FASE,
        )
        if int(np.sum(self.CF_destruido)) > destruidos_antes:
            self.grabador.marcar_evento()  # rotura de un filamento: se guarda este paso

    def rama_percolante(self, ctx: ContextoPaso, topo: TopologiaFilamentos) -> None:
        self.paso.cf_clean_matrix = topo.matriz_filamentos
        # El muro solo aplica cuando TODOS los filamentos están intactos y hay al menos 2
        intactos = int(np.sum(~self.CF_destruido))
        con_muro = not np.any(self.CF_destruido) and self.usar_muro and intactos >= 2
        self.conduccion_ohmica(ctx, self.paso.cf_clean_matrix, con_muro=con_muro, perfiles_siempre=False)

    def rama_no_percolante(self, ctx: ContextoPaso) -> None:
        # La corriente se calcula con la temperatura de Joule de la corriente del paso anterior
        # (en k = 0 no hay corriente: la temperatura inicial)
        self.joule_sin_percolar(ctx, self.temperatura_joule(ctx))

    def validar_final(self, estado: EstadoFase) -> None:
        logger.info(f"La temperatura máxima al final del reset es: {self.termico.actual.maximo():.1f} K")
        if sum(self.CF_destruido) < self.N:
            raise exceptions.FilamentosNoDestruidosException(
                simulation_path=self.rutas["simulation_path"],
                figures_path=self.rutas["figures_path"],
                voltage=self.voltage,
                num_simulation=self.num_simulation,
                actual_state=self.actual_state,
                CF_destruidos=sum(self.CF_destruido),
                CF_esperados=self.N,
                estado_fase=estado,
            )


class EtapaSPReset(EtapaReset):
    """
    Segunda parte del RESET: sigue la recombinación con TODOS los filamentos ya destruidos.

    Lo comprueba al arrancar (rotos en el estado de partida y registrados por PP_reset); si no,
    `FilamentosNoDestruidosException` antes de simular. La recombinación solo quita vacantes, así
    que la etapa no puede volver a percolar.
    """

    FASE = "sp_reset"
    NOMBRE = "SP_reset"

    def __init__(
        self,
        previo: EstadoFase,
        num_simulation: int,
        CF_ranges: List[tuple],
        protocolo: ProtocoloVoltaje,
        cada_estados: int,
        results_dir="Results",
    ):
        previo = como_estado_fase(previo, "pp_reset")  # un dict antiguo también vale
        super().__init__(
            num_simulation=num_simulation, params=previo.params, sim_ctes=previo.sim_ctes, CF_ranges=CF_ranges,
            centros=previo.centros, actual_state=previo.actual_state, protocolo=protocolo, cada_estados=cada_estados,
            previo=previo, results_dir=results_dir, tiempo_origen=previo.tiempo_fin,
            # El controlador lee la corriente del paso anterior y en k = 0 todavía no hay ninguna
            corriente_inicial=0.0, percola_inicial=previo.percola,
        )
        sim_ctes = self.sim_ctes
        self.CF_destruido = previo.CF_destruido
        self.voltage_CF_destruido = previo.voltage_CF_destruido
        self.roturas_dict = previo.roturas_dict
        logger.info(f"Voltajes de rotura en PP_reset: {self.voltage_CF_destruido}; destruidos: {self.CF_destruido}")
        self._exigir_filamentos_destruidos()

        # Máximo de iones simultáneos: el que resolvió PP_reset. Si el manual de la configuración es
        # distinto (p. ej. al arrancar desde un pp_reset guardado), manda el manual; un estado
        # antiguo sin el dato y sin manual para con error.
        heredado = previo.max_oxigenos_reset
        if heredado is not None and sim_ctes.max_oxigenos_reset in (None, heredado):
            max_oxigenos = int(heredado)
            logger.info(f"Límite de oxígenos del sp reset (heredado de pp reset): {max_oxigenos}")
        else:
            max_oxigenos = Recombination.resolver_limite_oxigenos(heredado, sim_ctes.max_oxigenos_reset)

        # Relojes por ion heredados de PP_reset: solo se reinician si el primer V difiere del último de PP_reset
        reloj = previo.reloj_iones
        if reloj is None:  # estados antiguos, anteriores a los relojes por ion
            logger.warning("El estado de PP_reset no trae 'reloj_iones': los relojes de los iones empiezan en 0.")
            reloj = np.zeros(previo.oxygen_state.shape, dtype=np.int32)
        self.recombinacion = ProcesoRecombinacion(
            previo.oxygen_state,
            RelojesIones(reloj, previo.v_prev_reloj),
            oxygen_config={float(sim_ctes.voltaje_gen_oxigeno_sp): int(sim_ctes.num_oxigenos_sp_reset)},
            velocity_thresholds={float(sim_ctes.voltaje_gen_oxigeno_sp): sim_ctes.velocidad_iones_sp},
            filas_permitidas=Recombination.filas_interfaz_filamento(self.actual_state.shape[0], self.centros, sim_ctes.grosor_filamento),
            max_oxigenos=max_oxigenos,
            rng=self.rng,
        )
        self._informar_oxigeno()

    def _exigir_filamentos_destruidos(self) -> None:
        """En el ciclo completo PP_reset aborta si no los destruye; al arrancar con --start-from sp_reset se comprueba aquí."""
        topo = self.cache_topologia.analizar(self.actual_state)
        sin_destruir = []
        for i, (existe, destruido) in enumerate(zip(topo.existe, self.CF_destruido), start=1):
            if existe:
                sin_destruir.append(f"filamento {i}: sigue formado en el estado de partida")
            elif not destruido:
                sin_destruir.append(f"filamento {i}: PP_reset no lo registró como destruido")
        if sin_destruir:
            logger.error("SP_reset necesita todos los filamentos destruidos: " + "; ".join(sin_destruir))
            raise exceptions.FilamentosNoDestruidosException(
                simulation_path=self.rutas["simulation_path"],
                figures_path=self.rutas["figures_path"],
                voltage=self.voltage if self.voltage is not None else 0.0,
                num_simulation=self.num_simulation,
                actual_state=self.actual_state,
                CF_destruidos=self.N - len(sin_destruir),
                CF_esperados=self.N,
            )

    def rama_percolante(self, ctx: ContextoPaso, topo: TopologiaFilamentos) -> None:
        raise RuntimeError("SP_reset: el sistema ha vuelto a percolar, pero la recombinación solo puede quitar vacantes.")

    def rama_no_percolante(self, ctx: ContextoPaso) -> None:
        # La corriente se calcula con la temperatura del paso anterior, que debe ser escalar (ValueError si es un mapa)
        self.joule_sin_percolar(ctx, self.termico.actual.como_escalar())


def PP_reset(
    final_state_sp_set: EstadoFase,
    num_simulation: int,
    CF_ranges: List[tuple],
    protocolo: ProtocoloVoltaje,
    num_pasos_guardar_estado: int,
    usar_muro: bool = True,
    results_dir: str = "Results",
) -> EstadoFase:
    """
    Primera parte del RESET (ver `EtapaPPReset`), desde el estado de `SP_set`. Devuelve el de `SP_reset`.

    Lanza `NullResistanceException` o `FilamentosNoDestruidosException`.
    """
    return EtapaPPReset(
        final_state_sp_set, num_simulation, CF_ranges, protocolo, num_pasos_guardar_estado,
        usar_muro=usar_muro, results_dir=results_dir,
    ).ejecutar()


def SP_reset(
    final_state_pp_reset: EstadoFase,
    num_simulation: int,
    CF_ranges: List[tuple],
    protocolo: ProtocoloVoltaje,
    num_pasos_guardar_estado: int,
    results_dir: str = "Results",
) -> EstadoFase:
    """
    Segunda parte del RESET (ver `EtapaSPReset`), desde el estado de `PP_reset`. Devuelve el estado final del ciclo.

    Lanza `FilamentosNoDestruidosException` al arrancar si queda algún filamento sin destruir.
    """
    return EtapaSPReset(
        final_state_pp_reset, num_simulation, CF_ranges, protocolo, num_pasos_guardar_estado, results_dir=results_dir
    ).ejecutar()
