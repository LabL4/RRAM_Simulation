"""
Procesos estocásticos que cambian el estado de vacantes al final de cada paso.

- `ProcesoGeneracion` (SET): genera vacantes hasta un cupo.
- `ProcesoRecombinacion` (RESET): genera iones de oxígeno, los mueve y los recombina con vacantes.

Cada proceso guarda su propia configuración y su propio estado (cupo alcanzado; iones, relojes y
campo de probabilidad), de modo que el bucle de la etapa solo le pide "actualiza el estado".
La física vive en `Generation` y `Recombination`; aquí están también las funciones que la aplican
al estado (`update_state_*`) y las semillas por fase (`semilla_de_fase`, `crear_rng`).
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np

from . import Generation, Recombination
from .acumulador_tiempo import RelojesIones
from .constants_simulation import SimulationConstants
from .estado_fase import FASES
from .parameters import SimulationParameters

logger = logging.getLogger(__name__)


class ProcesoGeneracion:
    """Generación de vacantes con cupo máximo. Cuando el cupo se alcanza deja de sortear."""

    def __init__(
        self,
        factor_vecinos: float,
        factor_libre: float,
        max_vacantes: int,
        neighbor_mode: str,
        mascara: Optional[np.ndarray] = None,
        aviso_cupo: str = "Se ha alcanzado la ocupación máxima",
        rng=None,
    ):
        self.rng = rng
        self.factor_vecinos = factor_vecinos
        self.factor_libre = factor_libre
        self.max_vacantes = max_vacantes
        self.neighbor_mode = neighbor_mode
        #: Máscara que limita dónde se generan vacantes (None = toda la malla).
        self.mascara = mascara
        self.aviso_cupo = aviso_cupo
        #: True desde el primer paso en que no se sortea por estar el cupo lleno.
        self.cupo_alcanzado = False

    def actualizar(self, actual_state, total_vacantes, params, sim_ctes, E_filas, temperatura, k: int, tiempo_efectivo: float):
        """Devuelve (estado, probabilidades). Con el cupo lleno no sortea: (estado sin tocar, None)."""
        if total_vacantes < self.max_vacantes:
            return update_state_generation(
                actual_state,
                params,
                sim_ctes,
                E_filas,
                temperatura,
                self.factor_vecinos,
                self.factor_libre,
                self.max_vacantes,
                neighbor_mode=self.neighbor_mode,
                custom_mask=self.mascara,
                num_iteracion=k,
                tiempo_efectivo=tiempo_efectivo,
                rng=self.rng,
            )
        if not self.cupo_alcanzado:
            logger.info(f"{self.aviso_cupo} en el paso {k}.")
            self.cupo_alcanzado = True
        return actual_state, None


class ProcesoRecombinacion:
    """Iones de oxígeno del RESET: generación por voltaje, movimiento con reloj propio y recombinación."""

    def __init__(
        self,
        oxygen_state: np.ndarray,
        relojes: RelojesIones,
        oxygen_config: dict,
        velocity_thresholds: dict,
        filas_permitidas: np.ndarray,
        max_oxigenos: int,
        rng=None,
    ):
        self.rng = rng
        # Copia: la generación y el movimiento escriben sobre el array, y el de la etapa anterior
        # (EstadoFase.oxygen_state) no debe cambiar.
        self.oxygen_state = np.array(oxygen_state)
        self.relojes = relojes
        #: {voltaje umbral: máximo de oxígenos por paso}. Manda el umbral superado de mayor |V|,
        #: se escriban en el orden y con el signo que se escriban.
        self.oxygen_config = oxygen_config
        #: {voltaje umbral: velocidad de los iones [m/s]}.
        self.velocity_thresholds = velocity_thresholds
        #: Filas de la interfaz donde se pueden generar oxígenos.
        self.filas_permitidas = filas_permitidas
        #: Máximo de iones simultáneos.
        self.max_oxigenos = max_oxigenos
        self._campo_probabilidad = None

    def empezar_paso(self, voltage: float) -> None:
        """Avanza el reloj de los iones que ya existen (antes de generar: los nuevos empiezan en 0)."""
        self.relojes.actualizar(voltage, self.oxygen_state)

    def actualizar(self, actual_state, voltage, E_filas, temperatura, params, sim_ctes, tiempo_efectivo: float):
        """Genera, mueve y recombina. Devuelve el estado de vacantes; los iones quedan en `oxygen_state`."""
        actual_state, self.oxygen_state, self._campo_probabilidad = update_state_recombinate(
            voltage=voltage,
            E_field_filas=E_filas,
            oxygen_config=self.oxygen_config,
            sim_ctes=sim_ctes,
            params=params,
            actual_state=actual_state,
            oxygen_state=self.oxygen_state,
            temperatura=temperatura,
            velocity_thresholds=self.velocity_thresholds,
            relojes=self.relojes,
            filas_permitidas=self.filas_permitidas,
            max_oxigenos=self.max_oxigenos,
            tiempo_efectivo=tiempo_efectivo,
            rng=self.rng,
        )
        return actual_state

    def probabilidad(self):
        """Campo de probabilidad de recombinación del último paso (None si aún no hubo ninguno)."""
        return evaluar_campo_probabilidad(self._campo_probabilidad)


# ---------------------------------------------------------------------------
# Actualización del estado (generación y recombinación)
# ---------------------------------------------------------------------------


def update_state_generation(
    state: np.ndarray,
    params: SimulationParameters,
    sim_ctes: SimulationConstants,
    E_field: np.ndarray | float,
    temperatura: np.ndarray | float,
    factor_vecinos: float,
    factor_sin_vecinos: float,
    max_vacantes_permitidas: int,
    neighbor_mode: str = "both",
    custom_mask: np.ndarray | None = None,
    num_iteracion=None,
    tiempo_efectivo: float | None = None,
    rng=None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Orquesta el proceso de generación de vacantes asegurando que no se supere
    el límite máximo establecido. Si las candidatas superan el cupo, se eligen
    al azar entre ellas (todas con la misma oportunidad).

    Devuelve (estado, probabilidades).
    """

    act_state = state.copy()
    n_filas, n_cols = state.shape

    # 1. Ajuste del Campo Eléctrico
    if isinstance(E_field, np.ndarray) and E_field.ndim == 1:
        E_field_matrix = np.repeat(E_field[:, np.newaxis], n_cols, axis=1)
    else:
        E_field_matrix = E_field

    # 2. Ajuste de la matriz de Temperatura
    if isinstance(temperatura, np.ndarray) and temperatura.shape != state.shape:
        temp_matrix = temperatura[:, 1:-1]
    else:
        temp_matrix = temperatura

    # 3. Cálculo de la matriz de probabilidades
    prob_final = Generation.get_generation_probabilities_matrix(
        state=state,
        paso_temporal=params.paso_temporal if tiempo_efectivo is None else tiempo_efectivo,
        Electric_field=E_field_matrix,
        temperatura=temp_matrix,
        factor_vecinos=factor_vecinos,
        factor_sin_vecinos=factor_sin_vecinos,
        vibration_frequency=sim_ctes.vibration_frequency,
        generation_energy=sim_ctes.generation_energy,
        cte_red=sim_ctes.cte_red,
        gamma=sim_ctes.gamma,
        neighbor_mode=neighbor_mode,
        custom_mask=custom_mask,
    )

    # 4. Evaluación de la estocástica (cuáles "intentan" generarse)
    # TODO para que todas las simulaciones sean iguales, eliminar dps
    aleatorios = (np.random if rng is None else rng).rand(n_filas, n_cols)
    nueva_vacante = aleatorios < prob_final

    # 5. LIMITACIÓN INTELIGENTE DE VACANTES
    vacantes_actuales = np.sum(act_state)
    num_nuevas = np.sum(nueva_vacante)
    vacantes_disponibles = int(max_vacantes_permitidas - vacantes_actuales)

    if vacantes_disponibles <= 0:
        # Ya estamos en el límite, no se permite generar ninguna más
        return act_state, prob_final

    if num_nuevas > vacantes_disponibles:
        # Se ha superado el límite: se eligen al azar `vacantes_disponibles` candidatas.
        # Todas pasaron ya el sorteo `aleatorios < prob_final`, así que la probabilidad
        # ya está tenida en cuenta; ordenar por ella la aplicaría dos veces y, con
        # empates (p = 1), la elección la decidiría el orden interno de argsort.
        logger.warning(
            f"Paso {num_iteracion}: {num_nuevas} vacantes candidatas superan el cupo "
            f"({vacantes_disponibles} disponibles de {max_vacantes_permitidas}); "
            f"se eligen {vacantes_disponibles} al azar entre las candidatas."
        )

        # Obtenemos las coordenadas (i,j) de las candidatas
        coords = np.argwhere(nueva_vacante)

        # Elección al azar sin repetición (con el `rng` de la fase)
        elegidas = coords[(np.random if rng is None else rng).choice(len(coords), vacantes_disponibles, replace=False)]

        # Actualizamos la matriz solo con las elegidas
        act_state[elegidas[:, 0], elegidas[:, 1]] = 1
    else:
        # Si no se ha superado el límite, se aprueban todas
        act_state[nueva_vacante] = 1

    return act_state, prob_final


def update_state_recombinate(
    voltage: float,
    E_field_filas: np.ndarray,
    oxygen_config: dict,
    sim_ctes: SimulationConstants,
    params: SimulationParameters,
    actual_state: np.ndarray,
    oxygen_state: np.ndarray,
    temperatura: float | np.ndarray,
    velocity_thresholds: dict,  # Diccionario pasado como argumento
    relojes: RelojesIones,
    filas_permitidas: np.ndarray,
    max_oxigenos: int,
    tiempo_efectivo: float | None = None,
    rng=None,
) -> tuple[np.ndarray, np.ndarray, Callable[[], np.ndarray]]:
    """
    Orquesta el proceso completo de RESET en un paso de tiempo.

    `filas_permitidas` (Ny,) son las filas de la interfaz donde se pueden generar
    oxígenos (banda de los filamentos, ver Recombination.filas_interfaz_filamento).
    `max_oxigenos` es el máximo de iones simultáneos (ver Recombination.resolver_limite_oxigenos).
    `tiempo_efectivo` multiplica las tasas de recombinación (si no se pasa, vale `paso_temporal`);
    el exponente exp(-dt·v/L) usa siempre `paso_temporal`.
    `relojes` lleva el reloj propio de cada ion: el llamador ya lo ha avanzado este paso
    (`relojes.actualizar`, antes de generar); aquí se mueve con él y, tras recombinar, se
    borra el reloj de los iones desaparecidos (`relojes.limpiar`).
    Devuelve (vacantes, oxígenos, campo_probabilidad):
      - campo_probabilidad es una FUNCIÓN sin argumentos que devuelve el CAMPO de probabilidad de
        recombinación de este paso en toda la matriz (misma forma que actual_state), con el mismo
        tiempo, velocidad y temperatura que el sorteo real; solo se usa de verdad en las celdas con
        ion y vacante. Se devuelve sin evaluar porque solo se necesita al guardar el estado (cada
        `pasos_guardar_estado`) y evaluarlo en todos los pasos cuesta exponenciales sobre toda la
        matriz. Se evalúa con `evaluar_campo_probabilidad`.

    1. Generación de iones oxígenos por voltaje (solo en `filas_permitidas`).
    2. Movimiento de iones.
    3. Recombinación.
    """

    # 1. Se generan oxígenos según el voltaje. Los umbrales se recorren de mayor a menor |V|
    # (mismo criterio que la velocidad en Recombination.move_oxygen_ions).
    for threshold_voltage, max_oxigenos_paso in sorted(oxygen_config.items(), key=lambda par: abs(par[0]), reverse=True):
        if abs(voltage) > abs(threshold_voltage):
            filas_candidatas = Recombination.filas_libres_interfaz(oxygen_state, filas_permitidas)
            cupo = Recombination.cupo_oxigenos(oxygen_state, max_oxigenos)
            oxygen_state = Recombination.generate_oxygen(oxygen_state, max_oxigenos_paso, filas_candidatas, cupo, rng)
            break  # Solo usar el umbral más alto superado

    # 2. Movemos los ionesx
    oxygen_state, velocidad = Recombination.move_oxygen_ions(
        reloj_iones=relojes.reloj,
        oxygen_state=oxygen_state,
        temperature=temperatura,
        E_field_filas=E_field_filas,
        grid_size=params.atom_size,
        paso_temporal=params.paso_temporal,
        vibration_frequency=sim_ctes.vibration_frequency,
        gamma_drift=sim_ctes.gamma_drift,
        migration_energy=sim_ctes.E_m,
        cte_red=sim_ctes.cte_red,
        voltage=voltage,
        velocity_thresholds=velocity_thresholds,
    )

    # print(f"Velocidad de los iones oxígeno en este paso: {velocidad:.4e} m/s")

    # 3.Recombinación de iones con vacantes. t_ef solo multiplica la tasa (ν·t); el exponente
    # exp(-dt·v/L) usa siempre el dt del paso (ver Recombination.Prob_Recombination).
    tiempo_tasa = params.paso_temporal if tiempo_efectivo is None else tiempo_efectivo
    actual_state_update, oxygen_state_update = Recombination.Recombine_opt(
        vacancy_state=actual_state,
        oxygen_state=oxygen_state,
        tiempo_tasa=tiempo_tasa,
        paso_temporal=params.paso_temporal,
        velocidad=velocidad,
        temperatura=temperatura,
        vibration_frequency=sim_ctes.vibration_frequency,
        recom_enchancement_factor=sim_ctes.recom_enchancement_factor,
        recombination_energy=sim_ctes.recombination_energy,
        long_decaimiento_concentracion=sim_ctes.long_decaimiento_concentracion,
        rng=rng,
    )
    relojes.limpiar(oxygen_state_update)  # iones recombinados: su reloj a 0

    # 4. Campo de probabilidad de recombinación (para guardarlo, como la matriz de generación).
    # Se devuelve diferido: captura los valores de ESTE paso y solo se calcula si alguien lo pide.
    forma = actual_state.shape

    def campo_probabilidad() -> np.ndarray:
        prob = np.minimum(
            Recombination.Prob_Recombination(
                tiempo_tasa=tiempo_tasa,
                paso_temporal=params.paso_temporal,
                velocidad=velocidad,
                temp=temperatura,
                vibration_frequency=sim_ctes.vibration_frequency,
                recom_enchancement_factor=sim_ctes.recom_enchancement_factor,
                recombination_energy=sim_ctes.recombination_energy,
                long_decaimiento_concentracion=sim_ctes.long_decaimiento_concentracion,
            ),
            1.0,
        )
        return np.broadcast_to(prob, forma).copy()

    return actual_state_update, oxygen_state_update, campo_probabilidad


def evaluar_campo_probabilidad(campo: Callable[[], np.ndarray] | None) -> np.ndarray | None:
    """Evalúa el campo diferido de `update_state_recombinate`; None si aún no hubo ningún paso."""
    return None if campo is None else campo()


# ---------------------------------------------------------------------------
# Semillas y generadores aleatorios por fase
# ---------------------------------------------------------------------------


def semilla_de_fase(seed: int, fase: str) -> int:
    """
    Semilla propia de cada fase, derivada de la semilla de la corrida.

    Con la misma semilla en las cuatro fases todas consumían la misma secuencia U(0,1) y sus
    sorteos quedaban correlacionados. `SeedSequence.spawn` da una semilla estadísticamente
    independiente por fase y sigue siendo determinista: una fase arrancada con `--start-from`
    recibe exactamente la misma que en el ciclo completo.
    """
    hijos = np.random.SeedSequence(seed).spawn(len(FASES))
    return int(hijos[FASES.index(fase)].generate_state(1)[0])


def crear_rng(params, fase: str) -> np.random.RandomState:
    """
    Generador de números aleatorios propio de una fase.

    Con `params.seed` fijado, la secuencia es determinista y distinta en cada fase (ver
    `semilla_de_fase`), también con --start-from/--stop-at. Sin semilla es aleatorio real.
    Es un `RandomState`, el generador legado que usaba `np.random` de forma global, así que una
    semilla da la misma secuencia que antes; ya no se toca el estado global de numpy.
    """
    if params.seed is None:
        return np.random.RandomState()
    return np.random.RandomState(semilla_de_fase(params.seed, fase))
