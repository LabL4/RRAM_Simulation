"""
Estado térmico de una fase: la temperatura del paso, la del paso anterior y la de los electrodos.

Antes cada fase llevaba tres variables sueltas (`temperatura`, `temperatura_anterior`, `T_electrodo`)
cuyo tipo cambiaba según el modelo usado en el paso: un mapa FVM con columnas de electrodo
(Ny, Nx+2) o un escalar de Joule. Eso obligaba a comprobar
`isinstance` y a recortar `[:, 1:-1]` en muchos sitios. Aquí la temperatura del paso es siempre un
`CampoTemperatura`, que sabe qué es, y `EstadoTermico` aplica las reglas de amortiguación y del
electrodo que antes se resolvían en una función de `Temperature` que devolvía una tupla de 4.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from .thermal import amortiguacion, muro

MENSAJE_ANTERIOR_ESCALAR = (
    "La temperatura no se ha calculado como matriz, no se pueden extraer los perfiles de los filamentos. "
    "Se esperaba una matriz de temperaturas, pero se ha recibido un valor escalar."
)
MENSAJE_NO_ESCALAR = "La temperatura calculada no es un valor escalar, no se puede calcular la corriente de Poole-Frenkel."


@dataclass(frozen=True, eq=False)
class CampoTemperatura:
    """
    Temperatura de un paso. Una de dos:

    - `mapa_fvm`: mapa del solver térmico (Ny, Nx+2), con las columnas de electrodo.
    - `escalar`: temperatura única del modelo de Joule.
    """

    mapa_fvm: Optional[np.ndarray] = None
    escalar: Optional[float] = None

    @classmethod
    def fvm(cls, mapa: np.ndarray) -> "CampoTemperatura":
        return cls(mapa_fvm=mapa)

    @classmethod
    def joule(cls, T) -> "CampoTemperatura":
        return cls(escalar=T)

    @classmethod
    def desde_legado(cls, valor) -> "CampoTemperatura":
        """Desde lo que guardaban las fases en `Temperatura_final`: un mapa FVM o un escalar."""
        if isinstance(valor, np.ndarray) and valor.ndim == 2:
            return cls.fvm(valor)
        return cls.joule(valor)

    @property
    def es_escalar(self) -> bool:
        """Escalar de Joule."""
        return self.mapa_fvm is None

    @property
    def interior(self):
        """Temperatura en el marco de `actual_state` (sin electrodos): mapa (Ny, Nx) o escalar."""
        if self.mapa_fvm is not None:
            return self.mapa_fvm[:, 1:-1]
        return self.escalar

    def legado(self):
        """El valor que guardan y devuelven las fases: el mapa FVM o el escalar."""
        if self.mapa_fvm is not None:
            return self.mapa_fvm
        return self.escalar

    def como_escalar(self):
        """El escalar de Joule; ValueError si es un mapa (no se puede usar en Poole-Frenkel)."""
        if not self.es_escalar:
            raise ValueError(MENSAJE_NO_ESCALAR)
        return self.escalar

    def maximo(self) -> float:
        return float(np.max(self.legado()))


class EstadoTermico:
    """
    Temperatura actual, temperatura del paso anterior (marco interior) y electrodo de una fase.

    `registrar_fvm` y `registrar_joule` aplican, en este orden, las mismas reglas que tenía
    aquella función, y devuelven la temperatura de cada filamento.
    """

    def __init__(self, actual: CampoTemperatura, anterior, T_electrodo, params, centros: Sequence, N: int):
        self.actual = actual
        #: Temperatura del paso anterior en el marco interior: mapa (Ny, Nx) o escalar.
        self.anterior = anterior
        #: Temperatura de los electrodos: escalar o perfil (Ny,).
        self.T_electrodo = T_electrodo
        self.params = params
        self.centros = list(centros)
        self.N = N

    @classmethod
    def inicial(cls, params, sim_ctes, centros, N) -> "EstadoTermico":
        """Arranque de PP_set: todo a la temperatura inicial; el electrodo, a su constante."""
        return cls(CampoTemperatura.joule(params.init_temp), params.init_temp, sim_ctes.Temperatura_electrodo, params, centros, N)

    @classmethod
    def heredado(cls, estado_previo, params, sim_ctes, centros, N) -> "EstadoTermico":
        """Arranque desde el `EstadoFase` de la fase anterior (estados antiguos sin perfil: la constante)."""
        actual = CampoTemperatura.desde_legado(estado_previo.temperatura_final)
        T_electrodo = estado_previo.perfil_electrodo
        if T_electrodo is None:
            T_electrodo = sim_ctes.Temperatura_electrodo
        return cls(actual, actual.interior, T_electrodo, params, centros, N)

    # ------------------------------------------------------------------ consultas

    def perfiles_anteriores(self) -> list:
        """Perfil (fila central) de cada filamento en la temperatura anterior. ValueError si era escalar."""
        if not isinstance(self.anterior, np.ndarray):
            raise ValueError(MENSAJE_ANTERIOR_ESCALAR)
        return muro.extraer_perfiles_filamentos(matriz_temperaturas=self.anterior, filas_centros=self.centros)

    def anterior_a_mapa_si_escalar(self, forma) -> None:
        """Al percolar por primera vez: la anterior pasa a ser un mapa lleno de la temperatura actual."""
        if self.actual.es_escalar:
            self.anterior = amortiguacion.como_mapa(self.actual.escalar, forma)

    # ------------------------------------------------------------------ registro de un paso

    def registrar_fvm(self, T_fvm: np.ndarray, sim_ctes, amortiguar: bool = True) -> List[float]:
        """
        Mapa FVM (Ny, Nx+2) recién resuelto:
            1. perfil de los electrodos para el paso siguiente, con el mapa SIN amortiguar
            2. amortigua el mapa contra la anterior (si `amortiguar`)
            3. la anterior pasa a ser el mapa sin columnas de electrodo
            4. T de cada filamento = máximo de su fila central (T_0 si no tiene centro)
        """
        T_0, dt = self.params.init_temp, self.params.paso_temporal
        if amortiguacion.amortiguacion_activa(sim_ctes.k_electrodo_enfriamiento):
            self.T_electrodo = amortiguacion.actualizar_temperatura_electrodo(
                T_fvm,
                self.T_electrodo,
                muro.filas_filamento(T_fvm.shape[0], self.centros, sim_ctes.grosor_filamento),
                T_0=T_0,
                k=sim_ctes.k_electrodo_enfriamiento,
                dt=dt,
                umbral=sim_ctes.umbral_caida_electrodo,
            )
        temperatura = T_fvm
        if amortiguar:
            temperatura = amortiguacion.amortiguar_mapa_fvm(
                temperatura, self.anterior, T_0=T_0, k=sim_ctes.k_amortiguacion_termica, dt=dt,
                umbral=sim_ctes.umbral_caida_temperatura,
            )
        self.actual = CampoTemperatura.fvm(temperatura)
        self.anterior = self.actual.interior
        perfiles = muro.extraer_perfiles_filamentos(matriz_temperaturas=self.anterior, filas_centros=self.centros)
        return [float(np.max(p)) if p is not None else T_0 for p in perfiles]

    def registrar_joule(
        self,
        T,
        sim_ctes,
        amortiguar: bool = True,
        actualizar_anterior: bool = True,
    ) -> List[float]:
        """
        Temperatura escalar de Joule:
            1. amortigua contra la anterior (si `amortiguar` y la anterior es escalar: en el paso en que
               se pasa del mapa al escalar no hay con qué comparar)
            2. la anterior pasa a ser T (si `actualizar_anterior`)
            3. el electrodo sigue a la temperatura de Joule
            4. T de cada filamento: [T] * N
        """
        temperatura = T
        if amortiguar and not isinstance(self.anterior, np.ndarray):
            temperatura = amortiguacion.amortiguar_enfriamiento(
                temperatura, self.anterior, T_0=self.params.init_temp, k=sim_ctes.k_amortiguacion_termica,
                dt=self.params.paso_temporal, umbral=sim_ctes.umbral_caida_temperatura,
            )
        if actualizar_anterior:
            self.anterior = temperatura
        self.T_electrodo = amortiguacion.temperatura_electrodo_joule(temperatura, self.T_electrodo, k=sim_ctes.k_electrodo_enfriamiento)
        self.actual = CampoTemperatura.joule(temperatura)
        return [temperatura] * self.N
