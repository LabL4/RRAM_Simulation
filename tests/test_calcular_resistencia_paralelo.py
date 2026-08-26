"""
Ejemplo pedagógico: cómo testear `calcular_resistencia_paralelo`.

Ejecutar (requiere pytest instalado, ver README de esta sección en CLAUDE.md):
    pytest tests/test_calcular_resistencia_paralelo.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from RRAM import CurrentSolver  # noqa: E402


def test_dos_resistencias_iguales_dan_la_mitad():
    # Arrange
    resistencias = [2.0, 2.0]

    # Act
    resultado = CurrentSolver.calcular_resistencia_paralelo(resistencias)

    # Assert
    assert resultado == 1.0


def test_una_sola_resistencia_devuelve_ella_misma():
    resistencias = [5.0]

    resultado = CurrentSolver.calcular_resistencia_paralelo(resistencias)

    assert resultado == 5.0


def test_ningun_filamento_formado_da_resistencia_infinita():
    # Todos los filamentos son inf (no formados) → rama abierta total,
    # tal y como documenta OmhCurrent_filamentos.
    resistencias = [np.inf, np.inf, np.inf]

    resultado = CurrentSolver.calcular_resistencia_paralelo(resistencias)

    assert resultado == np.inf


def test_lista_vacia_da_resistencia_infinita():
    # Caso límite: sin filamentos que aportar, el comportamiento definido
    # es el mismo que "todos inf": rama abierta.
    resultado = CurrentSolver.calcular_resistencia_paralelo([])

    assert resultado == np.inf


def test_mezcla_de_finitos_e_infinitos_ignora_los_infinitos():
    # Un filamento no formado (inf) no debe afectar al resultado de los
    # que sí conducen: es como si no estuviera.
    resistencias = [4.0, np.inf, 4.0]

    resultado = CurrentSolver.calcular_resistencia_paralelo(resistencias)

    assert resultado == 2.0


def test_resistencia_cero_lanza_zero_division_error():
    # R=0 no es un caso físico esperado, pero documentamos qué hace
    # el código HOY si llega a pasar: no lo silencia, revienta.
    with pytest.raises(ZeroDivisionError):
        CurrentSolver.calcular_resistencia_paralelo([0.0, 4.0])
