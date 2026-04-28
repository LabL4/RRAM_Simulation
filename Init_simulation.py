"""
DEPRECATED: este script se mantiene como shim por compatibilidad.

La lógica de generación de estados iniciales vive ahora en
`RRAM.init_simulation.build_initial_states`. Equivalencia:

    python Init_simulation.py
        ↓ (equivale a)
    python -m RRAM init
"""

import sys

from RRAM.__main__ import main as _main


if __name__ == "__main__":
    sys.exit(_main(["init"]))
