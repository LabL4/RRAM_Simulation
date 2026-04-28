"""
DEPRECATED: este script se mantiene como shim por compatibilidad con
invocaciones antiguas (notebooks que llaman `python RRAM_mod_simulation.py N`).

La lógica vive ahora en el paquete `RRAM`. Equivalencia:

    python RRAM_mod_simulation.py <num> <guardar_datos> <num_filamentos>
        ↓ (equivale a)
    python -m RRAM all <num> [--guardar-datos] [--num-filamentos <n>]

Migra los notebooks a `python -m RRAM all <num>` cuando puedas.
"""

import sys

from RRAM.__main__ import main as _main


def _to_argv(argv: list[str]) -> list[str]:
    """Traduce los argumentos posicionales antiguos al nuevo CLI."""
    if not argv:
        # Default histórico
        return ["all", "1", "--num-filamentos", "4"]

    num = argv[0]
    new = ["all", num]
    if len(argv) > 1 and argv[1] == "True":
        new.append("--guardar-datos")
    if len(argv) > 2:
        new += ["--num-filamentos", argv[2]]
    return new


if __name__ == "__main__":
    sys.exit(_main(_to_argv(sys.argv[1:])))
