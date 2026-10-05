"""
Verificación de los puntos marcados de la curva I-V y del registro de eventos.

  1. procesar_filamentos_creados / _destruidos guardan el paso k y la etapa.
  2. localizar_puntos_iv: cada tipo de localización, y en particular que un
     punto por evento cae en su fila exacta aunque ese voltaje se repita.
  3. validar_puntos_iv rechaza tablas mal escritas.

Ejecutar desde la raíz del repo:
    python3 tests/test_puntos_iv.py
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from RRAM.filament_tracking import procesar_filamentos_creados, procesar_filamentos_destruidos  # noqa: E402
from RRAM.iv_analysis import PUNTOS_IV, localizar_puntos_iv, validar_puntos_iv  # noqa: E402

fallos = []


def check(nombre, cond, detalle=""):
    print(f"{'  OK  ' if cond else ' FALLO'} | {nombre}" + (f"  ({detalle})" if detalle else ""))
    if not cond:
        fallos.append(nombre)


def falla(fn):
    try:
        fn()
        return False
    except ValueError:
        return True


def etapa(V, I):
    """Datos de una etapa con el formato de Data_*.npz: columnas t, V, I."""
    V = np.asarray(V, dtype=float)
    return np.column_stack([np.arange(len(V)) * 1e-3, V, np.asarray(I, dtype=float)])


# ---------------------------------------------------------------- 1
print("\n[1] Registro de eventos: paso k y etapa")
with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    estado = np.zeros((4, 4), dtype=int)

    creaciones = {}
    CF_creado = np.array([False, False])
    procesar_filamentos_creados(tmp, tmp, [False, True], CF_creado, 0.61, np.zeros(2), estado, 1, k=5412, creaciones_dict=creaciones, etapa="pp_set")
    check("la creación guarda k, etapa, filamento y voltaje", creaciones == {0: {"filamento": 2, "voltaje": 0.61, "etapa": "pp_set", "k": 5412}}, f"{creaciones}")

    roturas = {}
    CF_destruido = np.array([False, False])
    V_rotura = np.zeros(2)
    procesar_filamentos_destruidos(tmp, tmp, [False, True], CF_destruido, -1.18, V_rotura, estado, 1, k=830, roturas_dict=roturas, etapa="pp_reset")
    procesar_filamentos_destruidos(tmp, tmp, [False, False], CF_destruido, -0.3, V_rotura, estado, 1, k=12, roturas_dict=roturas, etapa="sp_reset")
    check("cada rotura guarda su k y su etapa", roturas[0]["k"] == 830 and roturas[0]["etapa"] == "pp_reset" and roturas[1]["k"] == 12 and roturas[1]["etapa"] == "sp_reset", f"{roturas}")
    check("una rotura ya registrada no se repite", len(roturas) == 2)

# ---------------------------------------------------------------- 2
print("\n[2] localizar_puntos_iv")
# PP_set con tres pulsos a 1.1 V: el voltaje 1.1 aparece en las filas 0, 3 y 6.
# La percolación ocurre en la fila 6 (tercer pulso), con otra corriente.
pp_set = etapa([1.1, 1.1, 0.0, 1.1, 1.1, 0.0, 1.1, 1.1, 0.0], [1e-4, 1e-4, 1e-9, 2e-4, 2e-4, 1e-9, 5e-3, 6e-3, 1e-9])
pp_reset = etapa([-0.2, -0.4, -0.6, -0.8], [1e-2, 1e-2, 1e-3, 1e-4])
sp_reset = etapa([-0.8, -0.4, 0.0], [1e-4, 1e-5, 1e-9])
sin_filtrar = {"pp_set": pp_set, "sp_set": etapa([], []), "pp_reset": pp_reset, "sp_reset": sp_reset}
filtrado = {k: v[np.abs(v[:, 2]) >= 1e-7] if v.size else v for k, v in sin_filtrar.items()}
roturas = {0: {"filamento": 1, "voltaje": -0.6, "etapa": "pp_reset", "k": 2}, 1: {"filamento": 2, "voltaje": -0.4, "etapa": "sp_reset", "k": 1}}
creaciones = {0: {"filamento": 1, "voltaje": 0.0, "etapa": "pp_set", "k": 2}}


def localizar(tabla, paso_percolacion=6):
    return localizar_puntos_iv(validar_puntos_iv(tabla), filtrado, sin_filtrar, paso_percolacion, creaciones, roturas, 1e-7)


p = localizar({"b": {"etapa": "pp_set", "en": ("evento", "percolacion"), "desplazamiento": (0, 1)}})
check("percolación: fila exacta aunque el voltaje se repita", p.get("b") == (1.1, 5e-3), f"{p}")

p = localizar({"x": {"etapa": "pp_set", "en": ("voltaje", 1.1), "desplazamiento": (0, 1)}})
check("por voltaje: se queda con la PRIMERA aparición (contraste)", p.get("x") == (1.1, 1e-4), f"{p}")

p = localizar({"e": {"etapa": "pp_reset", "en": ("evento", "rotura_0"), "desplazamiento": (0, 1)}})
check("rotura_0 en su fila de pp_reset", p.get("e") == (-0.6, 1e-3), f"{p}")

p = localizar({"e": {"etapa": "pp_reset", "en": ("evento", "rotura_1"), "desplazamiento": (0, 1)}})
check("un evento de otra etapa se omite", "e" not in p, f"{p}")

p = localizar({"e": {"etapa": "sp_reset", "en": ("evento", "rotura_1"), "desplazamiento": (0, 1)}})
check("rotura_1 en su fila de sp_reset", p.get("e") == (-0.4, 1e-5), f"{p}")

p = localizar({"z": {"etapa": "pp_set", "en": ("evento", "creacion_0"), "desplazamiento": (0, 1)}})
check("un evento cuya fila es ruido se omite", "z" not in p, f"{p}")

p = localizar({"e": {"etapa": "pp_reset", "en": ("evento", "rotura_7"), "desplazamiento": (0, 1)}})
check("un evento que no ocurrió se omite", "e" not in p, f"{p}")

p = localizar({"b": {"etapa": "pp_set", "en": ("evento", "percolacion"), "desplazamiento": (0, 1)}}, paso_percolacion=None)
check("sin percolación, el punto b se omite", "b" not in p, f"{p}")

p = localizar({"c": {"etapa": "pp_set", "en": ("fin_etapa",), "desplazamiento": (0, 1)}})
check("fin_etapa: último punto de la curva que se dibuja", p.get("c") == (1.1, 6e-3), f"{p}")

p = localizar({"s": {"etapa": "sp_set", "en": ("fin_etapa",), "desplazamiento": (0, 1)}})
check("una etapa sin datos se omite", "s" not in p, f"{p}")

p = localizar(PUNTOS_IV)
check("la tabla por defecto se valida y localiza sin error", set(p) >= {"b", "c", "e"}, f"{sorted(p)}")

# ---------------------------------------------------------------- 3
print("\n[3] validar_puntos_iv")
bueno = {"etapa": "pp_set", "en": ["voltaje", 0.5], "desplazamiento": [0.0, 1.0]}
check("acepta listas de JSON y las pasa a tuplas", validar_puntos_iv({"a": bueno})["a"]["en"] == ("voltaje", 0.5))
malos = [
    {},
    {"a": {**bueno, "etapa": "pp_sett"}},
    {"a": {**bueno, "en": ["evento", "rotura"]}},
    {"a": {**bueno, "en": ["evento", "percolar"]}},
    {"a": {**bueno, "en": ["voltaje"]}},
    {"a": {**bueno, "en": ["fin_etapa", 1.0]}},
    {"a": {**bueno, "desplazamiento": [0.0]}},
    {"a": {"etapa": "pp_set", "en": ["voltaje", 0.5]}},
    {"a": {**bueno, "extra": 1}},
]
for m in malos:
    check(f"se rechaza {m}", falla(lambda m=m: validar_puntos_iv(m)))

# ----------------------------------------------------------------
print(f"\n{'=' * 60}")
if fallos:
    print(f"RESULTADO: {len(fallos)} fallo(s): {fallos}")
    sys.exit(1)
print("RESULTADO: todos los checks OK")
