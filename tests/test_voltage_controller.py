"""
Verificación de `RRAM.voltage_controller` (ProtocoloVoltaje y VoltageController).

Ejecutar desde la raíz del repo:
    python3 tests/test_voltage_controller.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from RRAM.voltage_controller import (  # noqa: E402
    CondicionExigidaNoCumplida,
    Medidas,
    ProtocoloVoltaje,
    VoltageController,
    constante,
    parsear_segmentos,
    rampa,
    tren_pulsos,
)

DS = 1.1 / 1000
DR = 1.4 / 1000

fallos = []


def check(nombre, cond, detalle=""):
    print(f"{'  OK  ' if cond else ' FALLO'} | {nombre}" + (f"  ({detalle})" if detalle else ""))
    if not cond:
        fallos.append(nombre)


def medidas(k, I=0.0, vac=0, fils=0, percola=False):
    return Medidas(I_total=I, n_vacantes=vac, n_filamentos=fils, percola=percola, k=k)


def recorrer(c, medida=None):
    """Voltajes que produce el controlador hasta `terminado` (como el bucle de una fase)."""
    vs = []
    for k in range(c.presupuesto() + 1):
        v = c.next(medida(k) if medida else medidas(k))
        if c.terminado:
            break
        vs.append(v)
    return np.array(vs)


def igual(vs, esperado):
    return len(vs) == len(esperado) and np.allclose(vs, esperado)


def falla(fn):
    try:
        fn()
        return False
    except (ValueError, CondicionExigidaNoCumplida):
        return True


# ---------------------------------------------------------------- 1
print("\n[1] Rampa: misma aritmética que np.arange, extremos incluidos")
c = VoltageController("pp_set", [rampa(0.0, 1.1, DS)])
vs = recorrer(c)
esperado = np.array([0.0 + k * DS for k in range(1001)])
check("rampa 0 → 1.1 V bit a bit igual a base + k·dV", np.array_equal(vs, esperado))
check("1001 puntos y presupuesto exacto", len(vs) == 1001 == c.presupuesto(), f"{len(vs)}, {c.presupuesto()}")
check("termina por 'hasta'", c.completada and c.fin_por == "hasta")

vs = recorrer(VoltageController("pp_reset", [rampa(0.0, -1.4, DR)]))
check("rampa decreciente 0 → -1.4 V (dV positivo)", abs(vs[0]) < 1e-15 and abs(vs[-1] + 1.4) < 1e-12 and np.all(np.diff(vs) < 0))

vs = recorrer(VoltageController("pp_set", [rampa(0.0, 1.0, 0.3)]))
check("si dV no divide el tramo, el último punto se recorta a 'hasta'", igual(vs, [0, 0.3, 0.6, 0.9, 1.0]), f"{vs}")

# ---------------------------------------------------------------- 2
print("\n[2] 'anterior'")
c = VoltageController("sp_set", [rampa("anterior", 0.0, 0.25)], v_previo=1.0)
vs = recorrer(c)
check("rampa 'anterior' continúa un dV más allá (no repite el punto)", igual(vs, [0.75, 0.5, 0.25, 0.0]), f"{vs}")
check("se registra el valor resuelto", c.anterior_resuelto[0]["V"] == 1.0)
check("el enganche lo marca como continuación", c.estado()["enganche"]["continua_anterior"])

c = VoltageController("pp_set", [rampa(0.0, 1.0, 0.1, hasta_I=1e-4), constante("anterior", 3)])
vs = recorrer(c, lambda k: medidas(k, I=2e-4 if k >= 5 else 0.0))
check("compliance: meseta en el voltaje congelado", igual(vs, [0, 0.1, 0.2, 0.3, 0.4, 0.4, 0.4, 0.4]), f"{vs}")
check("transición por hasta_I", c.transiciones[0].condicion == "hasta_I")

c = VoltageController("sp_reset", [rampa("anterior", 0.0, 0.5), constante(0.0, 2)], v_previo=0.0)
vs = recorrer(c)
check("rampa 'anterior' ya en su 'hasta' no produce puntos", igual(vs, [0.0, 0.0]), f"{vs}")

# ---------------------------------------------------------------- 3
print("\n[3] Mesetas y trenes de pulsos")
c = VoltageController("pp_set", tren_pulsos(1.1, 3, 0.0, 2, 2))
vs = recorrer(c)
check("tren de pulsos exacto", igual(vs, [1.1, 1.1, 1.1, 0, 0, 1.1, 1.1, 1.1, 0, 0]), f"{vs}")
check("una meseta a 1.1 V no corta la etapa", c.fin_por == "pasos" and len(vs) == 10)
check("presupuesto = suma de pasos", c.presupuesto() == 10)

# ---------------------------------------------------------------- 4
print("\n[4] Condiciones y 'exigir'")
c = VoltageController("pp_set", [rampa(0.0, 1.0, 0.1, hasta_percolacion=True, exigir=True), constante("anterior", 2)])
vs = recorrer(c, lambda k: medidas(k, percola=k >= 4))
check("exigir cumplido: sigue al siguiente segmento", igual(vs, [0, 0.1, 0.2, 0.3, 0.3, 0.3]), f"{vs}")

c = VoltageController("pp_set", [rampa(0.0, 0.3, 0.1, hasta_percolacion=True, exigir=True), constante("anterior", 2)])
check("exigir no cumplido: aborta la etapa", falla(lambda: recorrer(c)))

c = VoltageController("pp_set", [rampa(0.0, 0.3, 0.1, hasta_percolacion=True), constante("anterior", 2)])
vs = recorrer(c)
check("sin exigir: pasa al siguiente segmento en 'hasta'", igual(vs, [0, 0.1, 0.2, 0.3, 0.3, 0.3]), f"{vs}")

c = VoltageController("pp_set", [rampa(0.0, 1.0, 0.1, pasos=3), constante("anterior", 1)])
vs = recorrer(c)
check("rampa con 'pasos' corta a los N puntos", igual(vs, [0, 0.1, 0.2, 0.2]), f"{vs}")

# ---------------------------------------------------------------- 5
print("\n[5] Validación")
malos = [
    [("rampa", {"desde": 0.0, "hasta": 1.0})],  # falta dV
    [("rampa", {"desde": 0.0, "hasta": 1.0, "dV": -0.1})],  # dV negativo
    [("rampa", {"desde": 0.0, "hasta": 1.0, "dV": 0})],
    [("rampa", {"desde": 0.5, "hasta": 0.5, "dV": 0.1})],  # desde == hasta
    [("rampa", {"desde": 0.0, "hasta": "anterior", "dV": 0.1})],
    [("rampa", {"desde": 0.0, "hasta": 1.0, "dV": 0.1, "V": 1.0})],  # V en rampa
    [("constante", {"V": 1.0})],  # falta pasos
    [("constante", {"V": 1.0, "pasos": 0})],
    [("constante", {"V": 1.0, "pasos": 5, "dV": 0.1})],
    [("constante", {"V": 1.0, "pasos": 5, "exigir": True})],  # exigir sin condición
    [("constante", {"V": 1.0, "pasos": 5, "hasta_percolacion": False})],
    [("rampa", {"desde": 0.0, "hasta": 1.0, "dV": 0.1, "hasta_V": 0.5})],  # clave eliminada
    [("triangular", {})],
    [],
]
for m in malos:
    check(f"se rechaza {m}", falla(lambda m=m: parsear_segmentos(m, "test")))

check("'anterior' sin etapa previa se rechaza", falla(lambda: VoltageController("sp_set", [rampa("anterior", 0.0, 0.1)])))

base = {"pp_set": [rampa(0.0, 1.1, DS)], "sp_set": [rampa("anterior", 0.0, DS)],
        "pp_reset": [rampa("anterior", -1.4, DR)], "sp_reset": [rampa("anterior", 0.0, DR)]}
check("protocolo sin una etapa se rechaza", falla(lambda: ProtocoloVoltaje.desde_config({k: v for k, v in base.items() if k != "sp_reset"})))
check("protocolo con etapa desconocida se rechaza", falla(lambda: ProtocoloVoltaje.desde_config({**base, "pp_sett": base["pp_set"]})))
check("pp_set no puede empezar en 'anterior'", falla(lambda: ProtocoloVoltaje.desde_config({**base, "pp_set": [constante("anterior", 5)]})))

# ---------------------------------------------------------------- 6
print("\n[6] Protocolo: texto del CSV, enganches y recorrido")
p = ProtocoloVoltaje.desde_config(base)
q = ProtocoloVoltaje.desde_config(p.a_texto())
check("a_texto → desde_config conserva la configuración", q.configuracion() == p.configuracion())

rec = p.recorrido()
check("pp_set → sp_set continúa sin repetir el pico", abs(rec["sp_set"]["V"][0] - (1.1 - DS)) < 1e-12)
check("pp_reset arranca un paso por debajo de 0", abs(rec["pp_reset"]["V"][0] + DR) < 1e-12)
check("sp_reset termina en 0 V", abs(rec["sp_reset"]["V"][-1]) < 1e-12)

pulsos = ProtocoloVoltaje.desde_config({**base, "pp_set": tren_pulsos(1.1, 3, 0.0, 3, 2), "sp_set": [rampa(1.1, 0.0, DS)]})
e = pulsos.recorrido()["sp_set"]["enganche"]
check("un salto entre etapas queda registrado", abs(e["salto"] - 1.1) < 1e-12 and not e["continua_anterior"], f"{e}")
check("el resumen marca el salto", "SALTO" in pulsos.resumen())

c = p.controlador("pp_set")
recorrer(c)
check("controlador(sp_set) toma V_fin del estado previo", abs(p.controlador("sp_set", previo={"voltaje": c.estado()}).v_previo - 1.1) < 1e-12)
check("informe() incluye configuración y ejecución", set(p.informe()) == {"configuracion", "ejecucion"} and "pp_set" in p.informe()["ejecucion"])

# ----------------------------------------------------------------
print(f"\n{'=' * 60}")
if fallos:
    print(f"RESULTADO: {len(fallos)} fallo(s): {fallos}")
    sys.exit(1)
print("RESULTADO: todos los checks OK")
