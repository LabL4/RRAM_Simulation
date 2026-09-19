"""
Verificación de `RRAM.voltage_controller.VoltageController`.

Ejecutar desde la raíz del repo:
    python3 tests/test_voltage_controller.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from RRAM.voltage_controller import Medidas, VoltageController, parsear_segmentos  # noqa: E402

PASO = 1.1 / 10000  # paso_potencial_set típico (voltaje_final_set / num_pasos)

fallos = []


def check(nombre, cond, detalle=""):
    print(f"{'  OK  ' if cond else ' FALLO'} | {nombre}" + (f"  ({detalle})" if detalle else ""))
    if not cond:
        fallos.append(nombre)


def medidas(k, I=0.0, V_ant=0.0, vac=0, fils=0, percola=False):
    return Medidas(I_total=I, V_anterior=V_ant, n_vacantes=vac, n_filamentos=fils, percola=percola, k=k)


# ---------------------------------------------------------------- 1
print("\n[1] Modo legacy: reproduce exactamente np.arange (vector_ddp histórico)")
N = 12000
esperado = np.arange(0.000, 1.4 + PASO, PASO)  # mismo vector que PP_set construía
ctrl = VoltageController(segmentos=None, paso_potencial=PASO)
obtenido = np.array([ctrl.next(medidas(k)) for k in range(N)])
check("legacy es bit a bit idéntico a np.arange", np.array_equal(obtenido, esperado[:N]))
check("legacy nunca termina por sí mismo", not ctrl.terminado)
check("legacy siempre en rampa", ctrl.en_rampa())

# ---------------------------------------------------------------- 2
print("\n[2] Compliance de corriente: rampa hasta I >= umbral, luego V constante")
ctrl = VoltageController(
    segmentos=[("rampa", {"hasta_I": 1e-4}), ("constante", {})],
    paso_potencial=PASO,
)
K_CRUCE = 500  # a partir de este paso la corriente medida supera el umbral
vs = []
for k in range(1000):
    I = 2e-4 if k >= K_CRUCE else 1e-6
    vs.append(ctrl.next(medidas(k, I=I, V_ant=vs[-1] if vs else 0.0)))
vs = np.array(vs)
# La transición se evalúa en el paso siguiente al primero con I >= umbral
v_congelado = vs[K_CRUCE]
check("antes del cruce la rampa es lineal", np.allclose(np.diff(vs[:K_CRUCE]), PASO))
check("tras el cruce el voltaje queda constante", np.all(vs[K_CRUCE + 1 :] == v_congelado), f"V={v_congelado:.5f}")
check("el segmento activo ya no es rampa", not ctrl.en_rampa())
est = ctrl.estado()
check("se registró 1 transición por hasta_I", len(est["transiciones"]) == 1 and est["transiciones"][0]["condicion"] == "hasta_I")
# La medida con I >= umbral se pasa en k=K_CRUCE, así que la transición se
# registra en ese mismo paso y el V congelado es el del paso anterior.
check("la transición registra k y V coherentes", est["transiciones"][0]["k"] == K_CRUCE and abs(est["transiciones"][0]["V"] - v_congelado) < 1e-12)

# ---------------------------------------------------------------- 3
print("\n[3] Resto de condiciones de disparo")
casos = {
    "hasta_V": dict(config={"hasta_V": 0.5}, kwargs=lambda k, v_ant: dict(V_ant=v_ant)),
    "hasta_vacantes": dict(config={"hasta_vacantes": 300}, kwargs=lambda k, v_ant: dict(vac=k)),
    "hasta_filamentos": dict(config={"hasta_filamentos": 2}, kwargs=lambda k, v_ant: dict(fils=2 if k >= 100 else 1)),
    "hasta_percolacion": dict(config={"hasta_percolacion": True}, kwargs=lambda k, v_ant: dict(percola=k >= 100)),
}
for nombre, caso in casos.items():
    ctrl = VoltageController(segmentos=[("rampa", caso["config"]), ("constante", {})], paso_potencial=PASO)
    v_ant = 0.0
    disparo = None
    for k in range(10000):
        v = ctrl.next(medidas(k, **caso["kwargs"](k, v_ant)))
        if disparo is None and not ctrl.en_rampa():
            disparo = k
        v_ant = v
    est = ctrl.estado()
    ok = len(est["transiciones"]) == 1 and est["transiciones"][0]["condicion"] == nombre
    check(f"disparo por {nombre}", ok, f"k={est['transiciones'][0]['k'] if est['transiciones'] else '—'}")

# ---------------------------------------------------------------- 4
print("\n[4] Segmento constante con 'pasos': terminado al agotarlos")
ctrl = VoltageController(
    segmentos=[("rampa", {"hasta_V": 0.3}), ("constante", {"pasos": 50})],
    paso_potencial=PASO,
)
v_ant = 0.0
k_fin = None
for k in range(10000):
    v = ctrl.next(medidas(k, V_ant=v_ant))
    v_ant = v
    if ctrl.terminado:
        k_fin = k
        break
check("terminado se activa", ctrl.terminado)
k_transicion = ctrl.estado()["transiciones"][0]["k"]
check("dura exactamente 'pasos' pasos en constante", k_fin is not None and k_fin - k_transicion == 50, f"transición k={k_transicion}, fin k={k_fin}")

# ---------------------------------------------------------------- 5
print("\n[5] Constante con V explícito")
ctrl = VoltageController(segmentos=[("rampa", {"hasta_V": 0.2}), ("constante", {"V": 0.15})], paso_potencial=PASO)
v_ant = 0.0
for k in range(5000):
    v = ctrl.next(medidas(k, V_ant=v_ant))
    v_ant = v
check("mantiene el V explícito", not ctrl.en_rampa() and v_ant == 0.15)

# ---------------------------------------------------------------- 6
print("\n[6] estado() es JSON-serializable")
try:
    json.dumps(ctrl.estado())
    check("json.dumps(estado()) no falla", True)
except TypeError as e:
    check("json.dumps(estado()) no falla", False, str(e))

# ---------------------------------------------------------------- 7
print("\n[7] Validación de configuración")
for raw, debe_fallar in [
    (None, False),
    ([("rampa", {"hasta_I": 1e-4})], False),
    ([["rampa", {"hasta_I": 1e-4}], ["constante", {}]], False),  # listas en vez de tuplas (CSV)
    ([("triangular", {})], True),
    ([("rampa", {"hasta_X": 1})], True),
    ("rampa", True),
]:
    try:
        parsear_segmentos(raw)
        fallo = False
    except ValueError:
        fallo = True
    check(f"parsear_segmentos({raw!r})", fallo == debe_fallar)

# ----------------------------------------------------------------
print(f"\n{'=' * 60}")
if fallos:
    print(f"RESULTADO: {len(fallos)} fallo(s): {fallos}")
    sys.exit(1)
print("RESULTADO: todos los checks OK")
