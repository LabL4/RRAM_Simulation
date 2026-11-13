"""
Verificación de la generación de iones de oxígeno restringida a la banda de los filamentos
durante el reset: `Recombination.filas_interfaz_filamento`, `filas_libres_interfaz` y
`generate_oxygen`.

Ejecutar desde la raíz del repo:
    python3 tests/test_generacion_oxigeno_banda.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from RRAM import Recombination  # noqa: E402

fallos = []


def check(nombre, cond, detalle=""):
    print(f"{'  OK  ' if cond else ' FALLO'} | {nombre}" + (f"  ({detalle})" if detalle else ""))
    if not cond:
        fallos.append(nombre)


NY, NX = 40, 10
SIN_LIMITE = NY * NX  # cupo que nunca limita (el límite se prueba en test_limite_oxigenos_reset.py)

# --- 1. Banda: grosor común ---
banda = Recombination.filas_interfaz_filamento(NY, [10], 3)
check("grosor común: filas 7..13", np.array_equal(np.where(banda)[0], np.arange(7, 14)), str(np.where(banda)[0]))

# --- 2. Banda: grosor por filamento ---
banda2 = Recombination.filas_interfaz_filamento(NY, [5, 30], [1, 2])
esperado = np.r_[4:7, 28:33]
check("grosor por filamento: 4..6 y 28..32", np.array_equal(np.where(banda2)[0], esperado), str(np.where(banda2)[0]))

# --- 3. Banda recortada en los bordes ---
banda3 = Recombination.filas_interfaz_filamento(NY, [1, 38], 3)
check("recorte en bordes: 0..4 y 35..39", np.array_equal(np.where(banda3)[0], np.r_[0:5, 35:40]), str(np.where(banda3)[0]))

# --- 4. Número de grosores distinto al de centros ---
try:
    Recombination.filas_interfaz_filamento(NY, [5, 30], [1])
    check("grosores != centros lanza ValueError", False)
except ValueError:
    check("grosores != centros lanza ValueError", True)

# --- 5. Filas libres: dentro de la banda y sin ion ---
oxy = np.zeros((NY, NX), dtype=np.int8)
oxy[8, 0] = 1  # ocupada dentro de la banda
oxy[20, 0] = 1  # ocupada fuera de la banda (irrelevante)
libres = Recombination.filas_libres_interfaz(oxy, banda)
check("filas libres excluyen ocupadas y fuera de banda", np.array_equal(libres, [7, 9, 10, 11, 12, 13]), str(libres))

# --- 6. Muchas generaciones: nunca un ion fuera de la banda ---
np.random.seed(0)
oxy = np.zeros((NY, NX), dtype=np.int8)
for _ in range(500):
    oxy[:, 0] = 0  # se vacía la interfaz para seguir sorteando
    oxy = Recombination.generate_oxygen(oxy, 5, Recombination.filas_libres_interfaz(oxy, banda2), SIN_LIMITE)
    if np.any(oxy[~banda2, 0]):
        break
check("500 generaciones: ningún ion fuera de la banda", not np.any(oxy[~banda2, 0]))

# Todas las filas de la banda llegan a usarse alguna vez
usadas = np.zeros(NY, dtype=bool)
np.random.seed(1)
for _ in range(500):
    oxy = np.zeros((NY, NX), dtype=np.int8)
    oxy = Recombination.generate_oxygen(oxy, 5, Recombination.filas_libres_interfaz(oxy, banda2), SIN_LIMITE)
    usadas |= oxy[:, 0] == 1
check("todas las filas de la banda se usan", np.array_equal(usadas, banda2), str(np.where(usadas)[0]))

# --- 7. Solo columna 0 ---
check("solo se generan en la columna 0", not np.any(oxy[:, 1:]))

# --- 8. Banda llena: no se genera ninguno ---
np.random.seed(2)
oxy = np.zeros((NY, NX), dtype=np.int8)
oxy[banda, 0] = 1
antes = oxy.copy()
for _ in range(50):
    oxy = Recombination.generate_oxygen(oxy, 5, Recombination.filas_libres_interfaz(oxy, banda), SIN_LIMITE)
check("banda llena: no se genera ninguno", np.array_equal(oxy, antes), f"iones={int(oxy.sum())}")

# --- 9. Pocas plazas: se llena la banda sin desbordarla ---
np.random.seed(3)
oxy = np.zeros((NY, NX), dtype=np.int8)
for _ in range(200):
    oxy = Recombination.generate_oxygen(oxy, 5, Recombination.filas_libres_interfaz(oxy, banda), SIN_LIMITE)
check("pocas plazas: interfaz = banda exacta", np.array_equal(oxy[:, 0] == 1, banda), f"iones={int(oxy.sum())}")

# --- 10. max_num_oxygen = 0 no genera ---
oxy = np.zeros((NY, NX), dtype=np.int8)
oxy = Recombination.generate_oxygen(oxy, 0, Recombination.filas_libres_interfaz(oxy, banda), SIN_LIMITE)
check("max_num_oxygen = 0 no genera", oxy.sum() == 0)

print()
if fallos:
    print(f"{len(fallos)} FALLO(S): {fallos}")
    sys.exit(1)
print("Todos los checks OK")
