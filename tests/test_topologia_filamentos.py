"""
Verificación de `RRAM.topologia_filamentos` (B1/B2 en PLAN_IMPLEMENTACION.md).

Compara contra un oráculo independiente escrito con conjuntos y búsqueda en anchura en Python puro,
que sigue literalmente las reglas del código original con networkx + A*:
percolación, componentes que tocan los dos electrodos, poda de hojas fuera de las columnas de
electrodo, existencia por contacto con el electrodo izquierdo y borrado de bandas no formadas.

Ejecutar desde la raíz del repo:
    python3 tests/test_topologia_filamentos.py
"""

import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from RRAM import Percolation  # noqa: E402
from RRAM.topologia_filamentos import CacheTopologia, analizar_topologia, percola  # noqa: E402


def oraculo(estado, cf_ranges):
    H, W = estado.shape
    nodos = {(i, j) for i in range(H) for j in range(W) if estado[i, j] >= 1}

    def vecinos(n, conjunto):
        i, j = n
        return [v for v in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)) if v in conjunto]

    vistos, validos = set(), set()
    for n in nodos:
        if n in vistos:
            continue
        comp, cola = {n}, deque([n])
        while cola:
            for v in vecinos(cola.popleft(), nodos):
                if v not in comp:
                    comp.add(v)
                    cola.append(v)
        vistos |= comp
        cols = {c for _, c in comp}
        if 0 in cols and W - 1 in cols:
            validos |= comp
    while True:
        hojas = {n for n in validos if len(vecinos(n, validos)) == 1 and n[1] not in (0, W - 1)}
        if not hojas:
            break
        validos -= hojas
    limpia = np.zeros((H, W), dtype=int)
    for i, j in validos:
        limpia[i, j] = 1
    existe = [any(limpia[a : b + 1, 0]) for a, b in cf_ranges]
    filamentos = limpia.copy()
    for (a, b), ok in zip(cf_ranges, existe):
        if not ok:
            filamentos[a : b + 1, :] = 0
    return bool(validos), limpia, existe, filamentos


def rangos(H, n):
    a = H // n
    return [(i * a, (i + 1) * a - 1 if i < n - 1 else H - 1) for i in range(n)]


def test_igual_al_oraculo_en_estados_aleatorios():
    rng = np.random.default_rng(42)
    n_percolan = 0
    for _ in range(400):
        H, W = int(rng.integers(3, 40)), int(rng.integers(2, 20))
        estado = (rng.random((H, W)) < rng.uniform(0.3, 0.75)).astype(np.int64)
        for nf in (1, 2, 3):
            if H < nf:
                continue
            r = rangos(H, nf)
            p, limpia, existe, fil = oraculo(estado, r)
            t = analizar_topologia(estado, r)
            assert t.percola == p == percola(estado) == Percolation.is_path(estado)
            assert np.array_equal(t.matriz_limpia, limpia) and t.existe == existe
            assert np.array_equal(t.matriz_filamentos, fil)
            n_percolan += p
    assert n_percolan > 100, "la muestra debe incluir muchos casos que percolan"


def test_poda_rama_muerta_y_conserva_contacto_con_electrodo():
    e = np.zeros((5, 6), dtype=int)
    e[2, :] = 1          # camino recto entre electrodos
    e[0:2, 3] = 1        # rama muerta hacia arriba: se poda
    e[3, 0] = 1          # contacto extra con el electrodo izquierdo: se conserva
    t = analizar_topologia(e, [(0, 4)])
    esperado = np.zeros((5, 6), dtype=int)
    esperado[2, :] = 1
    esperado[3, 0] = 1
    assert t.percola and np.array_equal(t.matriz_limpia, esperado)


def test_filamento_asignado_por_su_contacto_izquierdo():
    e = np.zeros((6, 4), dtype=int)
    e[1, 0:2] = 1
    e[1:5, 1] = 1
    e[4, 1:4] = 1        # entra por la banda 0 (fila 1) y sale por la banda 1 (fila 4)
    t = analizar_topologia(e, [(0, 2), (3, 5)])
    assert t.existe == [True, False]
    assert not t.matriz_filamentos[3:6].any(), "la banda sin filamento se borra entera"


def test_sin_percolacion():
    e = np.zeros((4, 5), dtype=int)
    e[:, 0:2] = 1
    t = analizar_topologia(e, [(0, 1), (2, 3)])
    assert not t.percola and t.existe == [False, False] and not t.matriz_filamentos.any()


def test_fila_de_electrodo_fuera_de_las_bandas_lanza_valueerror():
    e = np.ones((4, 3), dtype=int)
    try:
        analizar_topologia(e, [(0, 1)])
    except ValueError:
        return
    raise AssertionError("debía lanzar ValueError")


def test_matrices_de_solo_lectura():
    t = analizar_topologia(np.ones((3, 3), dtype=int), [(0, 2)])
    for m in (t.matriz_limpia, t.matriz_filamentos):
        try:
            m[0, 0] = 0
        except ValueError:
            continue
        raise AssertionError("la matriz compartida por la caché debe ser de solo lectura")


def test_cache_reutiliza_solo_si_el_estado_no_cambia():
    rng = np.random.default_rng(3)
    e = (rng.random((20, 10)) < 0.6).astype(np.int64)
    c = CacheTopologia(rangos(20, 2))
    a = c.analizar(e)
    assert c.analizar(e.copy()) is a and (c.calculos, c.reutilizados) == (1, 1)
    e2 = e.copy()
    e2[0, 0] = 1 - e2[0, 0]
    b = c.analizar(e2)
    assert c.calculos == 2 and np.array_equal(b.matriz_limpia, analizar_topologia(e2, rangos(20, 2)).matriz_limpia)
    e[0, 0] = 1 - e[0, 0]  # mutar el array original no debe engañar a la caché
    assert c.analizar(e) is b


if __name__ == "__main__":
    fallos = 0
    for nombre, f in list(globals().items()):
        if nombre.startswith("test_"):
            try:
                f()
                print("ok  ", nombre)
            except Exception as ex:  # noqa: BLE001
                fallos += 1
                print("FALLA", nombre, repr(ex))
    print("Todo OK" if fallos == 0 else f"{fallos} fallos")
    sys.exit(1 if fallos else 0)
