from dataclasses import dataclass
import statistics
import timeit


@dataclass
class Params:
    device_size: float


def test_dataclass():
    params = Params(1.5)
    for _ in range(100000):
        x = params.device_size * 2


def test_direct():
    device_size = 1.5
    for _ in range(100000):
        x = device_size * 2


# Medir tiempo
n_tests = 1000
tiempo_dataclass = timeit.repeat(test_dataclass, number=n_tests)
tiempo_direct = timeit.repeat(test_direct, number=n_tests)

print(f"Tiempo promedio dataclass: {statistics.mean(tiempo_dataclass) * 1000:.3f} ms")
print(f"Tiempo promedio directo: {statistics.mean(tiempo_direct) * 1000:.3f} ms")
print(
    f"Diferencia: {(statistics.mean(tiempo_dataclass) - statistics.mean(tiempo_direct)) * 1000:.3f} ms"
)
