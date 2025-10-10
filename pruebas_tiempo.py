import numpy as np
import pickle
import gzip

# Parámetros
num_matrices = 40000  # Puedes poner 40000 o el número que necesites
dimension = 40

# Generar matrices binarias aleatorias (0 o 1)
matrices = [
    np.random.randint(0, 2, size=(dimension, dimension), dtype=np.uint8)
    for _ in range(num_matrices)
]

print(matrices[0])  # Imprime la matriz completa con ceros y unos

# Convertir a booleano y empaquetar bits para reducir tamaño (8 valores en 1 byte)
matrices_packed = [np.packbits(m.astype(np.bool_)) for m in matrices]

# Guardar con pickle + gzip comprimido
with gzip.open("matrices_bin.pkl.gz", "wb") as f:
    pickle.dump(matrices_packed, f)

print(
    f"Guardado de {num_matrices} matrices binarias empaquetadas comprimidas completado."
)

# Cargar de archivo
with gzip.open("matrices_bin.pkl.gz", "rb") as f:
    loaded_packed = pickle.load(f)

# Desempaquetar bits y remodelar a matriz original
matrices_restored = [
    np.unpackbits(arr)[: dimension * dimension].reshape((dimension, dimension))
    for arr in loaded_packed
]

# Verificación simple
print("Forma de la primera matriz restaurada:", matrices_restored[0].shape)

print(matrices_restored[120])  # Imprime la matriz completa con ceros y unos
print(np.unique(matrices_restored[120]))  # Debe imprimir solo [0 1]
