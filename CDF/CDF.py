from RRAM.Representate import config_ax, setup_paper_plt, config_ax_IV
from statsmodels.distributions.empirical_distribution import ECDF
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Ahora importa el módulo

# Configuración de la figura
setup_paper_plt(plt, latex=True, scaling=2.2)

ruta_all_data = Path.cwd() / "Datos_Experimentales" / "Medidas_Experimentales_RRAM"
ruta_set = Path.cwd() / "Datos_Experimentales" / "V_Set"

results_path_set = ruta_set / "V_set_experimental.txt"
results_path = ruta_set / "V_set_experimental.txt"
results_path = ruta_set / "V_set_experimental.txt"


# Leer la primera línea para contar columnas
with open(results_path_set, encoding="utf-8") as f:
    first_line = f.readline()
    ncols = len(first_line.strip().split())

# Determinar nombres según columnas (si suponemos siempre 1 columna de nombre + 2 o 4 voltajes)
assert ncols in (3, 5), (
    "Formato inesperado: el archivo debe tener 3 o 5 columnas por fila"
)
if ncols == 3:
    dtype = [("Archivo", "U50"), ("V_creacion_1", "f8"), ("V_creacion_2", "f8")]
    col_names = ["Archivo", "V_creacion_1", "V_creacion_2"]
elif ncols == 5:
    dtype = [
        ("Archivo", "U50"),
        ("V_creacion_1", "f8"),
        ("V_creacion_2", "f8"),
        ("V_creacion_3", "f8"),
        ("V_creacion_4", "f8"),
    ]
    col_names = [
        "Archivo",
        "V_creacion_1",
        "V_creacion_2",
        "V_creacion_3",
        "V_creacion_4",
    ]

# Cargar normalmente (sin forzar names explícitamente)
resultados_txt = np.genfromtxt(
    results_path, dtype=dtype, encoding="utf-8", names=col_names
)

# Crear el DataFrame
df_resultados_sim = pd.DataFrame(resultados_txt)

df_resultados_sim["Numero"] = (
    df_resultados_sim["Archivo"]
    .str.extract(r"log_simulacion_(\d+)", expand=False)
    .astype(int)
)

print(df_resultados_sim)
