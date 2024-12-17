from RRAM import Constants as cte


def Temperature_Joule(potencial: float, intensidad: float, T_0: float = 350, **kwargs) -> float:

    # Obtengo los valores de las constantes si las estoy pasando como argumentos
    if kwargs:
        # Obtengo el valor de las constantes que necesita la función
        r_termica = float(kwargs.get('r_termica'))
    else:
        r_termica = cte.r_termica

    Temperature_Joule = T_0 + abs(potencial * intensidad) * r_termica

    return Temperature_Joule
