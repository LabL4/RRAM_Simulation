from RRAM import Constants as cte


def Temperature_Joule(potencial: float, intensidad: float, percola: bool, T_0: float = 300, **kwargs) -> float:

    # Obtengo los valores de las constantes si las estoy pasando como argumentos
    if kwargs:
        if percola:
            r_termica = float(kwargs.get('r_termica_percola'))
        else:
            r_termica = float(kwargs.get('r_termica_no_percola'))
    else:
        if percola:
            r_termica = cte.r_termica_percola
        else:
            r_termica = cte.r_termica_no_percola

    Temperature_Joule = T_0 + abs(potencial * intensidad) * r_termica

    return Temperature_Joule
