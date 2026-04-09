from RRAM import Constants as cte


def Temperature_Joule(potencial: float, intensidad: float, percola: bool, T_0: float = 315, **kwargs) -> float:

    # Obtengo los valores de las constantes si las estoy pasando como argumentos
    if kwargs:
        if percola:
            r_termica = float(kwargs.get('r_termica_percola')) # type: ignore
        else:
            r_termica = float(kwargs.get('r_termica_no_percola')) # type: ignore
    else:
        if percola:
            r_termica = cte.r_termica_percola
        else:
            r_termica = cte.r_termica_no_percola

    Temperature_Joule = T_0 + abs(potencial * intensidad) * r_termica
    
    # print(f"Resistencia termica: {r_termica}")
    # print(f"voltaje: {abs(potencial)}")
    # print(f"EL sistema percola: {percola}")
    # print(f"Temperatura base: {T_0}")
    # print(f"La intensidad es: {intensidad:.5f}")
    # print(f"La temperatura es: {Temperature_Joule:.5f}\n")

    return Temperature_Joule
