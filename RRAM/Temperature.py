def Temperature_Joule(potencial: float, intensidad: float, T_0: float, r_termica: float) -> float:
    """
    Calcula el incremento de temperatura debido al calentamiento por efecto Joule.

    Esta función determina el aumento de temperatura en un dispositivo causado por
    la disipación de potencia eléctrica (efecto Joule), considerando la resistencia
    térmica del sistema.

    Args:
        potencial (float): Diferencia de potencial aplicada en Voltios [V]
        intensidad (float): Corriente eléctrica que circula en Amperios [A]
        T_0 (float): Temperatura inicial o ambiente en Kelvin [K] (parámetro no utilizado)
        r_termica (float): Resistencia térmica del dispositivo en K/W [K/W]

    Returns:
        float: Incremento de temperatura por efecto Joule en Kelvin [K]

    Fórmula:
        ΔT = |V * I| * R_th
        donde P = V * I es la potencia disipada y R_th es la resistencia térmica del sistema.
    """
    temperatura_disipacion = T_0 + abs(potencial * intensidad) * r_termica

    return temperatura_disipacion
