
def Simple(DDP: float, espesor: float) -> float:
    """
    Calcula el campo eléctrico en función de la diferencia de potencial y el espesor de la capa de óxido.
    Parameters:
    - DDP (float): Diferencia de potencial.
    - espesor (float): Espesor de la capa de óxido.
    Returns:
    - float: Campo eléctrico.
    """

    ElectricField = DDP / espesor
    return ElectricField
