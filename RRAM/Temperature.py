

def Temperature_Joule(ddp: float, intensidad: float, R_termica: float = 5e5, T_0: float = 300):
    Temperature_Joule = T_0 + ddp * intensidad * R_termica
    return Temperature_Joule
