Estas simulaciones se han hecho multiplicando la prob de generacion cerca de otra vacante por 1.1 durante el set. Los valores de E_a = [0.945, 0.9975, 1.029, 1.05, 1.071, 1.1025, 1.155] y los mismos para E_r, se ha comprobado que los toma bn con prints en el lugar donde emplea estos coeficientes.

I_0 = [2e-3]
ohm_resistence = [1.5]#, 1.5, 1.75, 2.00, 2.25, 2.5, 2.75, 3.0] # 1.5 es el valor óptimo
gamma = [8] #9 #10, 11, 12]

E_a_base = 1.05
# E_a = [0.9*E_a_base, 0.95*E_a_base, 0.98*E_a_base, E_a_base, 1.02*E_a_base, 1.05*E_a_base, 1.1*E_a_base]
E_a = [0.945, 0.9975, 1.029, 1.05, 1.071, 1.1025, 1.155]

factor_generacion = [1.2] # 1,1 1,2 funcion bn el bueno es 1.2
r_termica_no_percola = [10]
r_termica_percola = [500] #, 9e2, 7e2]

# E_r_base = 1.05
# E_r = [0.9*E_r_base, 0.95*E_r_base, 0.98*E_r_base, E_r_base, 1.02*E_r_base, 1.05*E_r_base, 1.1*E_r_base]
E_r = [0.945, 0.9975, 1.029, 1.05, 1.071, 1.1025, 1.155]