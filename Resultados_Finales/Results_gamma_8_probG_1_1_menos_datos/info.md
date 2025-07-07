Estas simulaciones se han hecho multiplicando la prob de generacion cerca de otra vacante por 1.1 durante el set. 

Los valores:
E_r: [0.9450000000000001, 0.9974999999999999, 1.1025, 1.1550000000000002]
E_a: [0.9450000000000001, 0.9974999999999999, 1.1025, 1.1550000000000002]

I_0 = [2e-3]
ohm_resistence = [1.5]#, 1.5, 1.75, 2.00, 2.25, 2.5, 2.75, 3.0] # 1.5 es el valor óptimo
gamma = [8] #9 #10, 11, 12]

E_a_base = 1.05
E_a = [0.9*E_a_base, 0.95*E_a_base, 1.05*E_a_base, 1.1*E_a_base]

factor_generacion = [1.2] # 1,1 1,2 funcion bn el bueno es 1.2
r_termica_no_percola = [10]
r_termica_percola = [500] #, 9e2, 7e2]

E_r_base = 1.05
E_r = [0.9*E_r_base, 0.95*E_r_base, 1.05*E_r_base, 1.1*E_r_base]

print("E_r:", E_r)
print("E_a:", E_a)