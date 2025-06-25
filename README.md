# Modelo
import numpy as np  
import matplotlib.pyplot as plt 

## Parámetros
N = 1.0; p = 0.3; q = 0.1; r = 0.2  

# Función biexponencial S(t)  
def S(t):  
    return (N/(p + r - q)) * ((p - q)*np.exp(-(p+r)*t) + r*np.exp(-q*t))  

# Simulación  
t = np.linspace(0, 10, 100)  
s_vals = [S(ti) for ti in t]  

# Visualización  
plt.plot(t, s_vals)  
plt.title("Decaimiento de Atención Colectiva")  
plt.savefig("decaimiento_biexponencial.png")  
