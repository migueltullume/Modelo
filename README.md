# MODELO 1
import numpy as np  
import matplotlib.pyplot as plt 

### Parámetros
N = 1.0; p = 0.3; q = 0.1; r = 0.2  

### Función biexponencial S(t)  
def S(t):  
    return (N/(p + r - q)) * ((p - q)*np.exp(-(p+r)*t) + r*np.exp(-q*t))  

### Simulación  
t = np.linspace(0, 10, 100)  
s_vals = [S(ti) for ti in t]  

### Visualización  
plt.plot(t, s_vals)  
plt.title("Decaimiento de Atención Colectiva")  
plt.savefig("decaimiento_biexponencial.png")  

# MODELO 2
import numpy as np

import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

### Parámetros del modelo (personalizables)
#### Exponente de apego preferencial
alpha = 1.2 
#### Atención mínima
f0 = 0.1  
#### Escala
N = 1.0 
#### Memoria comunicativa (rápida)
p = 0.4 
#### Memoria cultural (lenta)
q = 0.01  
#### Factor de acoplamiento
r = 0.15 
#### Máxima atención considerada
k_max = 50  
#### Rango temporal
t_span = [0, 10] 

### Función de decaimiento biexponencial S(t)
def S(t):
    return (N/(p + r - q)) * ((p - q)*np.exp(-(p+r)*t) + r*np.exp(-q*t))

### Función de apego preferencial f(k)
def f(k, alpha=alpha, f0=f0):
    return (k**alpha) + f0

### Ecuación maestra (sistema de EDOs)
def master_equation(t, n):
    dndt = np.zeros_like(n)
    lambda_vals = S(t) * f(np.arange(len(n)))
    
    for k in range(len(n)):
        # Término de pérdida
        dndt[k] -= lambda_vals[k] * n[k]
        
        # Término de ganancia (si k>0)
        if k > 0:
            dndt[k] += lambda_vals[k-1] * n[k-1]
            
    return dndt

### Condición inicial (todos comienzan con k=0)
n0 = np.zeros(k_max+1)
n0[0] = 1.0  # 100% de productos con k=0 en t=0

### Resolver numéricamente
solution = solve_ivp(master_equation, t_span, n0, t_eval=np.linspace(t_span[0], t_span[1], 100))

### Visualización
plt.figure(figsize=(12, 8))

#### 1. Evolución de S(t)
plt.subplot(2, 2, 1)
t_vals = np.linspace(t_span[0], t_span[1], 100)
plt.plot(t_vals, S(t_vals), 'r-', linewidth=2)
plt.title('Decaimiento Temporal $S(t)$')
plt.xlabel('Tiempo (t)')
plt.grid(alpha=0.3)

#### 2. Función de apego preferencial
plt.subplot(2, 2, 2)
k_vals = np.arange(0, k_max+1)
plt.plot(k_vals, f(k_vals), 'b-', linewidth=2)
plt.title('Apego Preferencial $f(k) = k^{{{}}} + {}$'.format(alpha, f0))
plt.xlabel('Atención Acumulada (k)')
plt.grid(alpha=0.3)

#### 3. Distribución n(k,t) en tiempos seleccionados
plt.subplot(2, 1, 2)
for t_idx in [0, 20, 40, 60, 80, 99]:
    t_val = solution.t[t_idx]
    plt.plot(k_vals, solution.y[:, t_idx], 
             label=f't={t_val:.1f}', 
             alpha=0.8)

plt.title('Evolución de la Distribución $n(k,t)$')
plt.xlabel('k (Atención Acumulada)')
plt.ylabel('Fracción de Productos')
plt.legend()
plt.yscale('log')
plt.grid(alpha=0.3)
plt.tight_layout()

#### Guardar resultados
plt.savefig('modelo_atencion_colectiva.png')
plt.show()
