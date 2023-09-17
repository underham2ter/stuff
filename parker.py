from scipy.stats import norm, t
import numpy as np
import pandas as pd
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1, rc={'text.usetex' : True})

sns.set_palette(sns.color_palette("hls", 8))

R = 8.31
g = 10
T_c = np.exp(-3/4)*g/(2*R)

def equation(u, r, T):
    u_c2 = R * T
    r_c = 2 * R * T / g
    return (u**2/u_c2) - np.log(u**2/u_c2) - 4*np.log(r) - 4*r_c/r

T_values = [0, 0.05, 0.15, 0.35, 0.65] + T_c
n = len(T_values)
cmap = plt.get_cmap('rainbow')
norm = plt.Normalize(0, n - 1)


rainbow_colors = [cmap(norm(i)) for i in range(n)]

r_values = np.linspace(0.02, 5, 1000)
u_values = np.linspace(0.02, 5, 1000)
r_grid, u_grid = np.meshgrid(r_values, u_values)


plt.figure(figsize=(10, 6))
plt.title(r'Solutions to the Equation for different Temperatures')


for i in range(len(T_values)):#
    T = T_values[i]
    u_c2 = R * T
    r_c = 2 * R * T / g

    r_values = np.linspace(0.15, 5, 1000)*r_c
    u_values = np.linspace(0.01, 4, 1000)*np.sqrt(u_c2)
    r_grid, u_grid = np.meshgrid(r_values, u_values)

    #equation_values = (u_grid**2 / (u_c**2 * T)) - np.log(u_grid**2 / (u_c**2 * T)) - 4 * np.log(r_grid) - (4 * r_c / r_grid)
    equation_values = equation(u_grid, r_grid, T)

    contours = plt.contour(r_grid/r_c, u_grid/np.sqrt(u_c2), equation_values, levels=0, colors=rainbow_colors[i])

plt.xlabel(r'$r/r_0$')
plt.xticks([1, 2, 3, 4, 5])
plt.yticks([0, 1, 2, 3, 4])
plt.ylabel(r'$v/v_0$')

plt.savefig('parker2.png', dpi=800)