from scipy.integrate import quad
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
sns.set(font_scale=1, rc={'text.usetex': True,
                          'text.latex.preamble': r'\usepackage[russian]{babel}'})
plt.figure(figsize=(9, 7))

lambd = 7e-7
L = 1
D = 2 * np.sqrt(1 * lambd * L)
r = D/2
k = 2 * np.pi / lambd
E0 = 1
A = np.sqrt(1/(lambd * L))
a = 1


def real_func(X, x):
    return np.sin(k*L + k*X**2/(2*L)-k*x*X/L + np.pi/4)


def imag_func(X, x):
    return np.sin(k*L + k*X**2/(2*L)-k*x*X/L - np.pi/4)


integ= lambda x, f: quad(f, -r, r, x)[0]
d_of_max = lambd*L/D
x = 8*d_of_max + D
N = 2000
dx = 2*x/N
x_ = np.linspace(-x, x, N)

U_slit = []

for x in x_:
    U_slit.append(A * E0 * (1j * integ(x, imag_func) + integ(x, real_func)))

U_slit = np.array(U_slit)

origin_points_y = np.zeros(1000)

end_points_y = U_slit.imag
end_points_x = U_slit.real
plt.yticks([0], [])
plt.xlabel(r"$x, mm$")
plt.xlim(-2, 5)

plt.quiver(x_[1000::20]*1000, origin_points_y[::20],
           end_points_x[1000::20], end_points_y[1000::20], width=0.001)

plt.savefig('hole-vectors.png', dpi=800)
plt.show()
plt.quiver(origin_points_y[::15], origin_points_y[::15],
           end_points_x[::15], end_points_y[::15], width=0.001)
plt.show()

I_slit = np.abs(U_slit)**2

E0_ = np.zeros(N)

r_light = D
# r_light = D/2
M = min(np.round(r_light/dx), N/2)
E0_[int(N/2-M):int(N/2+M)] = E0
E0_ = E0_
En0 = E0**2 * D
En_slit = np.trapz(I_slit, dx=dx)

print(En_slit, En0, np.abs(En0-En_slit)/En_slit)


U_coil = -U_slit+np.exp(2j*np.pi*L/lambd)*E0_
I_coil = np.abs(U_coil)**2


plt.xlabel(r"$x, mm$")
plt.ylabel(r"$I/I_0$")
plt.axvline(x=-r*1000, linestyle='dashed', color='black', label=r'Границы щели')
plt.axvline(x=r*1000, linestyle='dashed', color='black')

I_slit_norm = I_slit/E0**2
I_coil_norm = I_coil/E0**2
plt.plot(x_*1000, I_slit_norm, color='black')
plt.legend()
plt.savefig('hole-graph.png', dpi=800)

plt.show()

plt.figure(figsize=(9, 7))
plt.xlabel(r"$x, mm$")
plt.ylabel(r"$I/I_0$")

plt.axvline(x=-r*1000, linestyle='dashed', color='black', label=r'Границы проволоки')
plt.axvline(x=r*1000, linestyle='dashed', color='black')

plt.plot(x_*1000, I_coil_norm, color='black')
plt.legend()
plt.savefig('coil-graph.png', dpi=800)

plt.show()


En0 = E0**2 * (2*r_light)
En_coil = np.trapz(I_coil, dx=dx)
print(En_coil, En0, np.abs(En0-En_coil)/En_coil)

I_max = max(max(I_slit), max(I_coil))
I_matrix_slit = np.resize(I_slit/I_max, (20, N))
I_matrix_coil = np.resize(I_coil/I_max, (20, N))
plt.figure(figsize=(11, 9))
sns.heatmap(I_matrix_coil,
            cmap=sns.color_palette(palette='blend:black,red',
                                   as_cmap=True),
            vmax=1,
            vmin=0,
            yticklabels=False,
            xticklabels=False,
            cbar=False)
plt.savefig('hole-hm-red.png', dpi=800)
plt.show()
plt.figure(figsize=(11, 9))
sns.heatmap(I_matrix_slit,
            cmap=sns.color_palette(palette='blend:black,red',
                                   as_cmap=True),
            vmax=1,
            vmin=0,
            yticklabels=False,
            xticklabels=False,
            cbar=False)
plt.savefig('coil-hm-red.png', dpi=800)
plt.show()
