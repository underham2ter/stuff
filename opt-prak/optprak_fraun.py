import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import seaborn as sns
sns.set(font_scale=1, rc={'text.usetex': True,
                          'text.latex.preamble': r'\usepackage[russian]{babel}'})
plt.figure(figsize=(9, 7))

len2 = np.array([198, 259, 317, 373, 422]) * 4.5e-3
dlen2 = len2 - len2[0]
zer_y = 2 * 4.5e-3

len3 = np.array([296, 386, 479, 556, 635]) * 4.5e-3
dlen3 = len3 - len3[0]

dL = 11.2 - np.array([11.2, 8, 5, 2, 0])
dL *= 10

zer_x = 1


def fun(x, a):
    return a * x


def plot_error():

    popt, pcov = curve_fit(fun, dL, dlen2)
    a = popt[0]
    a_pcov = pcov[0][0]

    S_a = np.sqrt(a_pcov)
    print(S_a**2)
    print(sum((dlen2 - fun(dL, a))**2)/(4 * sum(dL**2)))
    print(sum((dlen2 - fun(dL, a))**2)/4)
    random_a_error = 2.8 * S_a  # P = 0.95, n = 5
    measurement_a_error = sum(dL) * (a * zer_x + zer_y)/sum(dL**2)
    full_error = random_a_error+measurement_a_error

    n = 4/a
    dn = 4*full_error/a**2

    plt.xlabel(r'$\Delta L,~~mm$')
    plt.ylabel(r'$\Delta w,~~mm$')
    plt.plot(dL, fun(dL, a + full_error), '--', color='r', linewidth=0.5, label='Границы ошибки')
    plt.plot(dL, fun(dL, a - full_error), '--', color='r', linewidth=0.5)
    plt.errorbar(dL,
                 dlen2,
                 xerr=zer_x,
                 yerr=zer_y,
                 fmt="none",
                 color="black",
                 label='Данные')

    plt.plot(dL, fun(dL, a), linewidth=0.8,
                 label='Приближение МНК')

    plt.legend()
    #plt.savefig('slit_graph_error.png', dpi=400)
    plt.show()

    zer_L_0 = np.sqrt((4.5e-3/a)**2 + (len2[0]*full_error/a**2)**2)
    L_0 = len2[0]/a

    return a, full_error, n, dn, L_0, zer_L_0

# plot_error()

def plot_error_2():
    popt, pcov = curve_fit(fun, dL, dlen3)
    a = popt[0]
    a_pcov = pcov[0][0]

    S_a = np.sqrt(a_pcov)
    random_a_error = 2.8 * S_a  # P = 0.95, n = 5
    measurement_a_error = sum(dL) * (a * zer_x + zer_y)/sum(dL**2)
    full_error = random_a_error+measurement_a_error

    n = 6/a
    dn = 6*full_error/a**2

    plt.plot(dL, fun(dL, a + full_error), '--', color='r', linewidth=0.5)
    plt.plot(dL, fun(dL, a - full_error), '--', color='r', linewidth=0.5)
    plt.errorbar(dL,
                 dlen3,
                 xerr=zer_x,
                 yerr=zer_y,
                 fmt="none",
                 color="black")

    plt.plot(dL, fun(dL, a), linewidth=0.8)

    plt.show()

    zer_L_0 = np.sqrt((4.5e-3/a)**2 + (len3[0]*full_error/a**2)**2)
    L_0 = len3[0]/a

    return a, full_error, n, dn, L_0, zer_L_0


# a, full_error, n, dn, L_0, zer_L_0 = plot_error()
# print(a, full_error, n, dn, L_0, zer_L_0)


def i(x):
    return (np.sin(np.pi*n*x*4.5e-3/L_0)/(np.pi*n*x*4.5e-3/L_0))**2


def coil_plot():
    zer_l = 0.05
    x = np.array([2.8, 4.4, 6.1])
    L = 121.2
    lmbd = 632.816
    m = np.array([2, 3, 4])
    m_big = np.array([1.5, 4.5])
    sin_theta = x/(2 * L)
    zer_theta = np.sqrt((zer_l/(2 * L))**2 + ((x*zer_l)/(2 * L**2))**2)
    popt, pcov = curve_fit(fun, m, sin_theta)
    a = popt[0]
    a_pcov = pcov[0][0]

    S_a = np.sqrt(a_pcov)
    random_a_error = 1.9 * S_a  # P = 0.8, n = 3
    measurement_a_error = sum(m*zer_theta)/sum(m**2)
    full_error = random_a_error+measurement_a_error
    plt.xlabel(r'$m$')
    plt.ylabel(r'$sin\Theta \sim x/L$')
    plt.xticks(m)
    plt.errorbar(m,
                 sin_theta,
                 yerr=zer_theta,
                 fmt=".",
                 color="black",
                 label='Данные')
    plt.plot(m_big, fun(m_big, a + full_error), '--', color='r', linewidth=0.5, label='Границы ошибки')
    plt.plot(m_big, fun(m_big, a - full_error), '--', color='r', linewidth=0.5)
    plt.plot(m_big, fun(m_big, a), '--', linewidth=0.8,
                 label='Приближение МНК')

    d = lmbd / a
    d_d = lmbd / a**2 * full_error
    print(a, full_error)
    print(d, d_d)
    # plt.legend()
    # plt.savefig('coil_graph_error.png', dpi=400)
    # plt.show()

# coil_plot()

# x = np.arange(-148, 149)
# plt.figure(figsize=(11, 9))
# plt.xticks([0, 29, 58, 87, 116, 145, 174, 203, 232, 261])
# plt.ylim(0, 0.1)
# plt.plot(x+148, i(x))
# plt.savefig('I_fraun', dpi=400)

def plot_grate():
    lmbd = 632.816

    m = np.array([-2, -1, 1, 2])
    x_1 = np.array([0, 96.5, 196.4, 286])
    x_1 -= (x_1[1] + x_1[2])/2
    x_2 = np.array([0, 66.4, 148.7, 204])
    x_2 -= (x_2[1] + x_2[2])/2
    L = 122.8

    zer_l = 0.05

    sin_theta_1 = x_1/np.sqrt(x_1**2 + L**2)
    sin_theta_2 = x_2/np.sqrt(x_2**2 + L**2)
    zer_theta_1 = np.sqrt((1/np.sqrt(x_1**2 + L**2) - x_1**2/(x_1**2 + L**2)**(3/2))**2 + (x_1*L/(x_1**2 + L**2)**(3/2))**2)*zer_l
    zer_theta_2 = np.sqrt((1/np.sqrt(x_2**2 + L**2) - x_2**2/(x_2**2 + L**2)**(3/2))**2 + (x_2*L/(x_1**2 + L**2)**(3/2))**2)*zer_l
    popt, pcov = curve_fit(fun, m, sin_theta_1)
    a_1 = popt[0]
    a_1_pcov = pcov[0][0]

    S_a = np.sqrt(a_1_pcov)

    print(S_a**2)
    print(sum((sin_theta_1 - fun(m, a_1))**2)/(3 * sum(m**2)))

    random_a_1_error = 1.6 * S_a  # P = 0.8, n = 4
    measurement_a_1_error = sum(m*zer_theta_1)/sum(m**2)
    full_error_1 = random_a_1_error+measurement_a_1_error

    popt, pcov = curve_fit(fun, m, sin_theta_2)
    a_2 = popt[0]
    a_2_pcov = pcov[0][0]

    S_a = np.sqrt(a_2_pcov)
    random_a_2_error = 1.6 * S_a  # P = 0.8, n = 4
    measurement_a_2_error = sum(m*zer_theta_2)/sum(m**2)
    print(measurement_a_1_error, random_a_1_error)
    full_error_2 = random_a_2_error+measurement_a_2_error

    plt.xticks([-2, -1, 0, 1, 2])
    plt.xlabel(r'$m$')
    plt.ylabel(r'$sin\Theta = x/\sqrt{x^2+L^2}$')
    plt.errorbar(m,
                 sin_theta_1,
                 yerr=zer_theta_1,
                 fmt=".",
                 color="red",
                 label='Данные для красного лазера')
    plt.plot(m, fun(m, a_1), '--', linewidth=0.8, color='orange')

    plt.errorbar(m,
                 sin_theta_2,
                 yerr=zer_theta_2,
                 fmt=".",
                 color="green",
                 label='Данные для зеленого лазера')
    plt.plot(m, fun(m, a_2), '--', linewidth=0.8, color='cyan')

    plt.legend()
    plt.savefig('grate_graph.png', dpi=400)
    plt.show()

    d = lmbd / a_1
    d_d = lmbd / a_1**2 * full_error_1

    N = 1e6 / d
    d_N = 1e6 / d**2 * d_d

    lmbd_2 = a_2 / a_1 * lmbd

    d_ldmd_2 = np.sqrt((full_error_2/a_1)**2 + (a_2*full_error_1/a_1**2)**2) * lmbd

    print(a_2, full_error_2, a_1, full_error_1)
    print(lmbd_2, d_ldmd_2)
    print(N, d_N)
    print(d, d_d)


plot_grate()

def i_approx():

    i = np.array([44.7, 97, 66, 11.7])/97
    x_3 = np.array([51, 108, 146.6, 184.5]) - 146.6
    L = 112.7
    lmbd = 532.7
    print()
    m = np.array([-2, -1, 0, 1])
    zer_l = 0.1

    sin_theta_3 = x_3 / np.sqrt(x_3 ** 2 + L ** 2)

    zer_theta_1 = np.sqrt((1 / np.sqrt(x_3 ** 2 + L ** 2) - x_3 ** 2 / (x_3 ** 2 + L ** 2) ** (3 / 2)) ** 2 + (
                x_3 * L / (x_3 ** 2 + L ** 2) ** (3 / 2)) ** 2) * zer_l

