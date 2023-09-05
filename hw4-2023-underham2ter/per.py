#!/usr/bin/env python3

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier
from mixfit import em_double_cluster
import json
import matplotlib.pyplot as plt

plt.rcParams['axes.facecolor'] = 'black'
if __name__ == "__main__":
    center_coord = SkyCoord('02h21m00s +57d07m42s')
    vizier = Vizier(
        columns=['RAJ2000', 'DEJ2000', 'pmRA', 'pmDE'],
        column_filters={'BPmag': '<16', 'pmRA': '!=', 'pmDE': '!='},  # число больше — звёзд больше
        row_limit=10000
    )
    stars = vizier.query_region(
        center_coord,
        width=1.0 * u.deg,
        height=1.0 * u.deg,
        catalog=['I/350'],  # Gaia EDR3
    )[0]

    ra = stars['RAJ2000']._data  # прямое восхождение, аналог долготы
    dec = stars['DEJ2000']._data  # склонение, аналог широты

    x1 = (ra - ra.mean()) * np.cos(dec / 180 * np.pi) + ra.mean()
    x2 = dec
    v1 = stars['pmRA']._data
    v2 = stars['pmDE']._data

    muv = np.array([np.mean(v1), np.mean(v2)])
    mu1 = np.array([np.mean(x1) - np.std(x1), np.mean(x2) - np.std(x2)])
    mu2 = np.array([np.mean(x1) + np.std(x1), np.mean(x2) + np.std(x2)])
    tau1, tau2 = 0.3, 0.3

    sigmax2 = np.array([np.var(x1), np.var(x2)])
    sigmav2 = np.array([np.var(v1), np.var(v2)])
    sigma02 = np.var(v1) + np.var(v2)

    x = np.transpose(np.vstack((x1, x2, v1, v2)))

    theta_ = (tau1, tau2, muv, mu1, mu2, sigma02, sigmax2, sigmav2)
    res, T1, T2, T3 = em_double_cluster(x, *theta_)
    tau1, tau2, mu1, mu2, muv, sigmax2, sigmav2, sigma02 = res

    size_ratio = np.round(tau1 / tau2, 2)

    rav, decv = np.round(muv, 2)

    ra1, dec1 = np.round(mu1, 2)
    ra2, dec2 = np.round(mu2, 2)

    sigmax = np.sqrt(sigmax2[0])
    sigmav = np.sqrt(sigmav2[0])
    sigma0 = np.sqrt(sigma02)

    # print(size_ratio, rav, decv, ra1, dec1, ra2, dec2, sigmax, sigmav, sigma0)

    ans = {"size_ratio": size_ratio,
           "motion":
               {"ra": rav, "dec": decv},
           "clusters": [
               {
                   "center": {"ra": ra1, "dec": dec1},
               },
               {
                   "center": {"ra": ra2, "dec": dec2},
               }
           ]
           }

    with open('per.json', 'w') as f:
        json.dump(ans, f, indent=4)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))

    pict1 = ax1.scatter(x1, x2, s=1, c=T1, cmap='RdYlBu')
    fig.colorbar(pict1, ax=ax2, location='right')
    ax1.scatter(ra1, dec1, s=13, c='cornflowerblue')
    ax1.scatter(ra2, dec2, s=13, c='red')
    circle1 = plt.Circle((ra1, dec1), sigmax, color='cornflowerblue', fill=False, label='Первое скопление')
    circle2 = plt.Circle((ra2, dec2), sigmax, color='red', fill=False, label='Второе скопление')
    ax1.add_patch(circle1)
    ax1.add_patch(circle2)
    ax1.set_title(r'Распределение координат')
    ax1.set_xlabel(r'Прямое восхождение(RA),$ \degree $ ')
    ax1.set_ylabel(r'Склонение,(DE), $ \degree $')
    ax1.legend(loc='best', labelcolor='white')

    ax2.scatter(v1, v2, s=1, c=T1, cmap='RdYlBu')
    ax2.set_xlim(-1.2, -0.25)
    ax2.set_ylim(-1.6, -0.8)
    ax2.set_title('Распределение скоростей')
    ax2.set_xlabel('pmRA')
    ax2.set_ylabel('pmDE')

    plt.savefig('per.png', dpi=600)
