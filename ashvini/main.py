# -*- coding: utf-8 -*-
import numpy as np
from scipy.integrate import solve_ivp

from utils import read_trees

import utils as utils

from star_formation import star_formation_rate

from gas_evolve import gas_inflow_rate, update_gas_reservior
from metallicity import evolve_gas_metals, evolve_stars_metals

t_d = 0.015  # GYR; THIS SHOULD BE PUT AS A CHOICE FOR DELAYED/INSTANTANEOUS

tiny = 1e-12  # small number for numerical gymnastics...

method = "RK45"
method = "LSODA"
for i in np.arange(1):
    # print(i)

    halo_mass, halo_mass_rate, redshift = read_trees()

    # TODO: Taking only the first Ntest values for testing
    Ntest = 10000
    Ntest = len(redshift)
    halo_mass, halo_mass_rate, redshift = (
        halo_mass[:Ntest],
        halo_mass_rate[:Ntest],
        redshift[:Ntest],
    )

    cosmic_time = utils.time_at_z(redshift)  # Gyr

    tsn = cosmic_time[0] + t_d  # Also a varying parameter
    gas_accretion_rate = gas_inflow_rate(redshift, halo_mass, halo_mass_rate)

    gas_mass = tiny * np.ones(len(cosmic_time))
    gas_mass = np.zeros(len(cosmic_time))

    gas_metals = np.zeros(len(cosmic_time))

    stars_mass = np.zeros(len(cosmic_time))
    stars_metals = np.zeros(len(cosmic_time))
    sfr = np.zeros(len(cosmic_time))

    stellar_metallicity = np.zeros(len(cosmic_time))

    dust_mass = np.zeros(len(cosmic_time))

    for j in range(1, len(cosmic_time)):
        print(j)
        t_span = [cosmic_time[j - 1], cosmic_time[j]]

        if cosmic_time[j] <= tsn:
            solution = solve_ivp(
                update_gas_reservior,
                t_span,
                [gas_mass[j - 1]],
                method=method,
                args=(
                    gas_accretion_rate[j - 1],
                    halo_mass[j - 1],
                    stellar_metallicity[j - 1],
                    0.0,
                    "no",
                ),
            )

            gas_mass[j] = solution.y[0, -1]

            solution = solve_ivp(
                lambda t, y: [star_formation_rate(t, gas_mass[j - 1])],
                t_span,
                [stars_mass[j - 1]],
                method=method,
            )
            stars_mass[j] = solution.y[0, -1]

            solution = solve_ivp(
                evolve_gas_metals,
                t_span,
                [gas_metals[j - 1]],
                method=method,
                args=(
                    gas_mass[j - 1],
                    gas_accretion_rate[j - 1],
                    halo_mass[j - 1],
                    stellar_metallicity[j - 1],
                    sfr[j - 1],
                    "no",
                ),
            )
            gas_metals[j] = solution.y[0, -1]

            solution = solve_ivp(
                lambda t, y: [
                    evolve_stars_metals(t, gas_metals[j - 1], gas_mass[j - 1])
                ],
                t_span,
                [stars_metals[j - 1]],
                method=method,
            )
            stars_metals[j] = solution.y[0, -1]

            delay_counter = j  # catch the last value of j before switching on SNe

        elif cosmic_time[j] > tsn:
            solution = solve_ivp(
                update_gas_reservior,
                t_span,
                [gas_mass[j - 1]],
                method=method,
                args=(
                    gas_accretion_rate[j - 1],
                    halo_mass[j - 1],
                    stellar_metallicity[j - 1],
                    sfr[j - delay_counter - 1],
                    "instantaneous",
                ),
            )

            gas_mass[j] = solution.y[0, -1]

            solution = solve_ivp(
                lambda t, y: [star_formation_rate(t, gas_mass[j - 1])],
                t_span,
                [stars_mass[j - 1]],
                method=method,
            )
            stars_mass[j] = solution.y[0, -1]

            solution = solve_ivp(
                evolve_gas_metals,
                t_span,
                [gas_metals[j - 1]],
                method=method,
                args=(
                    gas_mass[j - 1],
                    gas_accretion_rate[j - 1],
                    halo_mass[j - 1],
                    stellar_metallicity[j - 1],
                    sfr[j - 1 - delay_counter],
                    "instantaneous",
                ),
            )

            gas_metals[j] = solution.y[0, -1]

            solution = solve_ivp(
                lambda t, y: [
                    evolve_stars_metals(
                        t,
                        gas_metals[j - 1],
                        gas_mass[j - 1],
                    )
                ],
                t_span,
                [stars_metals[j - 1]],
                method=method,
            )
            stars_metals[j] = solution.y[0, -1]

        sfr[j] = star_formation_rate(cosmic_time[j], gas_mass[j])

        if gas_mass[j] < 0.0:
            gas_mass[j] = 0.0
            gas_metals[j] = 0.0

        if stars_mass[j] < 0.0:
            stars_mass[j] = 0.0
            stars_metals[j] = 0.0

        if gas_metals[j] < 0.0:
            gas_metals[j] = 0.0

        if stars_metals[j] < 0.0:
            stars_metals[j] = 0.0

        if stars_mass[j] > 0:
            stellar_metallicity[j] = stars_metals[j] / stars_mass[j]
        else:
            stellar_metallicity[j] = 0

dir_out = "../data/outputs/"
np.savez(
    dir_out + f"first_{Ntest}.npz",
    gas_mass=gas_mass,
    stars_mass=stars_mass,
    gas_metals=gas_metals,
    stars_metals=stars_metals,
)

import matplotlib.pyplot as plt

plt.plot(cosmic_time, halo_mass, label="halo_mass")
plt.plot(cosmic_time, halo_mass_rate, label="halo_mass_rate")
plt.plot(cosmic_time, gas_mass, label="gas_mass")
plt.plot(cosmic_time, stars_mass, label="stars_mass")
plt.plot(cosmic_time, sfr, label="sfr")
plt.plot(cosmic_time, gas_metals, label="gas_metals")
plt.plot(cosmic_time, stars_metals, label="stars_metals")
plt.yscale("log")
plt.legend()
plt.show()
