import numpy as np
import utils as utils
from scipy.integrate import solve_ivp

from run_params import PARAMS

UV_background = PARAMS.reion.UVB_enabled
t_d = PARAMS.sn.delay_time  # delay time for SNe feedback, in Gyr

from utils import read_trees

from star_formation import star_formation_rate
from gas_evolve import gas_inflow_rate, update_gas_reservior
from metallicity import evolve_gas_metals, evolve_stars_metals


tiny = 1e-12  # small number for numerical gymnastics...


method = "LSODA"

for i in np.arange(1):
    halo_mass, halo_mass_rate, redshift = read_trees()

    # TODO: Taking only the first Ntest values for testing
    Ntest = len(redshift)
    halo_mass = halo_mass[:Ntest]
    halo_mass_rate = halo_mass_rate[:Ntest]
    redshift = redshift[:Ntest]

    cosmic_time = utils.time_at_z(redshift)  # Gyr
    tsn = cosmic_time[0] + t_d  # Supernova switch-on time

    n = len(cosmic_time)
    gas_mass = np.zeros(n)
    gas_metals = np.zeros(n)
    stars_mass = np.zeros(n)
    stars_metals = np.zeros(n)
    sfr = np.zeros(n)
    stellar_metallicity = np.zeros(n)
    # dust_mass = np.zeros(n)

    gas_accretion_rate = gas_inflow_rate(
        redshift, halo_mass, halo_mass_rate, UV_background
    )

    delay_counter = None

    for j in range(1, n):
        print(j)
        t_span = [cosmic_time[j - 1], cosmic_time[j]]

        if cosmic_time[j] <= tsn:
            feedback_type = "no"
            sfr_feedback = 0.0
            delay_counter = j
        else:
            feedback_type = "instantaneous"
            feedback_type = "delayed"
            sfr_feedback = (
                sfr[j - delay_counter - 1] if delay_counter is not None else 0.0
            )

        # Update gas mass
        sol = solve_ivp(
            update_gas_reservior,
            t_span,
            [gas_mass[j - 1]],
            method=method,
            args=(
                gas_accretion_rate[j - 1],
                halo_mass[j - 1],
                stellar_metallicity[j - 1],
                sfr_feedback,
                feedback_type,
            ),
        )
        gas_mass[j] = sol.y[0, -1]

        # Update stellar mass
        sol = solve_ivp(
            lambda t, y: [star_formation_rate(t, gas_mass[j - 1])],
            t_span,
            [stars_mass[j - 1]],
            method=method,
        )
        stars_mass[j] = sol.y[0, -1]

        # Update gas metals
        sfr_input = sfr[j - 1] if cosmic_time[j] <= tsn else sfr[j - 1 - delay_counter]
        sol = solve_ivp(
            evolve_gas_metals,
            t_span,
            [gas_metals[j - 1]],
            method=method,
            args=(
                gas_mass[j - 1],
                gas_accretion_rate[j - 1],
                halo_mass[j - 1],
                stellar_metallicity[j - 1],
                sfr_input,
                feedback_type,
            ),
        )
        gas_metals[j] = sol.y[0, -1]

        # Update stellar metals
        sol = solve_ivp(
            lambda t, y: [evolve_stars_metals(t, gas_metals[j - 1], gas_mass[j - 1])],
            t_span,
            [stars_metals[j - 1]],
            method=method,
        )
        stars_metals[j] = sol.y[0, -1]

        # Star formation rate at current time
        sfr[j] = star_formation_rate(cosmic_time[j], gas_mass[j])

        # Enforce non-negativity
        gas_mass[j] = max(gas_mass[j], 0.0)
        gas_metals[j] = max(gas_metals[j], 0.0)
        stars_mass[j] = max(stars_mass[j], 0.0)
        stars_metals[j] = max(stars_metals[j], 0.0)

        # Stellar metallicity
        if stars_mass[j] > 0:
            stellar_metallicity[j] = stars_metals[j] / stars_mass[j]
        else:
            stellar_metallicity[j] = 0.0

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
