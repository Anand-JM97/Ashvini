# -*- coding: utf-8 -*-
import numpy as np
from scipy.integrate import solve_ivp
from utils import read_trees
from utils import omega_m, omega_b
from metallicity import (
    IGM_metallicity,
    metallicity_yield,
)

import utils as utils
import reionization as reion
import supernovae_feedback as sn

from star_formation import star_formation_rate
from supernovae_feedback import wind_mass_evolution_rate

m_halo, m_dot_halo, redshift = read_trees()


def baryon_accretion_rate(redshift, halo_mass, halo_mass_dot, uv_suppression=True):
    """
    Function to obtain the cosmological baryonic accretion rate.

    Args:
        z (float): Parameter representing redshift.
        m_h (float): Halo mass value at z.
        m_dot_h (float): Halo mass accretion rate at z.
        uv_suppresion_check (bool ?): Check for whether accretion suppression due to background UV is on or not.

    Returns:
        Float: The baryonic cosmological accretion rate at z.
    """
    gas_accretion_rate = (omega_b / omega_m) * halo_mass_dot
    if uv_suppression:
        gas_accretion_rate *= reion.uv_suppression(redshift, halo_mass, halo_mass_dot)

    return np.asarray(gas_accretion_rate)


def evolve_gas(
    t,  # cosmic time
    gas_mass,  # gas mass
    gas_accretion_rate,
    halo_mass,
    stellar_metallicity,
    past_sfr=0,
    kind="delayed",
):
    """
    Eqn 1 in Menon et al 2024 with 2 and 3 substituted
    """

    redshift = utils.z_at_time(t)
    present_sfr = star_formation_rate(t, gas_mass=gas_mass)
    wind_sfr = past_sfr

    if kind == "no":
        wind_sfr = 0
    if kind == "instantaneous":
        wind_sfr = present_sfr

    gas_mass_evolution_rate = (
        gas_accretion_rate
        - present_sfr
        - sn.mass_loading(redshift, halo_mass, stellar_metallicity)
        * wind_sfr  # - wind_mass_evolution_rate(redshift, gas_mass, halo_mass, wind_sfr, stellar_metallicity)
    )
    return np.asarray(gas_mass_evolution_rate)


def evolve_gas_metals(
    t,
    gas_metal_mass,
    gas_mass,
    gas_accretion_rate,
    halo_mass,
    past_sfr,
    stellar_metallicity,
    kind="delayed",
):
    redshift = utils.z_at_time(t)
    present_sfr = star_formation_rate(t, gas_mass=gas_mass)
    wind_sfr = past_sfr

    if kind == "no":
        wind_sfr = 0.0
    if kind == "instantaneous":
        wind_sfr = present_sfr

    gas_metals_rate = (
        (IGM_metallicity * gas_accretion_rate)  # Enriched gas accreting from IGM
        - (
            gas_metal_mass * present_sfr / gas_mass
        )  # Removal from ISM during star formation
        + (metallicity_yield * wind_sfr)  # Delayed enrichment of ISM by dying stars
        - (
            sn.mass_loading(redshift, halo_mass, stellar_metallicity)
            * (gas_metal_mass / gas_mass)
            * wind_sfr
        )  # Delayed removal from ISM by SN feedback
    )

    return gas_metals_rate


def evolve_stars_metals(t, gas_metals, gas_mass):
    # TODO: We can simply call starformation_rate here- Should SFR use gas_metals as argument instead of gas_mass?
    stars_metals_rate = (
        star_formation_rate(t, gas_mass=gas_mass)
        * gas_metals
        / gas_mass  # Heavy elements captured during star formation
    )
    return stars_metals_rate


# uv_choice = input("Do you want to include background UV suppression or not?")

t_d = 2.015  # GYR; THIS SHOULD BE PUT AS A CHOICE FOR DELAYED/INSTANTANEOUS

for i in np.arange(1):
    print(i)
    halo_mass, halo_mass_rate, redshift = read_trees()

    # TODO: Taking only the first Ntest values for testing
    Ntest = 20000
    halo_mass, halo_mass_rate, redshift = (
        halo_mass[:Ntest],
        halo_mass_rate[:Ntest],
        redshift[:Ntest],
    )
    cosmic_time = utils.time_at_z(redshift)  # Gyr

    tsn = cosmic_time[0] + t_d  # Also a varying parameter

    gas_accretion_rate = baryon_accretion_rate(redshift, halo_mass, halo_mass_rate)
    gas_mass = np.array((omega_b / omega_m) * halo_mass)
    gas_mass = np.zeros(len(cosmic_time))
    gas_metals = np.zeros(len(cosmic_time))

    stars_mass = np.zeros(len(cosmic_time))
    stars_metals = np.zeros(len(cosmic_time))

    stars_metals = np.zeros(len(cosmic_time))

    dust_mass = np.zeros(len(cosmic_time))

    for j in range(1, len(cosmic_time)):
        print(j)
        t_span = [cosmic_time[j - 1], cosmic_time[j]]

        if cosmic_time[j] <= tsn:
            solution = solve_ivp(
                evolve_gas,
                t_span,
                [gas_mass[j - 1]],
                args=(
                    gas_accretion_rate[j - 1],
                    halo_mass[j - 1],
                    stars_mass[j - 1],
                    stars_metals[j - 1],
                    "delayed",
                ),
            )
            gas_mass[j] = solution.y[0, -1]

            solution = solve_ivp(
                lambda t, y: [star_formation_rate(t, gas_mass[j - 1])],
                t_span,
                [stars_mass[j - 1]],
            )
            stars_mass[j] = solution.y[0, -1]

            solution = solve_ivp(
                evolve_gas_metals,
                t_span,
                [gas_metals[j - 1]],
                args=(
                    gas_mass[j - 1],
                    gas_accretion_rate[j - 1],
                    halo_mass[j - 1],
                    stars_mass[j - 1],
                    stars_metals[j - 1],
                    "delayed",
                ),
            )
            gas_metals[j] = solution.y[0, -1]

            solution = solve_ivp(
                lambda t, y: [
                    evolve_stars_metals(t, gas_metals[j - 1], gas_mass[j - 1])
                ],
                t_span,
                [stars_metals[j - 1]],
            )
            stars_metals[j] = solution.y[0, -1]

# print(gas_mass, stars_mass, gas_metals, stars_metals)

dir_out = "../data/outputs/"
np.savez(
    dir_out + f"first_{Ntest}.npz",
    gas_mass=gas_mass,
    stars_mass=stars_mass,
    gas_metals=gas_metals,
    stars_metals=stars_metals,
)
