# -*- coding: utf-8 -*-

import numpy as np
from astropy.cosmology import Planck15 as cosmo
import astropy.units as u

from scipy.integrate import solve_ivp

from utils import read_trees

from utils import omega_m, omega_b, omega_L, H_0
from metallicity import (
    accreting_gas_metallicity,
    stellar_metallicity_yield,
    wind_metallicity_enhancement_factor,
)

import utils as utils
import reionization as rei
import star_formation as sf
import supernovae_feedback as snw

m_halo, m_dot_halo, redshift = read_trees()
# print(type(m_halo), m_dot_halo, redshift)


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
        gas_accretion_rate = gas_accretion_rate * rei.epsilon_uv(
            redshift, halo_mass, halo_mass_dot
        )

    return np.asarray(gas_accretion_rate)


def evolve_gas(
    t,  # cosmic time
    gas_mass,  # gas mass
    gas_accretion_rate,
    halo_mass,
    past_sfr=0,
    stellar_metallicity=0,
    kind="delayed",
):
    """
    Eqn 1 in Menon et al 2024 with 2 and 3 substituted
    """

    redshift = utils.z_at_time(t)
    present_sfr = (sf.e_ff / sf.time_freefall(redshift)) * gas_mass
    wind_sfr = past_sfr

    if kind == "no":
        wind_sfr = 0
    if kind == "instantaneous":
        wind_sfr = present_sfr

    gas_mass_evolution_rate = (
        gas_accretion_rate
        - present_sfr
        - snw.eta(redshift, halo_mass, stellar_metallicity) * wind_sfr
    )
    # print(gas_mass_evolution_rate)
    return np.asarray(gas_mass_evolution_rate)


def star_formation_rate(
    t,  # cosmic time
    y,
    gas_mass,
):
    """
    Equation 2 in Menon et al 2024
    """

    redshift = utils.z_at_time(t)

    star_formation_rate = (sf.e_ff / sf.time_freefall(redshift)) * gas_mass
    return np.asarray(star_formation_rate)


def evolve_wind_mass(
    t,  # cosmic time
    y,  # wind mass
    gas_mass,
    halo_mass,
    stellar_metallicity,
):
    """
    Equation 3 in Menon et al 2024
    """

    redshift = utils.z_at_time(t)

    star_formation_rate = (sf.e_ff / sf.time_freefall(redshift)) * gas_mass

    wind_mass_rate = (
        snw.metallicity_function(stellar_metallicity)
        * snw.eta(redshift, halo_mass)
        * star_formation_rate
    )

    return wind_mass_rate


def evolve_gas_metals(
    t,
    y,
    gas_mass,
    gas_accretion_rate,
    halo_mass,
    past_sfr,
    stellar_metallicity,
    kind="delayed",
):
    redshift = utils.z_at_time(t)
    present_sfr = (sf.e_ff / sf.time_freefall(redshift)) * gas_mass
    wind_sfr = past_sfr

    if kind == "no":
        wind_sfr = 0.0
    if kind == "instantaneous":
        wind_sfr = present_sfr

    gas_metal_mass_evolution_rate = (
        (
            accreting_gas_metallicity * gas_accretion_rate
        )  # Enriched gas accreting from IGM
        - (y * present_sfr / gas_mass)  # Removal from ISM during star formation
        + (
            stellar_metallicity_yield * wind_sfr
        )  # Delayed enrichment of ISM by dying stars
        - (
            snw.eta(redshift, halo_mass, stellar_metallicity)
            * (y / gas_mass)
            * wind_sfr
        )  # Delayed removal from ISM by SN feedback
    )

    return gas_metal_mass_evolution_rate


def evolve_stellar_metallicity(t, y, gas_mass, gas_metal_mass):
    redshift = utils.z_at_time(t)
    stellar_metallicity_evolution_rate = sf.e_ff / sf.t_ff(redshift) * gas_metal_mass

    return stellar_metallicity_evolution_rate


def evolve_galaxies(t, state, params):
    """This will be the main function. We should have this call star formation, supernovae feedback, and such defined
    in the other files. Ideally this should be the only the function that is defined in this file.
    """
    gas_mass = state[0]
    star_mass = state[1]
    gas_metal_mass = state[2]
    stellar_metal_mass = state[3]

    # Extract relevant args for each subfunction
    gas_args = params["gas"]
    star_args = params["star"]
    gas_metals_args = params["gas_metals"]
    star_metals_args = params["star_metals"]

    # Call evolution functions
    d_gas_mass = evolve_gas(t, gas_mass, **gas_args)
    d_star_mass = evolve_star_formation(t, star_mass, gas_mass, **star_args)
    d_gas_metal_mass = evolve_gas_metals(t, gas_metal_mass, gas_mass, **gas_metals_args)
    d_stellar_metal_mass = evolve_stellar_metallicity(
        t, stellar_metal_mass, gas_mass, gas_metal_mass, **star_metals_args
    )

    return [d_gas_mass, d_star_mass, d_gas_metal_mass, d_stellar_metal_mass]


params = {
    "gas": {
        "gas_accretion_rate": baryon_accretion_rate,
        # "halo_mass": halo_mass,
        # "past_sfr": past_sfr,
        # "stellar_metallicity": z_star_val,
        "kind": "delayed",
    },
    "star": {},  # evolve_star_formation only needs gas_mass
    "gas_metals": {
        "gas_accretion_rate": baryon_accretion_rate,
        # "halo_mass": halo_mass,
        # "past_sfr": past_sfr,
        # "stellar_metallicity": z_star_val,
        "kind": "delayed",
    },
    "star_metals": {},  # evolve_stellar_metallicity only needs gas and gas metals
}

# uv_choice = input("Do you want to include background UV suppression or not?")

t_d = 2.015  # GYR; THIS SHOULD BE PUT AS A CHOICE FOR DELAYED/INSTANTANEOUS

for i in range(1):
    print(i)
    halo_mass, halo_mass_rate, redshift = read_trees()
    cosmic_time = utils.time_at_z(redshift)  # Gyr

    tsn = cosmic_time[0] + t_d  # Also a varying parameter

    gas_accretion_rate = baryon_accretion_rate(redshift, halo_mass, halo_mass_rate)
    gas_mass = np.array((omega_b / omega_m) * halo_mass)

    sfr = np.zeros(len(cosmic_time))
    stars_mass = np.zeros(len(cosmic_time))
    stars_metals = np.zeros(len(cosmic_time))

    gas_metals = np.zeros(len(cosmic_time))

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
                    sfr[j - 1],
                    stars_metals[j - 1],
                    "no",
                ),
            )

            solution = solve_ivp(
                star_formation_rate,
                t_span,
                [sfr[j - 1]],
                args=(gas_mass[j - 1],),
            )
