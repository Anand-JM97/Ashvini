# -*- coding: utf-8 -*-

import numpy as np
from astropy.cosmology import Planck15 as cosmo
import astropy.units as u

from scipy.integrate import solve_ivp

from io import read_trees

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


# MERGER TREE INPUT

# m_halo, m_dot_halo, redshift = read_trees()


def cosmological_accretion_rate(z, m_h, m_dot_h, uv_suppression_check):
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

    if uv_suppression_check == 1:
        m_dot_cg_val = (omega_b / omega_m) * m_dot_h * rei.epsilon_uv(z, m_h, m_dot_h)

    elif uv_suppression_check == 0:
        m_dot_cg_val = (omega_b / omega_m) * m_dot_h

    return m_dot_cg_val


def evolve_gas(
    t,  # cosmic time
    gas_mass,  # gas mass
    gas_accretion_rate,
    halo_mass,
    past_sfr,
    stellar_metallicity,
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
    return gas_mass_evolution_rate


def evolve_star_formation(
    t,  # cosmic time
    y,  # stellar mass
    gas_mass,
):
    """
    Equation 2 in Menon et al 2024
    """

    redshift = utils.z_at_time(t)

    star_formation_rate = (sf.e_ff / sf.time_freefall(redshift)) * gas_mass
    return star_formation_rate


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
    d_stellar_metal_mass = evolve_stellar_metallicity(t, stellar_metal_mass, gas_mass, gas_metal_mass, **star_metals_args)

    return [d_gas_mass, d_star_mass, d_gas_metal_mass, d_stellar_metal_mass]


params = {
    "gas": {
        "gas_accretion_rate": gas_accretion_rate[j],
        "halo_mass": halo_mass[j],
        "past_sfr": past_sfr[j],
        "stellar_metallicity": z_star_val,
        "kind": "delayed"
    },
    "star": {},  # evolve_star_formation only needs gas_mass
    "gas_metals": {
        "gas_accretion_rate": gas_accretion_rate[j],
        "halo_mass": halo_mass[j],
        "past_sfr": past_sfr[j],
        "stellar_metallicity": z_star_val,
        "kind": "delayed"
    },
    "star_metals": {}  # evolve_stellar_metallicity only needs gas and gas metals
}

start = 0
stop = 100
check = 86

no = 10  # HALO MASS POWER VALUE

uv_choice = input("Do you want to include background UV suppression or not?")

t_d = 0.015  # GYR; THIS SHOULD BE PUT AS A CHOICE FOR DELAYED/INSTANTANEOUS


# DELAYED FEEDBACK

for i in range(start, stop, 1):
    redshift = np.array([])
    halo_mass = np.array([])
    halo_mass_rate = np.array([])
    print(i)

    redshift = np.loadtxt(
        f"C:/Users/Anand Menon/Documents/ICRAR Stuff/Data Sets/Sorted Data/mh{no}_data/Redshifts/redshift_{i}.txt",
        delimiter=" ",
    )
    halo_mass = np.loadtxt(
        f"C:/Users/Anand Menon/Documents/ICRAR Stuff/Data Sets/Sorted Data/mh{no}_data/Halo Mass/halo_mass_{i}.txt",
        delimiter=" ",
    )
    halo_mass_rate = np.loadtxt(
        f"C:/Users/Anand Menon/Documents/ICRAR Stuff/Data Sets/Sorted Data/mh{no}_data/Halo Mass Rate/halo_mass_rate_{i}.txt",
        delimiter=" ",
    )

    cosmic_time = t(redshift)
    h = (cosmic_time[len(cosmic_time) - 1] - cosmic_time[0]) / len(cosmic_time)

    tsn = cosmic_time[0] + t_d  # Also a varying parameter

    gas_accretion_rate = cosmological_accretion_rate(
        redshift, halo_mass, halo_mass_rate
    )  # TODO: UV suppression check

    ini_m_gas = [0.0]  # [(omega_b / omega_m) * halo_mass]  # TODO:
    ini_m_star = [0.0]

    ini_m_z_gas = [0.0]
    ini_m_z_star = [0.0]

    ini_m_dust = [0.0]

    m_g_val_1 = []
    m_star_val_1 = []
    m_z_g_val_1 = []
    m_z_star_val_1 = []
    m_dust_val_1 = []

    m_dot_star_vals = []

    z_star_val = 0.0

    f_vals = []

    k = 0

    for j in range(1, len(cosmic_time)):
        t_span = [cosmic_time[j - 1], cosmic_time[j]]

        if cosmic_time[j] <= tsn:
            solution = solve_ivp(
                evolve_gas, t_span, ini_m_gas, args=[gas_accretion_rate[j-1], halo_mass[j-1], past_sfr[j-1], stellar_metallicity[j-1], kind="no?"]
            )
            m_g = solution.y[0][len(solution.y[0]) - 1]

            solution = solve_ivp(
                diff_eqn_star_1, t_span, ini_m_star, args=[ini_m_gas[0]], max_step=h
            )
            m_s = solution.y[0][len(solution.y[0]) - 1]

            solution = solve_ivp(
                diff_eqn_zgas_1,
                t_span,
                ini_m_z_gas,
                args=[ini_m_gas[0], m_dot_cg_val[j]],
                max_step=h,
            )
            m_z_g = solution.y[0][len(solution.y[0]) - 1]

            solution = solve_ivp(
                diff_eqn_zstar_2,
                t_span,
                ini_m_z_star,
                args=[ini_m_gas[0], ini_m_z_gas[0]],
                max_step=h,
            )
            m_z_s = solution.y[0][len(solution.y[0]) - 1]

        elif cosmic_time[j] > tsn:
            solution = solve_ivp(
                diff_eqn_gas_2,
                t_span,
                ini_m_gas,
                args=[m_dot_cg_val[j], halo_mass[j], m_dot_star_vals[k], z_star_val],
                max_step=h,
            )
            m_g = solution.y[0][len(solution.y[0]) - 1]

            solution = solve_ivp(
                diff_eqn_star_2, t_span, ini_m_star, args=[ini_m_gas[0]], max_step=h
            )
            m_s = solution.y[0][len(solution.y[0]) - 1]

            solution = solve_ivp(
                diff_eqn_zgas_2,
                t_span,
                ini_m_z_gas,
                args=[
                    ini_m_gas[0],
                    m_dot_cg_val[j],
                    halo_mass[j],
                    m_dot_star_vals[k],
                    z_star_val,
                ],
                max_step=h,
            )
            m_z_g = solution.y[0][len(solution.y[0]) - 1]

            solution = solve_ivp(
                diff_eqn_zstar_2,
                t_span,
                ini_m_z_star,
                args=[ini_m_gas[0], ini_m_z_gas[0]],
                max_step=h,
            )
            m_z_s = solution.y[0][len(solution.y[0]) - 1]

            k = k + 1

        ini_m_gas = [m_g]
        ini_m_star = [m_s]
        ini_m_z_gas = [m_z_g]
        ini_m_z_star = [m_z_s]

        if ini_m_gas[0] < 0.0:
            ini_m_gas[0] = 0.0
            ini_m_z_gas[0] = 0.0

        if ini_m_star[0] < 0.0:
            ini_m_star[0] = 0.0
            ini_m_z_star[0] = 0.0

        if ini_m_z_gas[0] < 0.0:
            ini_m_z_gas[0] = 0.0

        if ini_m_z_star[0] < 0.0:
            ini_m_z_star[0] = 0.0

        if ini_m_dust[0] < 0.0:
            ini_m_dust[0] = 0.0

        if ini_m_star[0] == 0.0:
            z_star_val = 0.0

        elif ini_m_star[0] > 0.0:
            z_star_val = ini_m_z_star[0] / ini_m_star[0]

        m_dot_star_val = (sf.e_ff / sf.t_ff(redshift[j])) * ini_m_gas[0]
        m_dot_star_vals = np.append(m_dot_star_vals, [m_dot_star_val])

        m_g_val_1 = np.append(m_g_val_1, ini_m_gas)
        m_star_val_1 = np.append(m_star_val_1, ini_m_star)
        m_z_g_val_1 = np.append(m_z_g_val_1, ini_m_z_gas)
        m_z_star_val_1 = np.append(m_z_star_val_1, ini_m_z_star)
        m_dust_val_1 = np.append(m_dust_val_1, ini_m_dust)

    print(len(m_g_val_1))
    print(m_z_star_val_1)

    np.savetxt(
        f"C:/Users/Anand Menon/Documents/ICRAR Stuff/Results/No UV Delay/mh{no}_data/gas/tree_{i}.txt",
        m_g_val_1,
        delimiter=" ",
    )
    np.savetxt(
        f"C:/Users/Anand Menon/Documents/ICRAR Stuff/Results/No UV Delay/mh{no}_data/star/tree_{i}.txt",
        m_star_val_1,
        delimiter=" ",
    )
    np.savetxt(
        f"C:/Users/Anand Menon/Documents/ICRAR Stuff/Results/No UV Delay/mh{no}_data/z_gas/tree_{i}.txt",
        m_z_g_val_1,
        delimiter=" ",
    )
    np.savetxt(
        f"C:/Users/Anand Menon/Documents/ICRAR Stuff/Results/No UV Delay/mh{no}_data/z_star/tree_{i}.txt",
        m_z_star_val_1,
        delimiter=" ",
    )
