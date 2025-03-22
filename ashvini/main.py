# -*- coding: utf-8 -*-


import numpy as np
from astropy.cosmology import Planck15 as cosmo
import astropy.units as u

from scipy.integrate import solve_ivp

from io import read_trees
from utils import omega_m, omega_b, omega_L, H_0

import utils as utils
import reionization as rei
import star_formation as sf
import supernovae_feedback as snw


A = 0.030


def evolve_galaxies():
    """This will be the main function. We should have this call star formation, supernovae feedback, and such defined
    in the other files. Ideally this should be the only the function that is defined in this file.
    """

    return 1


# MERGER TREE INPUT

m_halo, m_dot_halo, redshift = read_trees()

# Metallicity parameters

z_igm = 10 ** (-3)
y_z = 0.06
zeta_w = 1


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


# IVF SOLVER FUNCTIONS

# NO FEEDBACK ACTING SCENARIO FOR DELAYED FEEDBACK


def evolve_gas(
    time,
    y,      #gas mass
    gas_accretion_rate,
    halo_mass,
    past_sfr,
    stellar_metallicity,
    kind="delayed",
):

    """
    Eqn 1 in Menon et al 2024 with 2 and 3 substituted
    """

    redshift = utils.z_at_time(time)
    SFR_now = (sf.e_ff / sf.time_freefall(redshift)) * y
    wind_modifier = past_sfr

    if kind == "no":
        wind_modifier = 0
    if kind == "instantaneous":
        wind_modifier = SFR_now

    gas_mass_evolution_rate = (
        gas_accretion_rate
        - SFR_now
        - snw.eta(redshift, halo_mass, stellar_metallicity) * wind_modifier
    )
    return gas_mass_evolution_rate

def evolve_star_formation(
    time,
    y,          #stellar mass
    gas_mass,
):
    
    """
    Equation 2 in Menon et al 2024
    """
    
    redshift = utils.z_at_time(time)
    
    star_formation_rate = (sf.e_ff / sf.time_freefall(redshift)) * gas_mass
    return star_formation_rate

def evolve_supernova_feedback(
    time,
    y,          #wind mass
    gas_mass,
    halo_mass,
    stellar_metallicity
):
    
    """
    Equation 3 in Menon et al 2024
    """
    
    redshift = utils.z_at_time(time)
    
    star_formation_rate = (sf.e_ff / sf.time_freefall(redshift)) * gas_mass
    wind_mass_rate = (
        snw.metallicity_function(stellar_metallicity)
        * snw.eta(redshift, halo_mass)
        * star_formation_rate
    )
    
    return wind_mass_rate

def gas_metallicity_mass_evolution_equation_no_feedback(
    t, y, m_g, m_d_cg
):  # diff_eqns_1
    z_val = utils.z_at_time(t)

    f_m_z_gas = (z_igm * m_d_cg) - ((y) * (sf.e_ff / sf.t_ff(z_val)))
    return f_m_z_gas


# FEEDBACK ACTING SCENARIO FOR DELAYED FEEDBACK


def gas_metallicity_mass_evolution_equation_delayed_feedback(
    t, y, m_g, m_d_cg, m_halo, m_d_s_d, z_star
):  # diff_eqns_2
    z_val = utils.z_at_time(t)

    f_m_z_gas = (
        (z_igm * m_d_cg)
        - (y * (sf.e_ff / sf.t_ff(z_val)))
        + (y_z * m_d_s_d)
        - (snw.eta(z_val, m_halo, z_star) * (y / m_g) * m_d_s_d)
    )
    return f_m_z_gas


# INSTANTANEOUS FEEDBACK


def gas_metallicity_mass_evolution_equation_instantaneous_feedback(
    t, y, m_g, m_d_cg, m_halo, z_star
):  # diff_eqns_eq_1
    z_val = utils.z_at_time(t)
    f_m_star = (sf.e_ff / sf.t_ff(z_val)) * m_g

    f_m_z_gas = (
        (z_igm * m_d_cg)
        - (snw.eta(z_val, m_halo, z_star) * y * sf.e_ff / sf.t_ff(z_val))
        + (y_z * f_m_star)
        - (y * sf.e_ff / sf.t_ff(z_val))
    )
    return f_m_z_gas


# STELLAR METALLICITY EQUATIONS- Remove (y_z*e_ff/t_ff(z_val)*m_g) if not needed


def diff_eqn_zstar_1(t, y, m_g):
    z_val = z(t)
    f_m_z_star = 0.0
    return f_m_z_star


def diff_eqn_zstar_2(t, y, m_g, m_z_g):
    z_val = z(t)
    f_m_z_star = (m_z_g) * (sf.e_ff / sf.t_ff(z_val))
    return f_m_z_star


start = 0
stop = 100
check = 86


no = 10  # HALO MASS POWER VALUE

uv_choice = input("Do you want to include background UV suppression or not?")

t_d = input(
    "Do you want to include background UV suppression or not?"
)  # THIS SHOULD BE PUT AS A CHOICE FOR DELAYED/INSTANTANEOUS


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

    print(len(cosmic_time))

    tsn = cosmic_time[0] + t_d  # Also a varying parameter

    if uv_choice == "Yes" or uv_choice == "yes":
        m_dot_cg_val = m_dot_cg_with_UV(redshift, halo_mass, halo_mass_rate)

    elif uv_choice == "No" or uv_choice == "no":
        m_dot_cg_val = m_dot_cg_no_UV(redshift, halo_mass, halo_mass_rate)

    ini_m_gas = [0.0]
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

    for j in range(0, len(cosmic_time)):
        if j == 0:
            t_span = [cosmic_time[j], cosmic_time[j]]
        else:
            t_span = [cosmic_time[j - 1], cosmic_time[j]]

        if cosmic_time[j] <= tsn:
            solution = solve_ivp(
                diff_eqn_gas_1, t_span, ini_m_gas, args=[m_dot_cg_val[j]], max_step=h
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

    plt.semilogy(cosmic_time, halo_mass, label="Halo mass")
    plt.semilogy(cosmic_time, halo_mass_rate, label="Halo mass rate")

    plt.semilogy(cosmic_time, m_g_val_1, label="Gas mass values")
    plt.semilogy(cosmic_time, m_star_val_1, label="Stellar mass values")

    plt.semilogy(cosmic_time, m_z_g_val_1, label="Gas metallicity values")
    plt.semilogy(cosmic_time, m_z_star_val_1, label="Stellar metallicity")

    plt.semilogy(cosmic_time, m_dust_val_1, label="Dust mass")

    plt.ylim(10**-3, 10**10)

    plt.show()

# INSTANTANEOUS FEEDBACK


ini_m_gas_eq = [0.0]
ini_m_star_eq = [0.0]

ini_m_z_gas_eq = [0.0]
ini_m_z_star_eq = [0.0]

ini_m_dust_eq = [0.0]

m_g_eq_val_1 = []
m_star_eq_val_1 = []
m_z_g_eq_val_1 = []
m_z_star_eq_val_1 = []
m_dust_eq_val_1 = []

start = 0
stop = 100
check = 86

for i in range(start, stop, 1):
    print(i)
    redshift = np.array([])
    halo_mass = np.array([])
    halo_mass_rate = np.array([])

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

    tsn = cosmic_time[0] + t_d

    if uv_choice == "Yes" or uv_choice == "yes":
        m_dot_cg_val = m_dot_cg_with_UV(redshift, halo_mass, halo_mass_rate)

    elif uv_choice == "No" or uv_choice == "no":
        m_dot_cg_val = m_dot_cg_no_UV(redshift, halo_mass, halo_mass_rate)

    ini_m_gas_eq = [0.0]
    ini_m_star_eq = [0.0]

    ini_m_z_gas_eq = [0.0]
    ini_m_z_star_eq = [0.0]

    ini_m_dust_eq = [0.0]

    m_g_eq_val_1 = []
    m_star_eq_val_1 = []
    m_z_g_eq_val_1 = []
    m_z_star_eq_val_1 = []
    m_dust_eq_val_1 = []

    z_star_val = 0.0

    for j in range(0, len(cosmic_time)):
        if j == 0:
            t_span = [cosmic_time[j], cosmic_time[j]]
        else:
            t_span = [cosmic_time[j - 1], cosmic_time[j]]

        solution = solve_ivp(
            diff_eqn_eq_gas_1,
            t_span,
            ini_m_gas_eq,
            args=[m_dot_cg_val[j], halo_mass[j], z_star_val],
            max_step=h,
        )
        m_g = solution.y[0][len(solution.y[0]) - 1]

        solution = solve_ivp(
            diff_eqn_eq_star_1,
            t_span,
            ini_m_star_eq,
            args=[ini_m_gas_eq[0]],
            max_step=h,
        )
        m_s = solution.y[0][len(solution.y[0]) - 1]

        solution = solve_ivp(
            diff_eqn_eq_zgas_1,
            t_span,
            ini_m_z_gas_eq,
            args=[ini_m_gas_eq[0], m_dot_cg_val[j], halo_mass[j], z_star_val],
            max_step=h,
        )
        m_z_g = solution.y[0][len(solution.y[0]) - 1]

        solution = solve_ivp(
            diff_eqn_zstar_2,
            t_span,
            ini_m_z_star_eq,
            args=[ini_m_z_gas_eq[0]],
            max_step=h,
        )
        m_z_s = solution.y[0][len(solution.y[0]) - 1]

        ini_m_gas_eq = [m_g]
        ini_m_star_eq = [m_s]
        ini_m_z_gas_eq = [m_z_g]
        ini_m_z_star_eq = [m_z_s]

        if ini_m_gas_eq[0] < 0.0:
            ini_m_gas_eq[0] = 0.0
            ini_m_z_gas_eq[0] = 0.0

        if ini_m_star_eq[0] < 0.0:
            ini_m_star_eq[0] = 0.0
            ini_m_z_star_eq[0] = 0.0

        if ini_m_z_gas_eq[0] < 0.0:
            ini_m_z_gas_eq[0] = 0.0

        if ini_m_z_star_eq[0] < 0.0:
            ini_m_z_star_eq[0] = 0.0

        if ini_m_dust_eq[0] < 0.0:
            ini_m_dust_eq[0] = 0.0

        if ini_m_star_eq[0] == 0.0:
            z_star_val = 0.0

        elif ini_m_star_eq[0] > 0.0:
            z_star_val = ini_m_z_star_eq[0] / ini_m_star_eq[0]

        m_g_eq_val_1 = np.append(m_g_eq_val_1, ini_m_gas_eq)
        m_star_eq_val_1 = np.append(m_star_eq_val_1, ini_m_star_eq)
        m_z_g_eq_val_1 = np.append(m_z_g_eq_val_1, ini_m_z_gas_eq)
        m_z_star_eq_val_1 = np.append(m_z_star_eq_val_1, ini_m_z_star_eq)
        m_dust_eq_val_1 = np.append(m_dust_eq_val_1, ini_m_dust_eq)

    print(len(m_z_g_eq_val_1))
    print(len(m_z_star_eq_val_1))

    np.savetxt(
        f"C:/Users/Anand Menon/Documents/ICRAR Stuff/Results/No UV No Delay/mh{no}_data/gas/tree_{i}.txt",
        m_g_eq_val_1,
        delimiter=" ",
    )
    np.savetxt(
        f"C:/Users/Anand Menon/Documents/ICRAR Stuff/Results/No UV No Delay/mh{no}_data/star/tree_{i}.txt",
        m_star_eq_val_1,
        delimiter=" ",
    )
    np.savetxt(
        f"C:/Users/Anand Menon/Documents/ICRAR Stuff/Results/No UV No Delay/mh{no}_data/z_gas/tree_{i}.txt",
        m_z_g_eq_val_1,
        delimiter=" ",
    )
    np.savetxt(
        f"C:/Users/Anand Menon/Documents/ICRAR Stuff/Results/No UV No Delay/mh{no}_data/z_star/tree_{i}.txt",
        m_z_star_eq_val_1,
        delimiter=" ",
    )

    plt.semilogy(cosmic_time, halo_mass, label="Halo mass")
    plt.semilogy(cosmic_time, halo_mass_rate, label="Halo mass rate")

    plt.semilogy(cosmic_time, m_g_eq_val_1, label="Gas mass values")
    plt.semilogy(cosmic_time, m_star_eq_val_1, label="Stellar mass values")
    plt.semilogy(cosmic_time, m_z_g_eq_val_1, label="Gas metallicity values")
    plt.semilogy(cosmic_time, m_z_star_eq_val_1, label="Stellar metallicity")
    plt.semilogy(cosmic_time, m_dust_eq_val_1, label="Dust mass")

    plt.ylim(10**-3, 10**11)
    plt.legend()

    plt.show()
