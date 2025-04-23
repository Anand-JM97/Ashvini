# -*- coding: utf-8 -*-
import numpy as np
from astropy.cosmology import Planck15 as cosmo
import astropy.units as u

H_0 = cosmo.H0  # in km / (Mpc s)
H_0 = cosmo.H0.to(u.Gyr ** (-1))  # in 1/Gyr

omega_m = cosmo.Om0
omega_b = cosmo.Ob0
omega_L = cosmo.Ode0

# PARAMETERS FOR THE MODEL:

z_rei = 7
gamma = 15
omega = 2

beta = z_rei * ((np.log(1.82 * (10**3) * np.exp(-0.63 * z_rei) - 1)) ** (-1 / gamma))
c_omega = 2 ** (omega / 3) - 1


def s(x, y):
    """
    A step function used in the expression for accretion suppression due to UV background.
    Args:
        x, y (float): Two parameters representing any physical quantity.

    Returns:
        The value of the function s(x,y) based on the argument of the parameters.
    """
    s_val = (1 + (2 ** (y / 3) - 1) * (x ** (-y))) ** (-3 / y)
    return s_val


def M_c(z):
    """
    Characteristic mass scale for reionization (Okamoto et al. 2008).
    Args:
        z (float): Parameter for redshift.

    Returns:
        Float: The characteristic mass scale at which the baryon fraction is suppressed by a factor of two,
        compared to the universal value, because of background UV.
    """
    M_val = 1.69 * (10**10) * (np.exp(-0.63 * z) / (1 + np.exp((z / beta) ** gamma)))
    return M_val


def mu_c(z, m_halo):
    mu = m_halo / M_c(z)
    return mu


def X(z, m_halo):
    M_omega = (M_c(z) / m_halo) ** (omega)
    X_val = (3 * c_omega * M_omega) / (1 + c_omega * M_omega)
    return X_val


def epsilon(z):
    part2 = (
        (gamma * (z ** (gamma - 1)) / (beta**gamma))
        * (np.exp((z / beta) ** gamma))
        / ((1 + np.exp((z / beta) ** gamma)) ** 2)
    )
    epsilon = (0.63) / (1 + np.exp((z / beta) ** gamma)) + part2
    return epsilon


def uv_suppression(z_val, m_halo, mdot_halo):
    values = []
    for i in range(0, len(z_val)):
        if z_val[i] > 10:
            test_value = 1
        else:
            test_value = s(mu_c(z_val[i], m_halo[i]), omega) * (
                (1 + X(z_val[i], m_halo[i]))
                - 2
                * epsilon(z_val[i])
                * m_halo[i]
                * X(z_val[i], m_halo[i])
                * (1 + z_val[i])
                * cosmo.H(z_val[i]).value
                / mdot_halo[i]
            )
        if test_value < 0:
            test_value = 0.0
        values = np.append(values, [test_value])
    return values
