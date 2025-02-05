# -*- coding: utf-8 -*-
"""
Code for star formation

@author: Anand Menon
"""

import numpy as np
from astropy.cosmology import Planck15 as cosmo
from astropy.cosmology import units as cu
from astropy.cosmology import z_at_value
import astropy.units as u

H_0 = cosmo.H0  # in km / (Mpc s)
H_0 = H_0.to(u.Gyr ** (-1))  # in 1/Gyr

omega_m = cosmo.Om0
omega_b = cosmo.Ob0
omega_L = cosmo.Ode0

e_ff = 0.015


def t(z):
    """
    Function to convert redshift to cosmic time.
    Args:
        z (float): Parameter representing redshift.

    Returns:
        Float: The comsic time value.
    """
    t_val = cosmo.age(z)
    return t_val


def z(t):
    """
    Function to convert cosmic time to redshift.
    Args:
        t (float): Parameter representing cosmic time.

    Returns:
        Float: The redshift value.
    """
    return z_at_value(cosmo.age, t)


def time_freefall(z):
    return 0.141 / (H_0 * np.sqrt((omega_m * ((1 + z) ** 3)) + omega_L))


def star_formation_rate(m_gas, z):
    m_dot_star_val = e_ff * m_gas / t_ff(z)
    return m_dot_star_val
