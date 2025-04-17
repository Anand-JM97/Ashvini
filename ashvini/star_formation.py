# -*- coding: utf-8 -*-
"""
Code for star formation

@author: Anand Menon
"""

import numpy as np
from astropy.cosmology import Planck15 as cosmo
import astropy.units as u

import utils as utils

H_0 = cosmo.H0  # in km / (Mpc s)
H_0 = H_0.to(u.Gyr ** (-1))  # in 1/Gyr

omega_m = cosmo.Om0
omega_b = cosmo.Ob0
omega_L = cosmo.Ode0

e_ff = 0.015


def time_freefall(z):
    return 0.141 / (H_0 * np.sqrt((omega_m * ((1 + z) ** 3)) + omega_L)).value


def star_formation_rate(
    t,  # cosmic time
    y=0,  # sfr; dummy variable
    gas_mass=0,
):
    """
    Equation 2 in Menon et al 2024
    """

    redshift = utils.z_at_time(t)

    star_formation_rate = (e_ff / time_freefall(redshift)) * gas_mass
    return np.asarray(star_formation_rate)
