# -*- coding: utf-8 -*-
"""
Code for supernova feedback

@author: Anand Menon
"""

import numpy as np
from astropy.cosmology import Planck15 as cosmo
from astropy.cosmology import units as cu
from astropy.cosmology import z_at_value
import astropy.units as u

from io import mdot_h, m_h
import star_formation as sf

H_0 = cosmo.H0  # in km / (Mpc s)
H_0 = H_0.to(u.Gyr ** (-1))  # in 1/Gyr

omega_m = cosmo.Om0
omega_b = cosmo.Ob0
omega_L = cosmo.Ode0

# Supernova wind specific variables
# pi_fid=1
# epsilon_p=5

def metallicity_function(stellar_metallicity, m=0.1, s=0.01, a=1, b=0.25):
    function_value = (
        np.exp(
            -np.logaddexp(0, -(stellar_metallicity - m) / s)
        )
        * (b-a)
        + a
    )
    return function_value

def eta(z, m_halo, pi_fid=1, epsilon_p=5):
    eta_p = (
        epsilon_p
        * pi_fid
        * (((10**11.5) / m_h(z, m_halo)) ** (1 / 3))
        * ((9 / (1 + z)) ** (1 / 2))
    )
    return eta_p

def m_dot_wind(m_gas, z, m_halo, stellar_metallicity):
    m_dot_wind_val = (
        eta(z, m_halo)
        * metallicity_function(stellar_metallicity)
        * sf.star_formation_rate(m_gas, z, m_halo)
    )
    return m_dot_wind_val
