# -*- coding: utf-8 -*-
"""
Code for supernova feedback

@author: Anand Menon
"""

import numpy as np
from astropy.cosmology import Planck15 as cosmo
import astropy.units as u

from utils import omega_m, omega_b, omega_L, H_0

import utils as utils
import star_formation as sf


def metallicity_function(stellar_metallicity, m=0.1, s=0.01, a=1, b=0.25):
    function_value = (
        np.exp(-np.logaddexp(0, -(stellar_metallicity - m) / s)) * (b - a) + a
    )
    return function_value


def eta(z, m_halo, pi_fid=1, epsilon_p=5):
    eta_p = (
        epsilon_p
        * pi_fid
        * (((10**11.5) / m_halo) ** (1 / 3))
        * ((9 / (1 + z)) ** (1 / 2))
    )
    return eta_p


def m_dot_wind(m_gas, z, m_halo, stellar_metallicity):
    m_dot_wind_val = (
        metallicity_function(stellar_metallicity)
        * eta(z, m_halo)
        * sf.star_formation_rate(m_gas, z, m_halo)
    )
    return m_dot_wind_val
