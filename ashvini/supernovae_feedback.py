# -*- coding: utf-8 -*-
"""
Code for supernova feedback

@author: Anand Menon
"""

import numpy as np

import utils as utils
import star_formation as sf

epsilon_p = 5
pi_fid = 1


def metallicity_function(stellar_metallicity, m=0.1, s=0.01, a=1, b=0.25):
    function_value = (
        np.exp(-np.logaddexp(0, -(stellar_metallicity - m) / s)) * (b - a) + a
    )
    return function_value


def eta(redshift, halo_mass, stellar_metallicity):
    eta_p = (
        epsilon_p
        * pi_fid
        * ((10**11.5) / halo_mass ** (1 / 3))
        * ((9 / (1 + redshift)) ** (1 / 2))
        * metallicity_function(stellar_metallicity)
    )
    return eta_p


def wind_mass_evolution_rate(redshift, gas_mass, halo_mass, star_formation_rate_for_winds, stellar_metallicity):
    wind_mass_rate = (
        metallicity_function(stellar_metallicity)
        * eta(redshift, halo_mass)
        * star_formation_rate_for_winds
    )
    return wind_mass_rate
