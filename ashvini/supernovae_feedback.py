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

H_0=cosmo.H0 #in km / (Mpc s)
H_0=H_0.to(u.Gyr**(-1)) #in 1/Gyr

omega_m=cosmo.Om0
omega_b=cosmo.Ob0
omega_L=cosmo.Ode0

def t(z):
    """
    Function to convert redshift to cosmic time.
    Args:
        z (float): Parameter representing redshift.

    Returns:
        Float: The comsic time value.
    """
    t_val=cosmo.age(z)
    return t_val

def z(t):
    """
    Function to convert cosmic time to redshift.
    Args:
        t (float): Parameter representing cosmic time.

    Returns:
        Float: The redshift value.
    """
    z_val=z_at_value(cosmo.age,t)
    return z_val