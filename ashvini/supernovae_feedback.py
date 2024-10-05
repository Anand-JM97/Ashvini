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

from io import mdot_h,m_h
import star_formation as sf

H_0=cosmo.H0 #in km / (Mpc s)
H_0=H_0.to(u.Gyr**(-1)) #in 1/Gyr

omega_m=cosmo.Om0
omega_b=cosmo.Ob0
omega_L=cosmo.Ode0

#Supernova wind specific variables
pi_fid=1
epsilon_p=5

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

def eta(z,m_halo):   
    eta_p=epsilon_p*pi_fid*(((10**11.5)/m_h(z,m_halo))**(1/3))*((9/(1+z))**(1/2))
    return eta_p

def m_dot_wind(m_gas,z,m_halo):
    m_dot_wind_val=eta(z,m_halo,epsilon_p)*sf.m_dot_star(m_gas,z,m_halo)
    return m_dot_wind_val

