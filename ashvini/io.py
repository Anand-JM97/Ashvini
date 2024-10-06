# -*- coding: utf-8 -*-
"""
Code for input/output

@author: Anand Menon
"""

import numpy as np
from astropy.cosmology import Planck15 as cosmo
from astropy.cosmology import units as cu
from astropy.cosmology import z_at_value
import astropy.units as u

H_0=cosmo.H0 #in km / (Mpc s)
H_0=H_0.to(u.Gyr**(-1)) #in 1/Gyr

A=0.030
alfa=0.79

z_0=5

#HALO MASS ANALYTICAL FUNCTIONS

def mdot_h(z,m_h0_value):
    """
    Function that tracks the Halo growth rate.
    Args:
        z (float): Redshift parameter.
        m_h0_value (float): Halo mass initialization value at z_0
    """
    halo_rate=A*m_h0_value*np.exp(-alfa*(z-z_0))*(1+z)**(5/2)
    return halo_rate

def m_h(z,m_h0_value):
    """
    Function that gives the Halo mass.
    Args:
        z (float): Redshift parameter.
        m_h0_value (float): Halo mass initialization value at z_0
    """
    m_h_val=m_h0_value*np.exp(-alfa*(z-z_0))
    return m_h_val

#MERGER TREE FUNCTIONS


