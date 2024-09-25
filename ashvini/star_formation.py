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

H_0=cosmo.H0 #in km / (Mpc s)
H_0=H_0.to(u.Gyr**(-1)) #in 1/Gyr

omega_m=cosmo.Om0
omega_b=cosmo.Ob0
omega_L=cosmo.Ode0

e_ff=0.015

#t=cosmo.age(z) --For testing redshift to age of the Universe conversion

#z=z_at_value(cosmo.age,t) --For testing age of the Universe to redshift conversion

def t(z):
    t_val=cosmo.age(z)
    return t_val

def z(t):
    z_val=z_at_value(cosmo.age,t)
    return z_val

def t_ff(z):
    t_ff_val=0.141/(H_0*np.sqrt((omega_m*((1+z)**3))+omega_L))
    return t_ff_val

def m_dot_star(m_gas,z):
    m_dot_star_val=(e_ff*m_gas/t_ff(z))
    return m_dot_star_val


