# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 11:47:28 2024

@author: Anand Menon
"""

import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import Planck15 as cosmo
from astropy.cosmology import units as cu
from astropy.cosmology import z_at_value
import astropy.units as u

H_0=cosmo.H0 #in km / (Mpc s)
H_0=H_0.to(u.Gyr**(-1)) #in 1/Gyr

A=0.030

omega_m=cosmo.Om0
omega_b=cosmo.Ob0
omega_L=cosmo.Ode0

alfa=0.79
pi=np.pi

plt.rcParams['figure.dpi'] = 300

#t=cosmo.age(z)

#z=z_at_value(cosmo.age,t)

def t(z):
    t_val=cosmo.age(z)
    return t_val

def z(t):
    z_val=z_at_value(cosmo.age,t)
    return z_val

def H(z):
    H=cosmo.H(z)
    return H


def evolve_galaxies():
    """This will be the main function. We should have this call star formation, supernovae feedback, and such defined
    in the other files. Ideally this should be the only the function that is defined in this file.
    """

    return 1



