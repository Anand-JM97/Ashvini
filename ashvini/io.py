# -*- coding: utf-8 -*-
"""
Code for input/output

@author: Anand Menon
"""

import numpy as np


def read_trees():
    dir_name = "../data/inputs/"

    m_halo = np.loadtxt(dir_name + "halo_mass_0.txt", usecols=(0,))
    halo_accretion_rate = np.loadtxt(dir_name + "halo_mass_rate_0.txt", usecols=(0,))
    redshift = np.loadtxt(dir_name + "redshifts.txt", usecols=(0,))

    return m_halo, halo_accretion_rate, redshift
