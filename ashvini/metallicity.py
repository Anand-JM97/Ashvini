# -*- coding: utf-8 -*-
"""
Code for metallicity evolution

@author: Anand Menon, Balu Sreedhar, Chris Power
"""

import numpy as np
from astropy.cosmology import Planck15 as cosmo
import astropy.units as u

from utils import omega_m, omega_b, omega_L, H_0

from io import mdot_h, m_h
import utils as utils


# Metallicity parameters

accreting_gas_metallicity = 10 ** (-3)
stellar_metallicity_yield = 0.06
wind_metallicity_enhancement_factor = 1
