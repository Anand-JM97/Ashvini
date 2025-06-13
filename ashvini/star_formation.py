# -*- coding: utf-8 -*-

e_ff = 0.015

import numpy as np

import utils as utils
from utils import Hubble_time


def time_freefall(redshift):
    return 0.141 * Hubble_time(redshift)


def star_formation_rate(
    t,  # cosmic time
    gas_mass,
):
    """
    Equation 2 in Menon et al 2024
    """

    redshift = utils.z_at_time(t)

    star_formation_rate = (e_ff / time_freefall(redshift)) * gas_mass
    return np.asarray(star_formation_rate)
