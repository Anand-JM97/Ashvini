import numpy as np

from .utils import Hubble_time, z_at_time

from .run_params import PARAMS

e_ff = PARAMS.sf.efficiency


def time_freefall(redshift):
    return 0.141 * Hubble_time(redshift)


def star_formation_rate(
    t,  # cosmic time
    gas_mass,
):
    """
    Equation 2 in Menon et al 2024
    """

    redshift = z_at_time(t)

    star_formation_rate = (e_ff / time_freefall(redshift)) * gas_mass
    return np.asarray(star_formation_rate)
