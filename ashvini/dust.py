import numpy as np

from . import supernovae_feedback as sn
from . import utils as utils
from .star_formation import star_formation_rate

from .run_params import PARAMS

Y_d = PARAMS.dust.dust_yield
Gamma = PARAMS.dust.dust_gamma
M_crit = PARAMS.dust.m_crit
M_swept = PARAMS.dust.m_swept


def update_dust_reservoir(
    t,
    dust_mass,
    gas_mass,
    halo_mass,
    past_sfr,
    past_stars_mass,
    stellar_metallicity,
):
    growth_rate = Y_d * past_sfr

    if past_sfr <= 0:
        SNe_rate = 0.0
    else:
        SNe_rate = Gamma * past_stars_mass / past_sfr
    dust_loading = 1 - np.exp(-gas_mass / M_crit)
    destruction_rate = M_swept * SNe_rate * dust_loading

    redshift = utils.z_at_time(t)

    if gas_mass <= 0:
        wind_loss = 0.0
    else:
        wind_loss = (
            (dust_mass / gas_mass)
            * sn.mass_loading_factor(redshift, halo_mass, stellar_metallicity)
            * past_sfr
        )

    dust_mass_evolution_rate = growth_rate - destruction_rate - wind_loss

    return dust_mass_evolution_rate
