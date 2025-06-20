from . import utils as utils
from . import supernovae_feedback as sn
from .star_formation import star_formation_rate

from .run_params import PARAMS

IGM_metallicity = PARAMS.metals.Z_IGM
metallicity_yield = PARAMS.metals.Z_yield


def evolve_gas_metals(
    t,
    gas_metal_mass,
    gas_mass,
    gas_accretion_rate,
    halo_mass,
    stellar_metallicity,
    past_sfr,
    kind="delayed",
):
    redshift = utils.z_at_time(t)
    present_sfr = star_formation_rate(t, gas_mass=gas_mass)
    wind_sfr = past_sfr

    if kind == "no":
        wind_sfr = 0.0
    if kind == "instantaneous":
        wind_sfr = present_sfr

    if gas_mass == 0.0:
        gas_metals_rate = (
            (IGM_metallicity * gas_accretion_rate)  # Enriched gas accreting from IGM
            + (metallicity_yield * wind_sfr)  # Delayed enrichment of ISM by dying stars
        )
    else:
        gas_metals_rate = (
            (IGM_metallicity * gas_accretion_rate)  # Enriched gas accreting from IGM
            - (
                gas_metal_mass * present_sfr / gas_mass
            )  # Removal from ISM during star formation
            + (metallicity_yield * wind_sfr)  # Delayed enrichment of ISM by dying stars
            - (
                sn.mass_loading_factor(redshift, halo_mass, stellar_metallicity)
                * (gas_metal_mass / gas_mass)
                * wind_sfr
            )  # Delayed removal from ISM by SN feedback
        )

    return gas_metals_rate


def evolve_stars_metals(
    t,
    gas_metals,
    gas_mass,
):
    # TODO: We can simply call starformation_rate here- Should SFR use gas_metals as argument instead of gas_mass?
    if gas_mass == 0.0:
        stars_metals_rate = 0
    else:
        stars_metals_rate = (
            star_formation_rate(t, gas_mass=gas_mass)
            * gas_metals
            / gas_mass  # Heavy elements captured during star formation
        )
    return stars_metals_rate
