import numpy as np
from astropy.cosmology import Planck18 as cosmo
from astropy.cosmology import z_at_value
import astropy.units as u

H_0 = cosmo.H0  # in km / (Mpc s)
H_0 = H_0.to(u.Gyr ** (-1))  # in 1/Gyr

omega_m = cosmo.Om0
omega_b = cosmo.Ob0
omega_L = cosmo.Ode0


def read_trees():
    dir_name = "../data/inputs/"

    m_halo = np.loadtxt(dir_name + "halo_mass_0.txt", usecols=(0,))
    halo_accretion_rate = np.loadtxt(dir_name + "halo_mass_rate_0.txt", usecols=(0,))
    redshift = np.loadtxt(dir_name + "redshifts.txt", usecols=(0,))

    return np.asarray(m_halo), np.asarray(halo_accretion_rate), np.asarray(redshift)


def time_at_z(z):
    """
    Function to convert redshift to cosmic time.
    Args:
        z (float): Parameter representing redshift.

    Returns:
        Float: The comsic time value.
    """
    return cosmo.age(z).value


def z_at_time(time):
    """
    Function to convert cosmic time to redshift.
    Args:
        t (float): Parameter representing cosmic time.

    Returns:
        Float: The redshift value.
    """
    return z_at_value(cosmo.age, time * u.Gyr).value


def Hubble_time(z):
    """
    Hubble time is the inverse of the Hubble constant.
    Args:
        z (float): Parameter for redshift.

    Returns:
        Float: The value of the Hubble constant for the redshift value entered as the argument.
    """
    return (1 / cosmo.H(z)).value
