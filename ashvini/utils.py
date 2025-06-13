import numpy as np
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u

from scipy.interpolate import interp1d

Omega_m = cosmo.Om0
Omega_b = cosmo.Ob0
Omega_L = cosmo.Ode0


def read_trees():
    dir_name = "../data/inputs/"

    m_halo = np.loadtxt(dir_name + "halo_mass_1.txt", usecols=(0,))
    halo_accretion_rate = np.loadtxt(dir_name + "halo_mass_rate_1.txt", usecols=(0,))
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


z_vals = np.linspace(0, 50, 10000)
t_vals = cosmo.age(z_vals).value  # Gyr
z_interp = interp1d(t_vals[::-1], z_vals[::-1], kind="cubic", fill_value="extrapolate")


def z_at_time(t):  # t in Gyr
    return float(z_interp(t))


def Hubble_time(z):
    """
    Hubble time is the inverse of the Hubble constant.
    Args:
        z (float): Parameter for redshift.

    Returns:
        Float: The value of the Hubble constant for the redshift value entered as the argument.
    """
    return (1 / cosmo.H(z)).to(u.Gyr).value
