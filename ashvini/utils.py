from astropy.cosmology import Planck15 as cosmo
import astropy.units as u


def time(z):
    """
    Function to convert redshift to cosmic time.
    Args:
        z (float): Parameter representing redshift.

    Returns:
        Float: The comsic time value.
    """
    return cosmo.age(z)


def z_at_time(t):
    """
    Function to convert cosmic time to redshift.
    Args:
        t (float): Parameter representing cosmic time.

    Returns:
        Float: The redshift value.
    """
    return z_at_value(cosmo.age, t)


def Hubble_time(z):
    """
    Hubble time is the inverse of the Hubble constant.
    Args:
        z (float): Parameter for redshift.

    Returns:
        Float: The value of the Hubble constant for the redshift value entered as the argument.
    """
    return 1 / cosmo.H(z)
