import numpy as np
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u

from scipy.interpolate import interp1d

Omega_m = cosmo.Om0
Omega_b = cosmo.Ob0
Omega_L = cosmo.Ode0


def read_trees():
    dir_name = "./data/inputs/"

    m_halo = np.loadtxt(dir_name + "halo_mass_0.txt", usecols=(0,))
    halo_accretion_rate = np.loadtxt(dir_name + "halo_mass_rate_0.txt", usecols=(0,))
    redshift = np.loadtxt(dir_name + "redshifts.txt", usecols=(0,))

    return np.asarray(m_halo), np.asarray(halo_accretion_rate), np.asarray(redshift)


# --- Precompute time and redshift interpolation ---
_z_vals = np.linspace(0, 50, 10000)
_t_vals = cosmo.age(_z_vals).value  # Gyr

_z_interp = interp1d(
    _t_vals[::-1], _z_vals[::-1], kind="cubic", fill_value="extrapolate"
)
_t_interp = interp1d(_z_vals, _t_vals, kind="cubic", fill_value="extrapolate")

_Hubble_time_vals = (1 / cosmo.H(_z_vals)).to(u.Gyr).value
_Hubble_interp = interp1d(
    _z_vals, _Hubble_time_vals, kind="cubic", fill_value="extrapolate"
)


def time_at_z(z):
    """Convert redshift to cosmic time (Gyr)."""
    return np.asarray(_t_interp(z))


def z_at_time(t):
    """Convert cosmic time (Gyr) to redshift."""
    return np.asarray(_z_interp(t))


def Hubble_time(z):
    """Return Hubble time (Gyr) at redshift z."""
    return np.asarray(_Hubble_interp(z))
