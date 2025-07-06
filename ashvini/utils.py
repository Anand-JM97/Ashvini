import numpy as np
import h5py
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u

from scipy.interpolate import interp1d

Omega_m = cosmo.Om0
Omega_b = cosmo.Ob0
Omega_L = cosmo.Ode0


def read_trees(file_path, mass_bin):
    """
    Reads halo masses, halo growth rates, and redshifts for a mass bin.

    Parameters:
    - h5_file_path: str, path to the HDF5 file
    - mass_bin_float: float, e.g., 1e6, 5e7

    Returns:
    - m_halo: np.ndarray
    - halo_accretion_rate: np.ndarray
    - redshift: np.ndarray
    """
    mass_bin = float(mass_bin)
    exponent = int(np.log10(mass_bin))
    mantissa = int(mass_bin / 10**exponent)
    group_name = f"{mantissa:02d}e{exponent:02d}"

    with h5py.File(file_path, "r") as f:
        group = f[group_name]
        m_halo = group["halo_masses"][:]
        halo_accretion_rate = group["halo_growth_rates"][:]
        redshift = f["redshifts"][:]

    return m_halo, halo_accretion_rate, redshift


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
