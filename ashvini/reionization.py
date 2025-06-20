import numpy as np
import astropy.units as u

from astropy.cosmology import Planck15 as cosmo

from .run_params import PARAMS

z_rei = PARAMS.reion.z_reion
gamma = PARAMS.reion.gamma
omega = PARAMS.reion.omega


H_0 = cosmo.H0  # in km / (Mpc s)
H_0 = cosmo.H0.to(u.Gyr ** (-1))  # in 1/Gyr


beta = z_rei * ((np.log(1.82 * (10**3) * np.exp(-0.63 * z_rei) - 1)) ** (-1 / gamma))
c_omega = 2 ** (omega / 3) - 1


def s(x, y):
    """
    A step function used in the expression for accretion suppression due to UV background.
    Args:
        x, y (float): Two parameters representing any physical quantity.

    Returns:
        The value of the function s(x,y) based on the argument of the parameters.
    """
    s_val = (1 + (2 ** (y / 3) - 1) * (x ** (-y))) ** (-3 / y)
    return s_val


def M_c(z):
    """
    Characteristic mass scale for reionization (Okamoto et al. 2008).
    Args:
        z (float): Parameter for redshift.

    Returns:
        Float: The characteristic mass scale at which the baryon fraction is suppressed by a factor of two,
        compared to the universal value, because of background UV.
    """
    M_val = 1.69 * (10**10) * (np.exp(-0.63 * z) / (1 + np.exp((z / beta) ** gamma)))
    return M_val


def mu_c(z, m_halo):
    mu = m_halo / M_c(z)
    return mu


def X(z, m_halo):
    M_omega = (M_c(z) / m_halo) ** (omega)
    X_val = (3 * c_omega * M_omega) / (1 + c_omega * M_omega)
    return X_val


def epsilon(z):
    part2 = (
        (gamma * (z ** (gamma - 1)) / (beta**gamma))
        * (np.exp((z / beta) ** gamma))
        / ((1 + np.exp((z / beta) ** gamma)) ** 2)
    )
    epsilon = (0.63) / (1 + np.exp((z / beta) ** gamma)) + part2
    return epsilon


def uv_suppression(z_val, m_halo, mdot_halo):
    z_val = np.asarray(z_val)
    m_halo = np.asarray(m_halo)
    mdot_halo = np.asarray(mdot_halo)

    suppression = np.ones_like(z_val)

    mask = z_val <= 10

    z_masked = z_val[mask]
    mh_masked = m_halo[mask]
    mdot_masked = mdot_halo[mask]

    mu_vals = mu_c(z_masked, mh_masked)
    x_vals = X(z_masked, mh_masked)
    eps_vals = epsilon(z_masked)
    H_vals = cosmo.H(z_masked).value

    suppressed = s(mu_vals, omega) * (
        (1 + x_vals)
        - 2 * eps_vals * mh_masked * x_vals * (1 + z_masked) * H_vals / mdot_masked
    )

    suppression[mask] = np.maximum(suppressed, 0.0)

    return suppression
