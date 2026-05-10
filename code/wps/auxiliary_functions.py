"""
Auxiliary functions shared across multiple working papers (WP).
Contains common functions for CR3BP analysis, including potential gradients,
Hessians, Jacobian matrices, and solar sail calculations.
"""

import numpy as np


def nablaU(mu, x, y=0, z=0):
    """
    Computes the gradient of the effective potential U at (x, y, z) for a given mu.
    Returns a tuple (dUdx, dUdy, dUdz).
    
    Parameters
    ----------
    mu : float
        Mass ratio parameter (0 < mu < 1)
    x, y, z : float
        Position coordinates in the rotating frame
        
    Returns
    -------
    tuple
        (dUdx, dUdy, dUdz) - partial derivatives of the potential
    """
    r1 = np.sqrt((x + mu)**2 + y**2 + z**2)
    r2 = np.sqrt((x - (1 - mu))**2 + y**2 + z**2)

    if mu == 0:
        r2 = np.inf 

    dUdx = -(1 - mu) * (x + mu) / r1**3 - mu * (x - (1 - mu)) / r2**3 + x
    dUdy = -(1 - mu) * y / r1**3 - mu * y / r2**3 + y
    dUdz = -(1 - mu) * z / r1**3 - mu * z / r2**3

    return dUdx, dUdy, dUdz


def dUdx_colinear(x, mu):
    """
    Computes derivative of potential for collinear Lagrange points (y = z = 0).
    
    Parameters
    ----------
    x : float or array
        Position coordinate along x-axis
    mu : float
        Mass ratio parameter
        
    Returns
    -------
    float or array
        dU/dx at (x, 0, 0)
    """
    nablaU_x, _, _ = nablaU(mu, x, 0, 0)
    return nablaU_x 


def hessianU(mu, x, y=0):
    """
    Computes the second-order partial derivatives of the effective potential U at (x, y).
    Returns (Uxx, Uxy, Uyy).
    
    Parameters
    ----------
    mu : float
        Mass ratio parameter
    x, y : float
        Position coordinates in the rotating frame
        
    Returns
    -------
    tuple
        (Uxx, Uxy, Uyy) - second partial derivatives of the potential
    """
    r1 = np.sqrt((x + mu)**2 + y**2)
    r2 = np.sqrt((x - (1 - mu))**2 + y**2)

    if mu == 0:
        r2 = np.inf

    Uxx = (1
           - (1 - mu) / r1**3
           - mu / r2**3
           + 3 * (1 - mu) * (x + mu)**2 / r1**5
           + 3 * mu * (x - (1 - mu))**2 / r2**5)

    Uyy = (1
           - (1 - mu) / r1**3
           - mu / r2**3
           + 3 * (1 - mu) * y**2 / r1**5
           + 3 * mu * y**2 / r2**5)

    Uxy = (3 * (1 - mu) * (x + mu) * y / r1**5
           + 3 * mu * (x - (1 - mu)) * y / r2**5)

    return Uxx, Uxy, Uyy


def jacobian(mu, x, y=0):
    """
    Returns the 4x4 Jacobian matrix J of the linearised CR3BP equations of motion 
    evaluated at position (x, y).
    
    Parameters
    ----------
    mu : float
        Mass ratio parameter
    x, y : float
        Position coordinates
        
    Returns
    -------
    numpy.ndarray
        4x4 Jacobian matrix of the linearized equations of motion
    """
    Uxx, Uxy, Uyy = hessianU(mu, x, y)

    J = np.array([
        [  0,     0,    1,  0],
        [  0,     0,    0,  1],
        [Uxx, Uxy,    0,  2],
        [Uxy, Uyy,   -2,  0]
    ])

    return J


def jacobian_sail(mu, beta, x, y=0):
    """
    Returns the 4x4 Jacobian matrix J of the linearised augmented CR3BP equations 
    of motion (with solar sail) evaluated at position (x, y).
    
    Parameters
    ----------
    mu : float
        Mass ratio parameter
    beta : float
        Solar sail lightness number
    x, y : float
        Position coordinates
        
    Returns
    -------
    numpy.ndarray
        4x4 Jacobian matrix of the linearized augmented CR3BP equations
    """
    r1 = np.sqrt((x + mu)**2 + y**2)

    Uxx, Uxy, Uyy = hessianU(mu, x, y)
    Uxx += -2*beta*(1-mu) / r1**3

    J = np.array([
        [  0,     0,    1,  0],
        [  0,     0,    0,  1],
        [Uxx, Uxy,    0,  2],
        [Uxy, Uyy,   -2,  0]
    ])

    return J


def beta_sail(x, y, MU):
    """
    Computes the required solar-sail lightness number to shift a Lagrange point.
    
    Parameters
    ----------
    x, y : float
        Position of the artificial Lagrange point
    MU : float
        Mass ratio parameter
        
    Returns
    -------
    float
        Solar sail lightness number (beta)
    """
    r1x, r1y = x + MU, y
    r1_sq = r1x**2 + r1y**2
    dUdx, dUdy, _ = nablaU(MU, x, y)
    grad_mag = np.sqrt(dUdx**2 + dUdy**2)
    r1_mag = np.sqrt(r1_sq)
    r1_dot_n = (r1x*dUdx + r1y*dUdy) / (r1_mag * grad_mag) 

    with np.errstate(divide='ignore', invalid='ignore'):
        b = -(r1_sq / (1 - MU)) * grad_mag / r1_dot_n
    return b
