import numpy as np
from coeffs_discretization import a02c_std, a02d_std

def apply_scheme(f, dx, coeffs, axis=0):
    """Apply a finite difference scheme to a f field with a dx step on a given axis"""
    deriv = np.zeros_like(f)
    stencil_size = len(coeffs)
    half_span = stencil_size // 2

    if axis == 0:  # Direction r
        f[0, :, :] = f[1, :, :]
        for i in range(half_span, f.shape[0] - half_span):
            deriv[i, :, :] = sum(coeffs[j] * f[i + j - half_span, :, :] for j in range(stencil_size)) / dx

    elif axis == 1:  # Direction θ
        for j in range(half_span, f.shape[1] - half_span):
            deriv[:, j, :] = sum(coeffs[k] * f[:, j + k - half_span, :] for k in range(stencil_size)) / dx

    elif axis == 2:  # Direction z
        for k in range(half_span, f.shape[2] - half_span):
            deriv[:, :, k] = sum(coeffs[l] * f[:, :, k + l - half_span] for l in range(stencil_size)) / dx

    else:
        raise ValueError("L'axe spécifié doit être 0 (r), 1 (θ) ou 2 (z).")

    return deriv

def compute_derivative(f, dx, scheme="3_points", axis=0):
    """Calcule la dérivée avec le schéma choisi et gère les bords et la singularité en r=0."""
    deriv = np.zeros_like(f)
    N = f.shape[axis]  # Taille selon l'axe choisi

    # Sélection du schéma de 3 points (plus stable)
    cent_coeffs, dec_coeffs = a02c_std, a02d_std

    half_span = len(cent_coeffs) // 2
    deriv_slice = [slice(None)] * f.ndim
    deriv_slice[axis] = slice(half_span, -half_span)
    deriv[tuple(deriv_slice)] = apply_scheme(f, dx, cent_coeffs, axis)[tuple(deriv_slice)]

    # Gestion de la singularité en r = 0
    if axis == 0:
        deriv[0, :, :] = deriv[1, :, :]

    # Gestion des bords avec schémas décentrés
    for i in range(half_span):
        idx_low = [slice(None)] * f.ndim
        idx_high = [slice(None)] * f.ndim

        idx_low[axis] = i
        idx_high[axis] = -(i + 1)

        deriv[tuple(idx_low)] = sum(dec_coeffs[j] * f[tuple(idx_low[:axis] + [min(i + j, N - 1)] + idx_low[axis + 1:])]
                                    for j in range(len(dec_coeffs))) / dx
        deriv[tuple(idx_high)] = -sum(dec_coeffs[j] * f[tuple(idx_high[:axis] + [max(-(i + j + 1), -N)] + idx_high[axis + 1:])]
                                     for j in range(len(dec_coeffs))) / dx

    return deriv
