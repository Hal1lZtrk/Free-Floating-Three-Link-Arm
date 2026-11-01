"""Small spatial-algebra toolbox (screw theory), twist ordering ``[v; omega]``.

Linear part first.

* A *twist* ``V = [v; w]`` is the velocity of a body, expressed in a frame.
* A *wrench* ``W = [F; tau]`` is a force/torque pair, expressed in a frame.
* ``ad(V)`` is the Lie bracket (motion cross product) used for velocity derivatives.
* ``adjoint(R, p)`` maps a twist expressed in frame ``b`` to the same twist expressed in
  frame ``a`` when ``b`` has pose ``(R, p)`` in ``a``.  Wrenches transform with its transpose
  (power invariance ``W_b . V_b = W_a . V_a``).
"""
import numpy as np


def skew(a):
    a = np.asarray(a, dtype=float).reshape(3)
    return np.array([[0.0, -a[2], a[1]],
                     [a[2], 0.0, -a[0]],
                     [-a[1], a[0], 0.0]])


def rot_x(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def rot_z(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def pose_inverse(R, p):
    return R.T, -R.T @ p


def pose_compose(Ra, pa, Rb, pb):
    """(Ra, pa) * (Rb, pb): pose of c in a, given b in a and c in b."""
    return Ra @ Rb, pa + Ra @ pb


def adjoint(R, p):
    """6x6 twist transformation  V^a = Ad V^b  for a frame b with pose (R, p) in a."""
    A = np.zeros((6, 6))
    A[:3, :3] = R
    A[:3, 3:] = skew(p) @ R
    A[3:, 3:] = R
    return A


def ad(V):
    """Motion cross product matrix:  ad(U) W = U x W  (twists, [v; w] ordering)."""
    v, w = V[:3], V[3:]
    A = np.zeros((6, 6))
    A[:3, :3] = skew(w)
    A[:3, 3:] = skew(v)
    A[3:, 3:] = skew(w)
    return A


def spatial_inertia(m, J):
    M = np.zeros((6, 6))
    M[:3, :3] = m * np.eye(3)
    M[3:, 3:] = J
    return M


def orthonormalise(R):
    """Closest rotation matrix (polar decomposition via SVD)."""
    U, _, Vt = np.linalg.svd(R)
    Q = U @ Vt
    if np.linalg.det(Q) < 0:
        U[:, -1] *= -1
        Q = U @ Vt
    return Q
