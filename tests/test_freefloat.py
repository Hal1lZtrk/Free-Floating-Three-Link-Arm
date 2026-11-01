"""Verification of the free-floating three-link model.

1. Conservation laws (strict consequence of "no external forces").
2. Energy balance  dT/dt = sum(tau * q_dot).
3. Independent check: planar chain compared with a symbolic Lagrangian (sympy).
4. Consistency of the parameters / geometry (link lengths, stored initial poses).
5. Numerical convergence (tolerance study).
"""
import numpy as np
import pytest

from freefloat import ThreeLinkArm
from freefloat import params as P
from freefloat.params import Link



@pytest.fixture(scope="module")
def traj():
    return ThreeLinkArm().simulate(10.0, 0.01)


def test_linear_momentum_is_conserved(traj):
    assert np.abs(traj.lin_momentum).max() < 1e-8


def test_angular_momentum_is_conserved(traj):
    assert np.abs(traj.ang_momentum).max() < 1e-8


def test_centre_of_mass_does_not_move(traj):
    assert np.abs(traj.com - traj.com[0]).max() < 1e-9


def test_energy_balance(traj):
    assert np.abs(traj.kinetic - traj.work).max() < 1e-8


def test_rotation_matrices_stay_orthonormal(traj):
    err = max(np.abs(R.T @ R - np.eye(3)).max() for Rs in traj.rot for R in Rs)
    assert err < 1e-9


def test_joint_points_are_shared_by_both_links():
    """Joint j seen from link j and from link j+1 must be the same world point at all times."""
    arm = ThreeLinkArm()
    traj = arm.simulate(4.0, 0.05)
    for k in range(len(traj.t)):
        for j in range(2):
            Ra, pa = arm.Ta[j]
            Rb, pb = arm.Tb[j]
            from_j = traj.pos[k, j] + traj.rot[k, j] @ pa
            from_j1 = traj.pos[k, j + 1] + traj.rot[k, j + 1] @ pb
            assert np.linalg.norm(from_j - from_j1) < 1e-10
            # joint axes coincide as well
            assert np.linalg.norm(traj.rot[k, j] @ Ra[:, 2] - traj.rot[k, j + 1] @ Rb[:, 2]) < 1e-10


def test_relative_twist_is_pure_rotation_about_the_joint_axis():
    arm = ThreeLinkArm()
    traj = arm.simulate(3.0, 0.05)
    for k in range(len(traj.t)):
        for j in range(2):
            # twist of link j+1 minus twist of link j transported to link j+1 must equal s*theta_dot
            Vs, Xs = arm.twists(traj.twist[k, 0], traj.theta[k], traj.theta_dot[k])
            rel = traj.twist[k, j + 1] - Xs[j] @ traj.twist[k, j]
            assert np.allclose(rel, arm.s[j] * traj.theta_dot[k, j], atol=1e-10)


def test_initial_geometry_matches_the_stored_initial_poses():
    """Chain geometry (from the input/output frame data) vs the initial poses of the assignment (rounded)."""
    arm = ThreeLinkArm()
    y = arm.initial_state()
    Ps, Rs = arm.poses(y[0:3], y[3:12].reshape(3, 3), np.zeros(2))
    for i, link in enumerate(P.LINKS):
        assert np.linalg.norm(Ps[i] - link.P0) < 5e-4          # stored values are rounded
        assert np.abs(Rs[i] - link.R0).max() < 1e-9


def test_parameters_are_physically_sensible():
    """Link lengths equal the UR5 link lengths; inertia tensors are symmetric positive definite."""
    lengths = []
    for l in P.LINKS:
        p_in = -l.in_rot.T @ l.in_pos
        p_out = -l.out_pos
        lengths.append(np.linalg.norm(p_in - p_out))
        assert np.allclose(l.J, l.J.T)
        assert np.all(np.linalg.eigvalsh(l.J) > 0)
        assert l.m > 0
    assert np.isclose(lengths[0], 0.425, atol=1e-9)
    assert np.isclose(lengths[1], 0.39225, atol=1e-9)
    assert P.TORQUE_AMPLITUDE == (1.0, 1.0) and P.TORQUE_OMEGA == (5.0, 5.0)


def test_tolerance_study():
    arm = ThreeLinkArm()
    ref = arm.simulate(10.0, 0.05, rtol=1e-12, atol=1e-13)
    for rtol in (1e-6, 1e-8, 1e-10):
        tr = arm.simulate(10.0, 0.05, rtol=rtol, atol=rtol * 0.1)
        assert np.abs(tr.q - ref.q).max() < 200 * rtol


# --------------------------------------------------------------------------------------
# Independent check: planar chain vs symbolic Lagrangian
# --------------------------------------------------------------------------------------
def _planar_links():
    """Three planar links: all joint axes parallel to z, CoG offsets in the x-y plane."""
    I = np.eye(3)
    def J(izz):
        return np.diag([0.01, 0.02, izz])
    return (
        Link("a", 2.0, J(0.30), I, np.array([-0.30, 0.05, 0.0]), np.array([0.25, 0.00, 0.0]),
             np.zeros(3), I),
        Link("b", 1.3, J(0.12), I, np.array([-0.20, -0.04, 0.0]), np.array([0.15, 0.03, 0.0]),
             np.zeros(3), I),
        Link("c", 0.8, J(0.05), I, np.array([-0.10, 0.02, 0.0]), np.array([0.05, 0.00, 0.0]),
             np.zeros(3), I),
    )


def _lagrange_planar(links, sign, amp, omega):
    """Symbolic Lagrangian of the planar free chain -> function (state, t) -> generalised accel."""
    import sympy as sp
    t = sp.symbols("t")
    x, y, f1, a1, a2 = [sp.Function(n)(t) for n in ("x", "y", "f1", "a1", "a2")]
    coords = [x, y, f1, a1, a2]
    ths = [a1, a2]
    phis = [f1, f1 + a1, f1 + a1 + a2]

    def R2(ph):
        return sp.Matrix([[sp.cos(ph), -sp.sin(ph)], [sp.sin(ph), sp.cos(ph)]])

    # CoG of link 1 at (x, y); in-plane vectors from CoG to joint points (CoG-frame coordinates)
    out = [sp.Matrix(-l.out_pos[:2]) for l in links]            # output frame origin in c coords
    inn = [sp.Matrix(-l.in_rot.T[:2, :2] @ l.in_pos[:2]) for l in links]   # input frame origin in c coords
    pos = [sp.Matrix([x, y])]
    for j in range(2):
        joint = pos[j] + R2(phis[j]) * out[j]
        pos.append(joint - R2(phis[j + 1]) * inn[j + 1])
    Tk = 0
    for i, l in enumerate(links):
        v = pos[i].diff(t)
        Tk += sp.Rational(1, 2) * l.m * (v.T * v)[0] + sp.Rational(1, 2) * l.J[2, 2] * phis[i].diff(t) ** 2
    Qs = [0, 0, 0] + [sp.Float(sign[j]) * sp.Float(amp[j]) * sp.cos(sp.Float(omega[j]) * t) for j in range(2)]
    eqs = []
    for c, Qc in zip(coords, Qs):
        eqs.append(sp.diff(Tk.diff(c.diff(t)), t) - Tk.diff(c) - Qc)
    qdd = [c.diff(t, 2) for c in coords]
    A, b = sp.linear_eq_to_matrix(eqs, qdd)
    q = sp.symbols("q0:5")
    qd = sp.symbols("d0:5")
    rep = {}
    for c, qs, qds in zip(coords, q, qd):
        rep[c.diff(t)] = qds
    sub = {}
    for c, qs in zip(coords, q):
        sub[c] = qs
    def subs_all(expr):
        e = expr.subs({c.diff(t): d for c, d in zip(coords, qd)}).subs(sub)
        return e
    A_f = sp.lambdify((q, qd, t), subs_all(A), "numpy")
    b_f = sp.lambdify((q, qd, t), subs_all(b), "numpy")
    pos_f = sp.lambdify(q, [subs_all(p) for p in pos], "numpy")
    return A_f, b_f


def test_planar_case_against_symbolic_lagrangian():
    sympy = pytest.importorskip("sympy")
    links = _planar_links()
    sign = np.array([-1.0, 1.0])
    amp, om = np.array([1.3, 0.7]), np.array([5.0, 3.0])
    arm = ThreeLinkArm(links=links, joint_sign=sign, amplitude=amp, omega=om)
    A_f, b_f = _lagrange_planar(links, sign, amp, om)
    rng = np.random.default_rng(3)
    for _ in range(6):
        x, y, f1 = rng.normal(size=3)
        a = rng.uniform(-1.5, 1.5, 2)
        xd, yd, fd = rng.normal(size=3)
        ad_ = rng.normal(size=2)
        tt = rng.uniform(0, 2)
        # Lagrange accelerations of (x, y, f1, a1, a2)
        qv = np.array([x, y, f1, a[0], a[1]])
        qd = np.array([xd, yd, fd, ad_[0], ad_[1]])
        A = np.array(A_f(qv, qd, tt), dtype=float)
        b = np.array(b_f(qv, qd, tt), dtype=float).reshape(-1)
        qdd_L = np.linalg.solve(A, b)
        # our model: link 1 body twist from (xd, yd, fd), world->body
        R1 = np.eye(3)
        R1[:2, :2] = [[np.cos(f1), -np.sin(f1)], [np.sin(f1), np.cos(f1)]]
        V1 = np.concatenate([R1.T @ np.array([xd, yd, 0.0]), [0, 0, fd]])
        a1_body, thdd, _ = arm.forward_dynamics(tt, V1, a, ad_)
        # body acceleration -> world acceleration of CoG 1:  d/dt (R v) = R (a + w x v)
        acc_w = R1 @ (a1_body[:3] + np.cross(V1[3:], V1[:3]))
        assert np.allclose(acc_w[:2], qdd_L[:2], atol=1e-9)
        assert np.isclose(a1_body[5], qdd_L[2], atol=1e-9)
        assert np.allclose(thdd, qdd_L[3:], atol=1e-9)
